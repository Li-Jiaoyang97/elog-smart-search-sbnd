#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_retrieval.py
-----------------
Score retrieval methods against a hand-labelled query set, so model / search
changes are compared with numbers instead of eyeballing.

Methods:
    mxbai, mpnet          dense search on the live Qdrant collections (raw top-k)
    mxbai-prod, mpnet-prod  what /api/search actually returns: top-10, then the
                          adaptive score threshold from QdrantSearch
    bm25                  keyword search over the local text cache
    hybrid-mxbai, hybrid-mpnet  dense + bm25 merged with reciprocal rank fusion
    rerank-<method>       <method>'s top --rerank_top (50) re-scored by a cross-encoder
                          (models/bge-reranker-base), which reads query and entry text
                          together; the rest keeps its order. Default: rerank-hybrid-*.
bm25/hybrid/rerank need the text cache (DebugScripts/build_text_cache.py).

Pool bias: labels come from pooling the methods' top hits (--label), so a new
method that surfaces entries nobody saw while labelling gets them counted as
"not relevant" even when they are. unj@10 = average number of top-10 entries
outside that pool (mxbai/mpnet/bm25/hybrid top --top); if it is high for a
method, its scores are understated. --show_unjudged lists them; --judge walks
through them asking y/n (random order, method names hidden so you judge the
entry, not the method) and writes the answers back into each query line:
"yes" into "relevant", "no" into "not_relevant" so they aren't asked again.

Metrics (averaged over queries):
    R@k     recall@k - fraction of the relevant entries found in the top k
            (can't reach 1 when a query has more than k relevant entries)
    cR@10   capped recall@10 - relevant found in the top 10 / min(#relevant, 10),
            so 1 = the best any method could do. The closest match to what the
            OpenWebUI LLM gets: it reads all 10 results, order matters little.
    MRR     1 / rank of the first relevant entry (0 if not in top 10)
    nDCG    nDCG@10 - rewards relevant entries ranked higher (main summary score)

Error bars: a bootstrap over queries (resample the query set with replacement,
recompute the means, repeat --bootstrap times) gives a 95% interval per method,
i.e. how much the score could move with a different but similar set of
queries. The paired table compares every method with --baseline on the SAME
resamples - that is what decides "is A really better than B": if the interval
of the difference excludes 0, the gap is unlikely to be luck of the queries.
W/T/L counts the queries where the method beats / ties / loses to the baseline.

Query file - JSON lines, blank lines and lines starting with '#' ignored:
    {"query": "CRT readout dropouts after firmware update", "relevant": [31002, 31057]}
    {"query": "timing cable change in 2026", "relevant": [39272], "start_date": "2026", "end_date": "2026"}
start_date/end_date are optional and work like /api/search's: "2026", "2026-03",
"2026-03-18" or a full ISO datetime, a partial date covering the whole period.
Every method then only sees entries in that range (dense via a Qdrant filter,
bm25 by masking the cache). --ignore_dates scores without them, to see what
the filter is worth.

Building the query set: write a query, then run --label on it. It pools the
top hits of every method, shows each candidate's text and asks y/n, then
appends {"query", "relevant", "style"} to eval_queries.jsonl. Pooling across
methods keeps the labels from favouring whichever model you happened to look
at. (--pool just prints the same candidates without asking.)

Usage (from anywhere, inside the venv):
    python DebugScripts/eval_retrieval.py --label "PMT high voltage trip"
    python DebugScripts/eval_retrieval.py --label "timing cable change in 2026" --start_date 2026 --end_date 2026
    python DebugScripts/eval_retrieval.py --pool "PMT high voltage trip"
    python DebugScripts/eval_retrieval.py --queries DebugScripts/eval_queries.jsonl
    python DebugScripts/eval_retrieval.py --queries ... --methods mxbai,bm25,hybrid-mxbai --per_query
    python DebugScripts/eval_retrieval.py --queries ... --ignore_dates
    python DebugScripts/eval_retrieval.py --queries ... --baseline hybrid-mxbai --bootstrap 5000
    python DebugScripts/eval_retrieval.py --queries ... --methods hybrid-mpnet,rerank-hybrid-mpnet --show_unjudged
    python DebugScripts/eval_retrieval.py --queries DebugScripts/eval_queries.jsonl --judge
"""

import os
import io
import re
import sys
import json
import logging
import argparse
import time
import operator
import contextlib
from datetime import datetime

import numpy as np

ELOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, ELOG_DIR)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

# entry_body/rerank_text/Reranker are shared with the API, so rerank-* here is
# exactly what /api/search does in production.
from QdrantSearch.QdrantSearch import SemanticSearch, date_range, entry_body, rerank_text, Reranker
from build_text_cache import load_cache, CACHE_FILE

# QdrantSearch sets the root logger to DEBUG at import time.
logging.getLogger().setLevel(logging.WARNING)

# Same model/collection/prefix pairing as searcher_configs in RESTAPI_Qdrant_HTML.py.
DENSE = {
    'mxbai': {
        'model_path': os.path.join(ELOG_DIR, 'models', 'mxbai-embed-large-v1'),
        'collection': 'Elog_collection_1024_METADATA',
        'query_prefix': 'Represent this sentence for searching relevant passages: ',
    },
    'mpnet': {
        'model_path': os.path.join(ELOG_DIR, 'models', 'all-mpnet-base-v2'),
        'collection': 'Elog_collection_768_METADATA',
        'query_prefix': '',
    },
    # "Text cleanup v2": same models and same entries, re-embedded from the
    # deduplicated prefix (DebugScripts/text_v2.py), built by
    # DebugScripts/build_v2_collections.py. Only the embedded text differs from
    # the pair above, so mxbai vs mxbai-v2 isolates the text change.
    'mxbai-v2': {
        'model_path': os.path.join(ELOG_DIR, 'models', 'mxbai-embed-large-v1'),
        'collection': 'Elog_collection_1024_V2',
        'query_prefix': 'Represent this sentence for searching relevant passages: ',
    },
    'mpnet-v2': {
        'model_path': os.path.join(ELOG_DIR, 'models', 'all-mpnet-base-v2'),
        'collection': 'Elog_collection_768_V2',
        'query_prefix': '',
    },
}
V2_METHODS = ['mxbai-v2', 'mpnet-v2']
RERANK_MODEL = os.path.join(ELOG_DIR, 'models', 'bge-reranker-base')
DEFAULT_QUERIES = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'eval_queries.jsonl')
PROD_LIMIT = 10  # ToolSearchRequest.limit default
RRF_K = 60       # standard reciprocal-rank-fusion constant
KS = (5, 10, 20)
CI_COLS = ('cR@10', 'MRR', 'nDCG')      # metrics shown with bootstrap intervals
PAIRED_COLS = ('cR@10', 'MRR', 'nDCG')  # metrics compared against --baseline

STOPWORDS = set("""a an the and or of to in on at for from by with is are was were be been
it its this that these those as not no but if then so do does did has have had
i we you he she they our your their me us them what which who when where why how""".split())
TOKEN_RE = re.compile(r"[a-z0-9]+(?:[._\-/][a-z0-9]+)*")


def bm25_tokenize(text):
    """Lowercase word tokens, keeping compounds like 'pmt-12' or 'v1.2' whole
    AND adding their parts, so 'PMT 12' still matches 'pmt-12'."""
    tokens = []
    for tok in TOKEN_RE.findall(text.lower()):
        parts = re.split(r"[._\-/]", tok)
        if len(parts) > 1:
            tokens.append(tok)
        tokens.extend(p for p in parts if p not in STOPWORDS)
    return tokens


RANGE_OPS = {'gt': operator.gt, 'gte': operator.ge, 'lt': operator.lt, 'lte': operator.le}


def in_range(stamp, dates):
    """Is an RFC3339 'YYYY-MM-DDTHH:MM:SSZ' stamp inside date_range() bounds?
    Both use that one fixed format, so plain string comparison orders them."""
    return bool(stamp) and all(RANGE_OPS[k](stamp, v) for k, v in dates.items())


def ecl_to_rfc3339(date_str):
    """Cache dates are raw ECL ('03/18/2026 10:04:46'); match what ingestion
    stores in Qdrant (text_to_vector.date_to_rfc3339: same clock, 'Z' added)."""
    try:
        return f"{datetime.strptime(date_str, '%m/%d/%Y %H:%M:%S'):%Y-%m-%dT%H:%M:%SZ}"
    except ValueError:
        return ''


class Retrievers:
    """Lazily builds each method's backend; every method returns a ranked list of entry IDs.
    `dates` (from date_range()) restricts every method to entries created in that range."""

    def __init__(self, depth, rerank_top=50):
        self.depth = depth
        self.rerank_top = rerank_top
        self.client = QdrantClient(host=os.environ.get('QDRANT_HOST', 'localhost'), port=6333)
        self._encoders = {}
        self._bm25 = None
        self._cache = None
        self._reranker = None
        self._memo = {}
        self.rerank_seconds = []  # cross-encoder time per reranked query

    def cache(self):
        if self._cache is None:
            self._cache = load_cache()
            if not self._cache:
                sys.exit(f"bm25/hybrid/rerank need the text cache - run build_text_cache.py first ({CACHE_FILE}).")
        return self._cache

    def _encoder(self, name):
        if name not in self._encoders:
            self._encoders[name] = SentenceTransformer(DENSE[name]['model_path'], device='cpu')
        return self._encoders[name]

    def dense_hits(self, name, query, limit, dates=None):
        cfg = DENSE[name]
        vector = self._encoder(name).encode(cfg['query_prefix'] + query).tolist()
        query_filter = models.Filter(must=[models.FieldCondition(
            key='creation_date', range=models.DatetimeRange(**dates))]) if dates else None
        return self.client.search(
            collection_name=cfg['collection'], query_vector=vector, limit=limit,
            search_params=models.SearchParams(exact=True), query_filter=query_filter,
        )

    def dense(self, name, query, dates=None):
        return [h.id for h in self.dense_hits(name, query, self.depth, dates)]

    def prod(self, name, query, dates=None):
        hits = self.dense_hits(name, query, PROD_LIMIT, dates)
        scores = [h.score for h in hits]
        if not scores:
            return []
        # adaptive_threshold_min never touches self; it just prints its cutoffs.
        with contextlib.redirect_stdout(io.StringIO()):
            cutoff = SemanticSearch.adaptive_threshold_min(None, scores)
        return [h.id for h in hits if h.score >= cutoff]

    def bm25(self, query, dates=None):
        if self._bm25 is None:
            cache = self.cache()
            self._bm25_ids = sorted(cache)
            self._bm25_stamps = [ecl_to_rfc3339(cache[i]['creation_date']) for i in self._bm25_ids]
            self._bm25 = BM25Okapi([bm25_tokenize(cache[i]['text']) for i in self._bm25_ids])
        scores = self._bm25.get_scores(bm25_tokenize(query))
        if dates:
            # Filter after scoring, so word rarity (IDF) still comes from the
            # whole logbook - same as a Qdrant filter on a sparse vector would.
            scores = scores * np.array([in_range(s, dates) for s in self._bm25_stamps])
        top = np.argsort(-scores)[:self.depth]
        return [self._bm25_ids[k] for k in top if scores[k] > 0]

    def hybrid(self, name, query, dates=None):
        fused = {}
        for ranked in (self.dense(name, query, dates), self.bm25(query, dates)):
            for rank, eid in enumerate(ranked, start=1):
                fused[eid] = fused.get(eid, 0.0) + 1.0 / (RRF_K + rank)
        return sorted(fused, key=fused.get, reverse=True)[:self.depth]

    def rerank(self, base, query, dates=None):
        ranked = self.run(base, query, dates)
        head, tail = ranked[:self.rerank_top], ranked[self.rerank_top:]
        if not head:
            return ranked
        if self._reranker is None:
            self._reranker = Reranker(RERANK_MODEL)
        cache = self.cache()
        start = time.perf_counter()
        scores = self._reranker.score(query, [rerank_text(cache.get(e, {})) for e in head])
        self.rerank_seconds.append(time.perf_counter() - start)
        return [head[k] for k in np.argsort(-scores, kind='stable')] + tail

    def run(self, method, query, dates=None):
        # Memoised: rerank-X reuses X's ranking, and the unjudged-pool check
        # reuses everything already run for the same query.
        key = (method, query, tuple(sorted(dates.items())) if dates else None)
        if key not in self._memo:
            self._memo[key] = self._run(method, query, dates)
        return self._memo[key]

    def _run(self, method, query, dates=None):
        if method.startswith('rerank-'):
            return self.rerank(method[len('rerank-'):], query, dates)
        if method in DENSE:
            return self.dense(method, query, dates)
        if method.endswith('-prod'):
            return self.prod(method[:-len('-prod')], query, dates)
        if method == 'bm25':
            return self.bm25(query, dates)
        if method.startswith('hybrid-'):
            return self.hybrid(method[len('hybrid-'):], query, dates)
        raise ValueError(f"Unknown method '{method}'")


ALL_METHODS = ['mxbai', 'mpnet', 'mxbai-v2', 'mpnet-v2', 'mxbai-prod', 'mpnet-prod',
               'bm25', 'hybrid-mxbai', 'hybrid-mpnet',
               'rerank-hybrid-mxbai', 'rerank-hybrid-mpnet']
POOL_BASE = ['mxbai', 'mpnet', 'bm25', 'hybrid-mxbai', 'hybrid-mpnet']  # what --label pooled from


def valid_method(m):
    """ALL_METHODS, plus rerank-<any first-stage method> (e.g. rerank-mxbai)."""
    base = m[len('rerank-'):] if m.startswith('rerank-') else None
    return m in ALL_METHODS or (base in ALL_METHODS and not base.endswith('-prod')
                                and not base.startswith('rerank-'))



def metrics(ranked, relevant):
    rel = set(relevant)
    out = {f'R@{k}': len(rel.intersection(ranked[:k])) / len(rel) for k in KS}
    out['cR@10'] = len(rel.intersection(ranked[:10])) / min(len(rel), 10)
    first =next((r for r, eid in enumerate(ranked[:10], start=1) if eid in rel), None)
    out['MRR'] = 1.0 / first if first else 0.0
    dcg = sum(1.0 / np.log2(r + 1) for r, eid in enumerate(ranked[:10], start=1) if eid in rel)
    idcg = sum(1.0 / np.log2(r + 1) for r in range(1, min(len(rel), 10) + 1))
    out['nDCG'] = dcg / idcg
    return out


def load_queries(path):
    queries = []
    with open(path) as f:
        for n, line in enumerate(f, start=1):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            try:
                q = json.loads(line)
            except json.JSONDecodeError as e:
                sys.exit(f"{path}:{n}: not valid JSON ({e}). Each query must be on its own line.")
            if not q.get('relevant'):
                sys.exit(f"{path}:{n}: query has no 'relevant' IDs")
            q['relevant'] = [int(i) for i in q['relevant']]
            q['not_relevant'] = [int(i) for i in q.get('not_relevant', [])]
            both = set(q['relevant']) & set(q['not_relevant'])
            if both:
                sys.exit(f"{path}:{n}: IDs in both 'relevant' and 'not_relevant': {sorted(both)}")
            q['_line'] = n  # for --judge to write answers back to the right line
            try:
                q['dates'] = date_range(q.get('start_date'), q.get('end_date'))
            except ValueError as e:
                sys.exit(f"{path}:{n}: {e}")
            queries.append(q)
    return queries


def date_label(start_date, end_date):
    return f"[{start_date or '...'} to {end_date or '...'}]"


def pool_base_methods():
    return [m for m in POOL_BASE if os.path.exists(CACHE_FILE) or m in DENSE]


def unjudged_for(retrievers, q, methods, dates, pool_top):
    """{entry_id: [methods]} for top-10 entries of `methods` that were never judged:
    not marked relevant / not_relevant, and outside the approximate --label pool
    (the pool_base_methods' top pool_top, which the labeller saw and passed on)."""
    judged = set(q['relevant']) | set(q['not_relevant'])
    judged |= {e for m in pool_base_methods() for e in retrievers.run(m, q['query'], dates)[:pool_top]}
    unj = {}
    for m in methods:
        for e in retrievers.run(m, q['query'], dates)[:10]:
            if e not in judged:
                unj.setdefault(e, []).append(m)
    return unj


def bootstrap_report(totals, methods, baseline, n_boot, seed=0):
    """Print bootstrap 95% intervals per method, then paired differences vs baseline.
    Every method is scored on the same resampled query sets, so the differences
    keep the per-query pairing (a hard query is hard for everyone)."""
    n_queries = len(totals[methods[0]])
    vals = {m: {c: np.array([t[c] for t in totals[m]]) for c in CI_COLS + PAIRED_COLS}
            for m in methods}
    # Row b of idx is one resampled query set; x[idx].mean(axis=1) is that set's mean.
    idx = np.random.default_rng(seed).integers(0, n_queries, size=(n_boot, n_queries))

    def interval(x):
        return np.percentile(x[idx].mean(axis=1), [2.5, 97.5])

    width = max(len(m) for m in methods) + 2
    print(f"\n95% intervals (bootstrap over the {n_queries} queries, {n_boot} resamples):")
    print(f"{'method':<{width}}" + "".join(f"{c:>21}" for c in CI_COLS))
    for m in methods:
        cells = []
        for c in CI_COLS:
            lo, hi = interval(vals[m][c])
            cells.append(f"{vals[m][c].mean():.3f} [{lo:.2f}, {hi:.2f}]")
        print(f"{m:<{width}}" + "".join(f"{x:>21}" for x in cells))

    print(f"\nPaired difference vs {baseline} (method - {baseline}; '*' = interval excludes 0):")
    print(f"{'method':<{width}}" + "".join(f"{'d' + c:>26}" for c in PAIRED_COLS)
          + f"{'W/T/L nDCG':>13}")
    for m in methods:
        if m == baseline:
            continue
        cells = []
        for c in PAIRED_COLS:
            diff = vals[m][c] - vals[baseline][c]
            lo, hi = interval(diff)
            star = '*' if lo > 0 or hi < 0 else ' '
            cells.append(f"{diff.mean():+.3f} [{lo:+.2f}, {hi:+.2f}]{star}")
        d = vals[m]['nDCG'] - vals[baseline]['nDCG']
        wtl = f"{(d > 1e-9).sum()}/{(abs(d) <= 1e-9).sum()}/{(d < -1e-9).sum()}"
        print(f"{m:<{width}}" + "".join(f"{x:>26}" for x in cells) + f"{wtl:>13}")


def evaluate(retrievers, queries, methods, per_query, ignore_dates=False,
             baseline=None, n_boot=2000, pool_top=10, show_unjudged=False):
    # A relevant ID that isn't in the collection can never be retrieved - flag it.
    all_rel = sorted({int(i) for q in queries for i in q['relevant']})
    stamps = {p.id: (p.payload or {}).get('creation_date', '') for p in retrievers.client.retrieve(
        DENSE['mxbai']['collection'], ids=all_rel, with_payload=['creation_date'])}
    missing = [i for i in all_rel if i not in stamps]
    if missing:
        print(f"WARNING: relevant IDs not in Qdrant (unretrievable, count as misses): {missing}\n")
    # Likewise one labelled relevant but created outside its query's date range.
    if not ignore_dates:
        for q in queries:
            outside = [i for i in q['relevant'] if q['dates'] and i in stamps
                       and not in_range(stamps[i], q['dates'])]
            if outside:
                print(f"WARNING: {q['query'][:50]!r}: relevant IDs outside its date range "
                      f"{date_label(q.get('start_date'), q.get('end_date'))} (count as misses): "
                      f"{[(i, stamps[i][:10]) for i in outside]}\n")

    pool_base = pool_base_methods()
    totals = {m: [] for m in methods}
    first_ranks = []
    unjudged = []  # per query: {entry_id: [methods with it in their top 10]}
    for q in queries:
        dates = None if ignore_dates else q['dates']
        rel = set(q['relevant'])
        unj = unjudged_for(retrievers, q, methods, dates, pool_top)
        row = {}
        for m in methods:
            ranked = retrievers.run(m, q['query'], dates)
            t = metrics(ranked, q['relevant'])
            t['unj@10'] = sum(1 for e in ranked[:10] if e in unj)
            totals[m].append(t)
            row[m] = next((r for r, eid in enumerate(ranked, start=1) if eid in rel), None)
        first_ranks.append(row)
        unjudged.append(unj)

    cols = [f'R@{k}' for k in KS] + ['cR@10', 'MRR', 'nDCG', 'unj@10']
    width = max(len(m) for m in methods) + 2
    n_dated = sum(1 for q in queries if q['dates'])
    note = (f", {n_dated} with a date range" + (" - IGNORED (--ignore_dates)" if ignore_dates else "")
            if n_dated else "")
    print(f"{len(queries)} queries{note}\n")
    print(f"{'method':<{width}}" + "".join(f"{c:>8}" for c in cols))
    for m in methods:
        means = {c: np.mean([t[c] for t in totals[m]]) for c in cols}
        print(f"{m:<{width}}" + "".join(f"{means[c]:>8.3f}" for c in cols))
    if retrievers.rerank_seconds:
        print(f"\nreranker: {np.mean(retrievers.rerank_seconds):.2f} s per query to re-score the top "
              f"{retrievers.rerank_top} (CPU, on top of the first-stage search)")

    if n_boot and len(queries) > 1:
        bootstrap_report(totals, methods, baseline, n_boot)

    if per_query:
        print(f"\nRank of first relevant entry per query ('-' = not in top {retrievers.depth}):")
        print(f"{'':<4}" + "".join(f"{m:>{width}}" for m in methods) + "  query")
        for n, (q, row) in enumerate(zip(queries, first_ranks), start=1):
            dated = (f"  {date_label(q.get('start_date'), q.get('end_date'))}"
                     if q['dates'] and not ignore_dates else "")
            print(f"{n:<4}" + "".join(f"{(row[m] or '-'):>{width}}" for m in methods)
                  + f"  {q['query'][:60]}{dated}")

    if show_unjudged:
        cache = retrievers.cache()
        print(f"\nUnjudged entries in a top 10 (not in the top-{pool_top} pool of {', '.join(pool_base)}, "
              f"nor in the query's relevant/not_relevant lists). Judge them with --judge:")
        for n, (q, unj) in enumerate(zip(queries, unjudged), start=1):
            if not unj:
                continue
            print(f"\n{n:<3} {q['query']}")
            for eid, ms in unj.items():
                rec = cache.get(eid, {})
                print(f"    {eid:>6}  {rec.get('creation_date', '')[:10]}  {rec.get('subject', '')[:60]}"
                      f"   ({', '.join(ms)})")
                print(f"            {BASE_URL}/show?e={eid}")


BASE_URL = os.environ.get('ECL_URL', 'https://dbweb1.fnal.gov:8443/ECL/sbnd/E')


def pool_candidates(retrievers, query, methods, top, dates=None):
    """Union of every method's top hits, best-ranked first.
    Returns (ordered IDs, {id: {method: rank}}, {id: payload})."""
    found = {}
    for m in methods:
        for rank, eid in enumerate(retrievers.run(m, query, dates)[:top], start=1):
            found.setdefault(eid, {})[m] = rank
    order = sorted(found, key=lambda e: -sum(1.0 / (RRF_K + r) for r in found[e].values()))
    payloads = {p.id: p.payload for p in retrievers.client.retrieve(
        DENSE['mxbai']['collection'], ids=order, with_payload=True)}
    return order, found, payloads


def pool(retrievers, query, methods, top, start_date=None, end_date=None):
    """Print the pooled candidates for one query, for labelling."""
    dates = date_range(start_date, end_date)
    order, found, payloads = pool_candidates(retrievers, query, methods, top, dates)
    dated = f" {date_label(start_date, end_date)}" if dates else ""
    print(f"Pooled top-{top} from {', '.join(methods)} for: {query!r}{dated}\n")
    for eid in order:
        p = payloads.get(eid, {})
        ranks = " ".join(f"{m}:{r}" for m, r in found[eid].items())
        print(f"{eid:>6}  {p.get('creation_date', '')[:10]}  {p.get('subject', '')[:70]}")
        print(f"        {ranks}   {BASE_URL}/show?e={eid}")


LABEL_HELP = """  y = relevant   n / Enter = not relevant   m = show full text
  o = open in browser   b = back to previous   d = done (skip remaining)   q = quit without saving"""


def label(retrievers, query, methods, top, out_path, style, start_date=None, end_date=None):
    """Walk through the pooled candidates asking y/n, then append the query to out_path."""
    import textwrap
    import webbrowser

    if os.path.exists(out_path):
        existing = [q['query'].strip().lower() for q in load_queries(out_path)]
        if query.strip().lower() in existing:
            print(f"Note: this exact query is already in {out_path}.\n")

    dates = date_range(start_date, end_date)
    cache = load_cache()
    order, found, payloads = pool_candidates(retrievers, query, methods, top, dates)
    dated = f" {date_label(start_date, end_date)}" if dates else ""
    print(f"{len(order)} candidates for: {query!r}{dated}\n{LABEL_HELP}\n")

    verdict = {}
    i = 0
    while i < len(order):
        eid = order[i]
        p = payloads.get(eid, {})
        body = entry_body(cache[eid]['text']) if eid in cache else '(not in text cache)'
        ranks = " ".join(f"{m}:{r}" for m, r in found[eid].items())
        print(f"── [{i + 1}/{len(order)}] {eid}  {p.get('creation_date', '')[:10]}  "
              f"{p.get('top_category', '')}/{p.get('subcategory', '')}   ({ranks})")
        print(f"   {p.get('subject', '') or '(no subject)'}")
        print(textwrap.indent(textwrap.fill(body[:400] + ('...' if len(body) > 400 else ''), 100), '   '))

        while True:
            ans = input(f"   relevant? [{'y' if verdict.get(eid) else 'n'}] ").strip().lower()
            if ans == 'm':
                print(textwrap.indent(textwrap.fill(body, 100), '   '))
            elif ans == 'o':
                webbrowser.open(f"{BASE_URL}/show?e={eid}")
            elif ans in ('y', 'n', ''):
                verdict[eid] = ans == 'y' or (ans == '' and verdict.get(eid, False))
                i += 1
                break
            elif ans == 'b':
                i = max(i - 1, 0)
                break
            elif ans == 'd':
                i = len(order)
                break
            elif ans == 'q':
                print("Quit - nothing saved.")
                return
            else:
                print(LABEL_HELP)
        print()

    relevant = [eid for eid in order if verdict.get(eid)]
    while True:
        extra = input("Other relevant entry IDs not shown above (comma-separated, Enter for none): ").strip()
        try:
            extra_ids = [int(x) for x in re.split(r"[,\s]+", extra) if x]
            break
        except ValueError:
            print("   IDs must be numbers, e.g. 1234, 5678")
    relevant += [x for x in extra_ids if x not in relevant]
    if not relevant:
        print("No relevant entries marked - nothing saved. (Reword the query, or try a larger --top.)")
        return

    record = {'query': query, 'relevant': relevant, 'style': style}
    if start_date:
        record['start_date'] = start_date
    if end_date:
        record['end_date'] = end_date
    line = json.dumps(record)
    print(f"\n{line}")
    if input(f"Append to {out_path}? [Y/n] ").strip().lower() in ('', 'y'):
        # A hand-edited file may lack a trailing newline; appending straight
        # onto its last line would merge two JSON objects into one line.
        needs_newline = False
        if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
            with open(out_path, 'rb') as f:
                f.seek(-1, os.SEEK_END)
                needs_newline = f.read(1) != b'\n'
        with open(out_path, 'a') as f:
            f.write(('\n' if needs_newline else '') + line + '\n')
        print(f"Saved ({len(load_queries(out_path))} queries in file).")
    else:
        print("Not saved.")


JUDGE_HELP = """  y = relevant   n = not relevant   Enter = skip (ask again next time)   m = show full text
  o = open in browser   b = back   d = done with this query   q = save answers so far and stop
  (Ctrl-C = stop WITHOUT saving)"""


def judge(retrievers, path, methods, pool_top, ignore_dates=False, seed=0):
    """Ask y/n about every unjudged top-10 entry (see unjudged_for), then write
    the answers into the query lines: y -> "relevant", n -> "not_relevant"."""
    import random
    import textwrap
    import webbrowser

    queries = load_queries(path)
    cache = retrievers.cache()
    todo = []
    for q in queries:
        unj = list(unjudged_for(retrievers, q, methods, None if ignore_dates else q['dates'], pool_top))
        # Random order and no method names: judge the entry, not who found it.
        random.Random(seed).shuffle(unj)
        if unj:
            todo.append((q, unj))
    total = sum(len(unj) for _, unj in todo)
    if not total:
        print("Nothing to judge - every top-10 entry of these methods is already judged.")
        return
    print(f"{total} unjudged entries across {len(todo)} queries (top 10 of {', '.join(methods)}).\n"
          f"Shown in random order without method names.\n{JUDGE_HELP}")

    answers = []  # (query dict, {eid: True/False})
    stop = False
    try:
        for q, unj in todo:
            dated = f"  {date_label(q.get('start_date'), q.get('end_date'))}" if q['dates'] else ""
            print(f"\n{'═' * 100}\nQUERY: {q['query']}{dated}")
            known = [cache[e]['subject'] for e in q['relevant'] if cache.get(e, {}).get('subject')][:3]
            print(f"  already relevant: {len(q['relevant'])} entries"
                  + (f", e.g. {'; '.join(s[:50] for s in known)}" if known else ""))
            verdict = {}
            i = 0
            while i < len(unj):
                eid = unj[i]
                rec = cache.get(eid, {})
                body = entry_body(rec['text']) if rec else '(not in text cache)'
                current = {True: 'y', False: 'n'}.get(verdict.get(eid), 'skip')
                print(f"\n── [{i + 1}/{len(unj)}] {eid}  {rec.get('creation_date', '')[:10]}  "
                      f"{rec.get('top_category', '')}/{rec.get('subcategory', '')}")
                print(f"   {rec.get('subject') or '(no subject)'}")
                print(textwrap.indent(textwrap.fill(body[:400] + ('...' if len(body) > 400 else ''), 100), '   '))
                while True:
                    ans = input(f"   relevant? [{current}] ").strip().lower()
                    if ans == 'm':
                        print(textwrap.indent(textwrap.fill(body, 100), '   '))
                    elif ans == 'o':
                        webbrowser.open(f"{BASE_URL}/show?e={eid}")
                    elif ans in ('y', 'n'):
                        verdict[eid] = ans == 'y'
                        i += 1
                        break
                    elif ans == '':
                        i += 1
                        break
                    elif ans == 'b':
                        i = max(i - 1, 0)
                        break
                    elif ans == 'd':
                        i = len(unj)
                        break
                    elif ans == 'q':
                        stop = True
                        break
                    else:
                        print(JUDGE_HELP)
                if stop:
                    break
            if verdict:
                answers.append((q, verdict))
            if stop:
                break
    except (KeyboardInterrupt, EOFError):
        print("\nStopped - nothing saved.")
        return

    n_yes = sum(v for _, verdict in answers for v in verdict.values())
    n_no = sum(not v for _, verdict in answers for v in verdict.values())
    if not answers:
        print("\nNo answers given - nothing saved.")
        return
    save_judgements(path, answers)
    print(f"\nSaved {n_yes} relevant + {n_no} not relevant across {len(answers)} queries to {path} "
          f"(previous version: {path}.bak). Re-run --queries to see the updated scores.")


def save_judgements(path, answers):
    """Merge --judge answers into the query file in place, touching only the
    answered query lines (comments, order and other lines stay as they are)."""
    import shutil

    with open(path) as f:
        lines = f.readlines()
    for q, verdict in answers:
        n = q['_line']
        obj = json.loads(lines[n - 1])
        if obj.get('query') != q['query']:
            sys.exit(f"{path}:{n} changed while judging (expected query {q['query']!r}) - nothing saved.")
        rel = [int(i) for i in obj['relevant']]
        not_rel = [int(i) for i in obj.get('not_relevant', [])]
        rel += [e for e, ok in verdict.items() if ok and e not in rel]
        not_rel += [e for e, ok in verdict.items() if not ok and e not in not_rel]
        obj['relevant'] = rel
        if not_rel:
            obj['not_relevant'] = not_rel
        lines[n - 1] = json.dumps(obj) + '\n'
    shutil.copy2(path, path + '.bak')
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        f.writelines(lines)
    os.replace(tmp, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--queries', help='JSONL file of {"query": ..., "relevant": [ids]}.')
    mode.add_argument('--pool', metavar='QUERY', help='Show pooled candidates for one query.')
    mode.add_argument('--label', metavar='QUERY',
                      help='Interactively mark pooled candidates relevant, then append the query to --out.')
    parser.add_argument('--out', default=DEFAULT_QUERIES,
                        help='--label: query file to append to (default: %(default)s).')
    parser.add_argument('--style', default='real', help='--label: style tag to record (default: real).')
    parser.add_argument('--methods', default=None,
                        help=f"Comma-separated subset of {','.join(ALL_METHODS)}. "
                             f"Default: all (bm25/hybrid only if the text cache exists, rerank only if "
                             f"the reranker model is also there). Also accepts rerank-<method>, e.g. rerank-mxbai.")
    parser.add_argument('--depth', type=int, default=100,
                        help='Results retrieved per method (and fused, for hybrid).')
    parser.add_argument('--rerank_top', type=int, default=50,
                        help='rerank-*: how many first-stage results the reranker re-scores (default %(default)s).')
    parser.add_argument('--top', type=int, default=10,
                        help='--pool/--label: hits taken from each method; --queries: pool depth assumed '
                             'for unj@10 (match what you labelled with).')
    parser.add_argument('--show_unjudged', action='store_true',
                        help='--queries: list top-10 entries that were never in the labelling pool.')
    parser.add_argument('--judge', action='store_true',
                        help='--queries: ask y/n about the unjudged top-10 entries of --methods and write the '
                             'answers into the query file (backup kept as <file>.bak).')
    parser.add_argument('--per_query', action='store_true',
                        help='Also print the rank of the first relevant hit per query.')
    parser.add_argument('--start_date', help='--pool/--label: only entries created on/after this '
                                             '("2026", "2026-03", "2026-03-18", ISO datetime); saved with the query.')
    parser.add_argument('--end_date', help='--pool/--label: only entries created on/before this '
                                           '(a partial date covers the whole period); saved with the query.')
    parser.add_argument('--ignore_dates', action='store_true',
                        help="--queries: ignore the queries' start_date/end_date, to compare with/without the filter.")
    parser.add_argument('--bootstrap', type=int, default=2000,
                        help='--queries: bootstrap resamples for the error bars (0 = off; default %(default)s).')
    parser.add_argument('--baseline', default=None,
                        help='--queries: method the paired comparison is made against (default: mxbai, '
                             'what production uses, else the first method).')
    args = parser.parse_args()
    try:
        date_range(args.start_date, args.end_date)
    except ValueError as e:
        parser.error(str(e))
    if args.judge and not args.queries:
        parser.error("--judge works on a query file: use it with --queries FILE")

    if args.methods:
        methods = args.methods.split(',')
        unknown = [m for m in methods if not valid_method(m)]
        if unknown:
            parser.error(f"unknown methods {unknown}; choose from {ALL_METHODS} or rerank-<method>")
    else:
        has_cache = os.path.exists(CACHE_FILE)
        has_reranker = has_cache and os.path.isdir(RERANK_MODEL)
        # The V2 collections only exist once build_v2_collections.py has run.
        built = {c.name for c in QdrantClient(
            host=os.environ.get('QDRANT_HOST', 'localhost'), port=6333).get_collections().collections}
        methods = [m for m in ALL_METHODS
                   if (has_reranker if m.startswith('rerank-')
                       else has_cache or not (m == 'bm25' or m.startswith('hybrid-')))]
        methods = [m for m in methods if m not in V2_METHODS or DENSE[m]['collection'] in built]
        if not any(m in V2_METHODS for m in methods):
            print("(no V2 collections - run DebugScripts/build_v2_collections.py to A/B the text change)\n")
        if not has_cache:
            print("(no text cache - skipping bm25/hybrid/rerank methods)\n")
        elif not has_reranker:
            print(f"(no reranker model at {RERANK_MODEL} - skipping rerank methods)\n")

    baseline = args.baseline or ('mxbai' if 'mxbai' in methods else methods[0])
    if baseline not in methods:
        parser.error(f"--baseline {baseline} is not among the evaluated methods {methods}")

    retrievers = Retrievers(depth=args.depth, rerank_top=args.rerank_top)
    # -prod is a subset of its dense method. rerank-* is left out so labelling
    # stays quick and unj@10 (which assumes the POOL_BASE pool) stays accurate;
    # --show_unjudged covers what the reranker brings up.
    pool_methods = [m for m in methods if not m.endswith('-prod') and not m.startswith('rerank-')]
    if args.pool:
        pool(retrievers, args.pool, pool_methods, args.top, args.start_date, args.end_date)
    elif args.label:
        label(retrievers, args.label, pool_methods, args.top, args.out, args.style,
              args.start_date, args.end_date)
    elif args.judge:
        judge(retrievers, args.queries, methods, args.top, args.ignore_dates)
    else:
        evaluate(retrievers, load_queries(args.queries), methods, args.per_query, args.ignore_dates,
                 baseline, args.bootstrap, args.top, args.show_unjudged)


if __name__ == '__main__':
    main()
