#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SBND elog semantic search API, used by OpenWebUI as a tool server.

OpenWebUI reads /openapi.json and offers every endpoint in it to its LLM as a
tool, so only the LLM-facing endpoints are in the schema:
    /api/search    natural-language search, optional date range
    /api/similar   entries similar to a known entry ID (follow-ups, reverts, ...)
    /api/entry     look up entries by ID (subject, date, category, link)
Neither needs ECL credentials - they only read the Qdrant index. Keeping the
index current is run_update.py's job, run on the host (where pwd.txt lives),
not through this API, so ECL credentials never reach this service.

Started as a port of RESTAPI_Qdrant_HTML.py; its login-gated /login, /search,
/recommend and /update endpoints and the HTML page that used them were dropped -
nothing used them once OpenWebUI went through /api/*.
"""

base_url = "https://dbweb1.fnal.gov:8443/ECL/sbnd/E"
import os
import re
from dotenv import load_dotenv
import logging
from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel, Field
from typing import List, Optional
from ElogSemanticSearch.QdrantSearch.QdrantSearch import SemanticSearch, Reranker, date_range

app = FastAPI(
    title="SBND Elog Semantic Search",
    description=(
        "Semantic search over the SBND ECL (electronic logbook). "
        "Use /api/search to find logbook entries relevant to a natural-language query, "
        "/api/similar to find entries related to one you already found, and "
        "/api/entry to look up entries by ID."
    ),
    version="1.0",
)

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def required_env(var_name: str) -> str:
    if var_name in os.environ:
        return os.environ[var_name]
    raise RuntimeError(f"Environment variable {var_name} must be set.")

# Two embedding models, loaded side by side for comparison. "mxbai" is the
# default (matches the old single-searcher behavior via MODEL_PATH /
# COLLECTION_NAME); "mpnet" is optional - it's only loaded if MPNET_MODEL_PATH
# is set, so the API still starts in single-model deployments.
searcher_configs = {
    "mxbai": {
        "model_path": required_env("MODEL_PATH"),
        "collection_name": required_env("COLLECTION_NAME"),
        "host": os.environ["QDRANT_HOST"],
        "port": 6333,
        # mxbai-embed-large-v1 is asymmetric: queries (not documents) need this
        # prefix for the model to score retrieval properly.
        "query_prefix": "Represent this sentence for searching relevant passages: ",
    },
}
if os.environ.get("MPNET_MODEL_PATH"):
    searcher_configs["mpnet"] = {
        "model_path": os.environ["MPNET_MODEL_PATH"],
        "collection_name": required_env("MPNET_COLLECTION_NAME"),
        "host": os.environ["QDRANT_HOST"],
        "port": 6333,
        "query_prefix": "",  # all-mpnet-base-v2 is symmetric, no prefix
    }

DEFAULT_MODEL = "mxbai"

# Cross-encoder reranker (bge-reranker-base): /api/search takes the top
# RERANK_TOP dense hits, re-sorts them with it and returns the best `limit`.
# Evaluated with DebugScripts/eval_retrieval.py (rerank-* methods). Unset
# RERANK_MODEL_PATH to switch it off; it needs the 'text' payload
# (run_update.py writes it, DebugScripts/backfill_text_payload.py for old points).
RERANK_MODEL_PATH = os.environ.get("RERANK_MODEL_PATH")
RERANK_TOP = int(os.environ.get("RERANK_TOP", "50"))

searchers = {}
reranker = None


def initialize_searchers():
    global reranker
    for key, config in searcher_configs.items():
        searchers[key] = SemanticSearch(
            model_path=config["model_path"],
            collection_name=config["collection_name"],
            qdrant_host=config["host"],
            qdrant_port=config["port"],
            query_prefix=config["query_prefix"],
        )
        logger.info(
            f"Searcher '{key}' initialized: model={config['model_path']}, collection={config['collection_name']}"
        )
    if RERANK_MODEL_PATH:
        reranker = Reranker(RERANK_MODEL_PATH)
    rerank_state = (f"reranking ON (top {RERANK_TOP}, {RERANK_MODEL_PATH})" if reranker
                    else "reranking OFF (RERANK_MODEL_PATH not set)")
    return f"Searchers initialized: {list(searchers)} (default: {DEFAULT_MODEL}); {rerank_state}"

print(initialize_searchers())


START_DATE_FIELD = Field(
    None,
    description=(
        "Only return entries created on or after this date. Accepts '2026', "
        "'2026-03', '2026-03-18' or a full ISO datetime. A partial date means "
        "the start of that year/month/day. Set this when the question names a "
        "time period, e.g. 'in 2026' -> start_date='2026', end_date='2026'."
    ),
)
END_DATE_FIELD = Field(
    None,
    description=(
        "Only return entries created on or before this date. Same formats as "
        "start_date; a partial date covers the whole period, so end_date='2026' "
        "includes all of 2026 and end_date='2026-03' includes all of March."
    ),
)


class ToolSearchRequest(BaseModel):
    query: str
    model: str = DEFAULT_MODEL  # "mxbai" or "mpnet" - which embedding model to search with
    limit: int = Field(10, ge=1, le=50)
    start_date: Optional[str] = START_DATE_FIELD
    end_date: Optional[str] = END_DATE_FIELD


SIMILAR_DATES_NOTE = (
    " Usually leave unset for /api/similar: the most useful similar entries are "
    "often from other years (the same procedure done every summer, the same fault "
    "recurring). Only set it when the user explicitly limits the period."
)


class ToolSimilarRequest(BaseModel):
    entry_id: int = Field(..., description="ID of the elog entry to find similar entries to, e.g. 39272.")
    model: str = DEFAULT_MODEL
    limit: int = Field(10, ge=1, le=50)
    start_date: Optional[str] = Field(None, description="Only return entries created on or after this "
                                      "date (same formats as /api/search)." + SIMILAR_DATES_NOTE)
    end_date: Optional[str] = Field(None, description="Only return entries created on or before this "
                                    "date (same formats as /api/search)." + SIMILAR_DATES_NOTE)


class ToolEntryRequest(BaseModel):
    entry_ids: List[int] = Field(..., description="Elog entry IDs to look up, e.g. [7298, 27447].",
                                 min_length=1, max_length=50)


# A query that is only an entry ID: "7298", "#7298", "entry 7298", "e=7298".
_BARE_ENTRY_ID = re.compile(r"^\s*(?:(?:elog\s+)?entry|elog|e\s*=)?\s*#?\s*(\d{1,9})\s*$", re.I)


def get_searcher(model: str) -> SemanticSearch:
    if model not in searchers:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model '{model}'. Available: {list(searchers)}",
        )
    return searchers[model]


def date_range_condition(start_date: Optional[str], end_date: Optional[str]):
    """
    Build the creation_date range condition (set_filter_conditions format) for
    the optional start_date/end_date, or None if neither is set.
    Raises HTTPException 400 on an unparseable or reversed range, so the calling
    LLM sees what was wrong and can retry.

    Filters reach the endpoints as flat keyword args (category, start_date,
    end_date) rather than set_filter_conditions' nested list-of-dicts, because
    LLM function-calling fills flat arguments far more reliably; this function
    does the translation.
    """
    try:
        bounds = date_range(start_date, end_date)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    if not bounds:
        return None
    return {"field": "creation_date", "type": "range", "value": bounds}


def entry_info(eid, payload):
    """One hit as the LLM sees it: ID, subject, category, date, link.

    The subject line is the part that matters. Everything here is already in
    Qdrant's payload from ingestion, so returning it costs no extra fetch,
    latency or credentials - but without the subject the calling LLM has no way
    to check *why* a hit matched, and it guesses instead (the "none of these
    mention Grafana v13.1" answer, where every ID was real and every claim
    about its content invented).

    Full entry body text is deliberately absent and can't be added cheaply: it
    is never stored in the payload, only used transiently to build the
    embedding at ingestion time, so returning it would mean a live
    authenticated re-fetch from the ECL per hit.
    """
    return {
        "entry_id": str(eid),
        "subject": payload.get("subject", ""),
        "category": payload.get("top_category", ""),
        "subcategory": payload.get("subcategory", ""),
        "date": payload.get("creation_date", ""),
        "url": f"{base_url}/show?e={eid}",
    }


def fetch_payloads(s: SemanticSearch, ids):
    """{str(id): payload} for the given entry IDs - only the fields responses use
    (the 'text' field is for the reranker and never goes back to the caller)."""
    points = s.client.retrieve(
        collection_name=s.collection_name,
        ids=[int(i) for i in ids],
        with_payload=["subject", "top_category", "subcategory", "creation_date"],
    )
    return {str(p.id): p.payload for p in points}


def lookup_entries(ids):
    """{str(id): payload} found in any loaded collection - the collections can be
    at different update points, so an entry may be in one but not yet the other."""
    found = {}
    for s in searchers.values():
        missing = [i for i in ids if str(i) not in found]
        if not missing:
            break
        found.update(fetch_payloads(s, missing))
    return found


@app.post(
    "/api/search",
    summary="Search the SBND logbook",
    description=(
        "Semantic search over SBND ECL logbook entries. Returns the most relevant "
        "entries with their subject, category, date, and a link to read the full entry. "
        "Write 'query' as a description of what happened, keeping the user's own "
        "descriptive words (e.g. 'timing cable swap for the summer and the revert "
        "after summer', not just 'timing cable 2026') - those words are often what "
        "connects related entries. If the question names a time period ('in 2026', "
        "'last March', 'since June 2025'), set start_date/end_date - the search "
        "itself does not know entry dates, so a date in the query text alone has "
        "little effect. The search only matches the words and meaning of each "
        "entry's own text, and the same kind of work is often written up with "
        "different words in different years (e.g. a 'timing cable swap' may be "
        "logged as 'swap from true to fake beam signals'). So if results look "
        "incomplete: retry using the technical terms that appear in the subjects of "
        "relevant hits you already found (hardware, signal and system names), and "
        "call /api/similar on the best hit, which finds related entries across "
        "years. Don't put entry IDs in 'query' - use /api/entry to look them up. "
        "The optional 'model' field selects the embedding model: 'mxbai' "
        "(default) or 'mpnet'."
    ),
)
def tool_search(
    request: ToolSearchRequest,
    # Per-request opt-out for testing/comparison ("X-Rerank: off"). Kept out of
    # the schema so the OpenWebUI LLM can't switch reranking off by accident.
    x_rerank: Optional[str] = Header(None, include_in_schema=False),
):
    s = get_searcher(request.model)

    # Semantic search can't match a bare number; answer it as a lookup instead.
    m = _BARE_ENTRY_ID.match(request.query)
    if m:
        eid = str(int(m[1]))
        payload = lookup_entries([eid]).get(eid)
        return {"model": request.model, "date_filter": None,
                "results": [entry_info(eid, payload)] if payload else [],
                "message": (f"The query is an entry ID, so this is that entry rather than a search. "
                            f"Use /api/entry to look up IDs and /api/similar to find related entries."
                            if payload else f"Entry {eid} is not in the search index.")}

    date_condition = date_range_condition(request.start_date, request.end_date)
    conditions = [date_condition] if date_condition else []
    date_filter = date_condition["value"] if date_condition else None

    if reranker is not None and (x_rerank or "").strip().lower() not in ("off", "0", "false", "no"):
        # Plain top `limit` after reranking - the adaptive score cutoff below is
        # tuned for cosine scores, not reranker scores (and was not evaluated here).
        candidates = s.search_candidates(request.query, conditions, RERANK_TOP)
        hits = reranker.rerank(request.query, candidates, request.limit)
        results = [entry_info(h.id, h.payload or {}) for h in hits]
        response = {"model": request.model, "reranked": True, "date_filter": date_filter,
                    "results": results}
        return response if results else {**response, "message": "No relevant entries found."}

    # score_threshold=0.0: rely on the adaptive threshold inside perform_search
    # rather than a fixed cutoff (the two models' score scales differ).
    hit_ids = s.perform_search(request.query, conditions, request.limit, 0.0)
    if not hit_ids:
        return {"model": request.model, "reranked": False, "date_filter": date_filter, "results": [],
                "message": "No relevant entries found."}

    payload_by_id = fetch_payloads(s, hit_ids)
    # retrieve() doesn't preserve order; keep ranking from search
    results = [entry_info(eid, payload_by_id.get(eid, {})) for eid in hit_ids]
    return {"model": request.model, "reranked": False, "date_filter": date_filter, "results": results}


@app.post(
    "/api/similar",
    summary="Find logbook entries similar to a given entry",
    description=(
        "Given the ID of an SBND ECL logbook entry (e.g. one returned by /api/search), "
        "return other entries most similar to it - typically follow-ups, reverts, "
        "repeats of the same problem, or related work on the same hardware. Use this "
        "after /api/search when the user asks for related entries, what happened "
        "next/before, or when a search found one relevant entry but likely missed "
        "others. It works from the entry's own content, so it often finds related "
        "entries that a re-worded search misses, including ones from other years. "
        "The given entry itself is not in the results. Leave start_date/end_date "
        "unset unless the user explicitly limits the period - filtering here is "
        "the usual way related entries from other years get lost."
    ),
)
def tool_similar(request: ToolSimilarRequest):
    s = get_searcher(request.model)
    source_id = str(request.entry_id)

    date_condition = date_range_condition(request.start_date, request.end_date)
    conditions = [date_condition] if date_condition else []
    date_filter = date_condition["value"] if date_condition else None

    # Check up front: recommend_results swallows Qdrant errors (e.g. unknown
    # ID) and returns [], which would look like "nothing similar".
    source_payload = fetch_payloads(s, [source_id]).get(source_id)
    if source_payload is None:
        raise HTTPException(
            status_code=404,
            detail=f"Entry {source_id} is not in the search index (it may be newer than the last update).",
        )

    # recommend_results lists the input entry first (for run_search.py's CLI);
    # drop it here. Its adaptive_threshold_max trims weak matches.
    hit_ids = [eid for eid in s.recommend_results([request.entry_id], [], conditions, request.limit, 0.0)
               if eid != source_id]
    response = {"model": request.model, "similar_to": entry_info(source_id, source_payload),
                "date_filter": date_filter}
    if not hit_ids:
        return {**response, "results": [], "message": "No similar entries found."}

    payload_by_id = fetch_payloads(s, hit_ids)
    results = [entry_info(eid, payload_by_id.get(eid, {})) for eid in hit_ids]
    return {**response, "results": results}


@app.post(
    "/api/entry",
    summary="Look up logbook entries by ID",
    description=(
        "Return the subject, category, date and link of SBND ECL logbook entries "
        "given their IDs (e.g. when the user mentions 'entry 7298' or an elog link "
        "with ?e=7298). Use this instead of putting an ID into /api/search. The "
        "full entry text is not available here - give the user the link to read it. "
        "To find entries related to one, use /api/similar."
    ),
)
def tool_entry(request: ToolEntryRequest):
    ids = [str(i) for i in dict.fromkeys(request.entry_ids)]  # dedupe, keep order
    found = lookup_entries(ids)
    return {
        "entries": [entry_info(eid, found[eid]) for eid in ids if eid in found],
        "not_found": [int(eid) for eid in ids if eid not in found],
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
