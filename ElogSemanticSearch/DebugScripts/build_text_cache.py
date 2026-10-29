#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_text_cache.py
-------------------
Fetch from ECL the text of every entry stored in Qdrant and save it locally
(one JSON object per line) so retrieval experiments - token_stats.py,
eval_retrieval.py - can run offline without re-crawling ECL each time.

The cached text is exactly what run_update.py embeds (entry_to_text), so
token counts and BM25 see the same input as the dense models.

Resumable: IDs already in the cache are skipped unless --refresh is given.
The cache holds logbook content - it is gitignored and dockerignored.

Usage (from anywhere, inside the venv):
    python DebugScripts/build_text_cache.py
    python DebugScripts/build_text_cache.py --refresh
"""

import os
import sys
import json
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

ELOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, ELOG_DIR)

from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

from tqdm import tqdm
from qdrant_client import QdrantClient

from run_update import entry_to_text, USERNAME, PASSWORD
from DataFetch.ElogFetcherQdrant import ElogFetcher
from DataFetch.text_to_vector import TextProcessor

CACHE_FILE = os.path.join(ELOG_DIR, 'entry_text_cache.jsonl')
COLLECTIONS = ['Elog_collection_1024_METADATA', 'Elog_collection_768_METADATA']
# entry_to_text only uses the processor's text-cleaning methods, not its
# model, but TextProcessor always loads one - use the smaller model.
PROCESSOR_MODEL = os.path.join(ELOG_DIR, 'models', 'all-mpnet-base-v2')


def load_cache(path=CACHE_FILE):
    """Return {entry_id: record} from the cache file ({} if it doesn't exist)."""
    cache = {}
    if not os.path.exists(path):
        return cache
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rec = json.loads(line)
                cache[rec['id']] = rec  # later lines win (re-fetched entries)
    return cache


def stored_ids(client):
    """Union of point IDs across both collections."""
    ids = set()
    for name in COLLECTIONS:
        offset = None
        while True:
            points, offset = client.scroll(
                collection_name=name, limit=1000, offset=offset,
                with_payload=False, with_vectors=False,
            )
            ids.update(p.id for p in points)
            if offset is None:
                break
    return ids


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--refresh', action='store_true',
                        help='Re-fetch every entry, even ones already cached.')
    parser.add_argument('--max_workers', type=int, default=8)
    parser.add_argument('-u', '--username', default=USERNAME)
    parser.add_argument('-p', '--password', default=PASSWORD)
    args = parser.parse_args()

    client = QdrantClient(host=os.environ.get('QDRANT_HOST', 'localhost'), port=6333)
    ids = stored_ids(client)
    cache = {} if args.refresh else load_cache()
    todo = sorted(ids - set(cache))
    print(f"{len(ids)} IDs in Qdrant, {len(cache)} already cached, {len(todo)} to fetch.")
    if not todo:
        return

    fetcher = ElogFetcher()
    if not fetcher.create_authenticated_session(args.username, args.password):
        raise RuntimeError("ECL authentication failed.")
    processor = TextProcessor(model_name=PROCESSOR_MODEL)

    failed, absent = {}, []
    mode = 'w' if args.refresh else 'a'
    with open(CACHE_FILE, mode) as out, \
            ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {executor.submit(fetcher.get_entry, eid): eid for eid in todo}
        for future in tqdm(as_completed(futures), total=len(futures), desc='Fetching'):
            eid = futures[future]
            try:
                entry = future.result()
            except Exception as e:
                failed[eid] = f"{type(e).__name__}: {e}"
                continue
            if entry is None:
                absent.append(eid)
                continue
            out.write(json.dumps({
                'id':            eid,
                'subject':       entry.get('subject', ''),
                'top_category':  entry.get('top_category', ''),
                'subcategory':   entry.get('subcategory', ''),
                'creation_date': entry.get('creation_date', ''),
                'text':          entry_to_text(entry, processor),
            }) + '\n')
            out.flush()

    print(f"\nCached {len(todo) - len(failed) - len(absent)} entries -> {CACHE_FILE}")
    if absent:
        print(f"{len(absent)} IDs are in Qdrant but no longer exist in ECL: {absent[:20]}")
    if failed:
        print(f"{len(failed)} fetches failed (re-run to retry):")
        for eid, reason in list(failed.items())[:20]:
            print(f"  {eid}: {reason}")


if __name__ == '__main__':
    main()
