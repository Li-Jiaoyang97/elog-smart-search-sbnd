#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
backfill_text_payload.py
------------------------
Copy each entry's text from the local text cache (build_text_cache.py) into the
'text' payload field of the Qdrant collections, where the API's reranker reads
it. Points ingested by run_update.py from now on carry the field already;
this fills in everything ingested before. Vectors are not touched.

By default only points WITHOUT text are filled, so re-running is safe and never
replaces text written by ingestion (which can be newer than the cache).

Usage (from anywhere, inside the venv; refresh the cache first so the newest
entries are in it):
    python DebugScripts/build_text_cache.py
    python DebugScripts/backfill_text_payload.py --dry_run
    python DebugScripts/backfill_text_payload.py
    python DebugScripts/backfill_text_payload.py --remove     # undo: drop the field everywhere
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

from qdrant_client import QdrantClient, models

from build_text_cache import load_cache, CACHE_FILE

COLLECTIONS = ['Elog_collection_1024_METADATA', 'Elog_collection_768_METADATA']
BATCH = 256
NO_TEXT = models.Filter(must=[models.IsEmptyCondition(is_empty=models.PayloadField(key='text'))])


def scroll_ids(client, collection, scroll_filter=None):
    ids, offset = [], None
    while True:
        points, offset = client.scroll(collection, scroll_filter=scroll_filter, limit=5000,
                                       offset=offset, with_payload=False)
        ids += [p.id for p in points]
        if offset is None:
            return ids


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--collections', default=','.join(COLLECTIONS),
                        help='Comma-separated collections (default: %(default)s).')
    parser.add_argument('--overwrite', action='store_true',
                        help='Rewrite text on every point from the cache, not just points without it.')
    parser.add_argument('--dry_run', action='store_true', help='Only report what would be written.')
    parser.add_argument('--remove', action='store_true', help="Delete the 'text' field from all points.")
    args = parser.parse_args()

    client = QdrantClient(host=os.environ.get('QDRANT_HOST', 'localhost'), port=6333)
    cache = {} if args.remove else load_cache()
    if not args.remove and not cache:
        sys.exit(f"No cache at {CACHE_FILE} - run build_text_cache.py first.")

    for collection in args.collections.split(','):
        if args.remove:
            if not args.dry_run:
                client.delete_payload(collection, keys=['text'],
                                      points=models.FilterSelector(filter=models.Filter()), wait=True)
            print(f"{collection}: 'text' removed from all points" + (" (dry run)" if args.dry_run else ""))
            continue

        targets = scroll_ids(client, collection, None if args.overwrite else NO_TEXT)
        todo = [i for i in targets if i in cache]
        not_cached = [i for i in targets if i not in cache]
        print(f"{collection}: {len(targets)} points to fill, {len(todo)} found in the cache"
              + (f", {len(not_cached)} NOT in the cache (re-run build_text_cache.py): {not_cached[:10]}"
                 if not_cached else ""))
        if args.dry_run:
            continue

        for start in range(0, len(todo), BATCH):
            client.batch_update_points(collection, wait=True, update_operations=[
                models.SetPayloadOperation(set_payload=models.SetPayload(
                    payload={'text': cache[i]['text']}, points=[i]))
                for i in todo[start:start + BATCH]
            ])
        left = client.count(collection, count_filter=NO_TEXT, exact=True).count
        total = client.count(collection, exact=True).count
        print(f"  done: {total - left} / {total} points have text" + (f", {left} still without" if left else ""))


if __name__ == '__main__':
    main()
