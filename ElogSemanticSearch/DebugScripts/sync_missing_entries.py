#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sync_missing_entries.py
-----------------------
Diff the entry IDs between two Qdrant collections and re-ingest the entries
missing from the target collection (fetched fresh from ECL, embedded with the
target collection's model).

Written to repair the Elog_collection_768_METADATA collection, which lost one
upsert batch (~500 points) during its backfill: the checkpoint counters said
stored=8386 but the collection held only 7886 points.

Usage (from the ElogSemanticSearch directory, inside the venv):
    python DebugScripts/sync_missing_entries.py              # report diff only
    python DebugScripts/sync_missing_entries.py --ingest     # diff + re-ingest

The default source/target/model match the mpnet-repair use case; override with
--source/--target/--model_path for other directions.
"""

import os
import sys
import argparse

ELOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, ELOG_DIR)
# run_updateSBND reads pwd.txt relative to the cwd at import time.
os.chdir(ELOG_DIR)

from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())  # load .env from cwd or parent dirs

# Reuse the exact ingest pipeline (skip filter + embedding/augmentation) so
# repaired points are byte-identical to what a normal run would have produced.
from run_updateSBND import entry_to_point, should_skip, USERNAME, PASSWORD
from DataFetch.ElogFetcherQdrantSBND import ElogFetcher
from DataFetch.text_to_vectorSBND import TextProcessor
from qdrant_client import QdrantClient


def scroll_all_ids(client, collection):
    ids = set()
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=collection,
            limit=1000,
            offset=offset,
            with_payload=False,
            with_vectors=False,
        )
        ids.update(int(p.id) for p in points)
        if offset is None:
            return ids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', default='Elog_collection_1024_METADATA')
    parser.add_argument('--target', default='Elog_collection_768_METADATA')
    parser.add_argument('--model_path',
                        default=os.path.join(ELOG_DIR, 'models', 'all-mpnet-base-v2'),
                        help='Embedding model for the TARGET collection.')
    parser.add_argument('--ingest', action='store_true',
                        help='Actually fetch and upsert the missing entries (default: report only).')
    parser.add_argument('-u', '--username', default=USERNAME)
    parser.add_argument('-p', '--password', default=PASSWORD)
    parser.add_argument('--batch_size', type=int, default=100)
    args = parser.parse_args()

    client = QdrantClient(host=os.environ.get('QDRANT_HOST', 'localhost'), port=6333)

    source_ids = scroll_all_ids(client, args.source)
    target_ids = scroll_all_ids(client, args.target)
    missing = sorted(source_ids - target_ids)
    extra = sorted(target_ids - source_ids)

    print(f"{args.source}: {len(source_ids)} points")
    print(f"{args.target}: {len(target_ids)} points")
    print(f"Missing from target: {len(missing)}")
    if missing:
        print(f"  ID range: {missing[0]}..{missing[-1]}")
        print(f"  IDs: {missing[:20]}{' ...' if len(missing) > 20 else ''}")
    print(f"In target but not source: {len(extra)}")
    if extra:
        print(f"  IDs: {extra[:20]}{' ...' if len(extra) > 20 else ''}")

    if not args.ingest:
        print("\nReport only - rerun with --ingest to repair the target collection.")
        return
    if not missing:
        print("\nNothing to ingest.")
        return

    vec_size = client.get_collection(args.target).config.params.vectors.size
    processor = TextProcessor(model_name=args.model_path)
    probe = processor.generate_embeddings("dimension probe")
    if len(probe) != vec_size:
        raise RuntimeError(
            f"Model produces {len(probe)}-dim vectors but '{args.target}' expects {vec_size}."
        )

    fetcher = ElogFetcher()
    if not fetcher.create_authenticated_session(args.username, args.password):
        raise RuntimeError("ECL authentication failed.")

    stored, skipped, failed = 0, 0, 0
    pending = []
    for i, eid in enumerate(missing, 1):
        try:
            entry = fetcher.get_entry(eid)
            if entry is None:
                failed += 1
                print(f"[{i}/{len(missing)}] ID {eid}: fetch returned nothing - left missing")
                continue
            if should_skip(entry):
                # The source collection stored it, but under current filter
                # rules (or current entry content) it would be skipped.
                skipped += 1
                print(f"[{i}/{len(missing)}] ID {eid}: now matches skip filter - not ingested")
                continue
            pending.append(entry_to_point(entry, processor))
        except Exception as e:
            failed += 1
            print(f"[{i}/{len(missing)}] ID {eid}: error - {e}")
            continue

        if len(pending) >= args.batch_size:
            client.upsert(collection_name=args.target, points=pending)
            stored += len(pending)
            print(f"[{i}/{len(missing)}] upserted batch of {len(pending)} (total stored: {stored})")
            pending = []

    if pending:
        client.upsert(collection_name=args.target, points=pending)
        stored += len(pending)

    final = client.get_collection(args.target).points_count
    print(f"\nDone. Stored: {stored} | Skipped: {skipped} | Failed: {failed}")
    print(f"'{args.target}' now has {final} points "
          f"('{args.source}' has {len(source_ids)}).")


if __name__ == '__main__':
    main()
