#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_v2_collections.py
-----------------------
Embed the "text cleanup v2" strings (DebugScripts/text_v2.py) into parallel
Qdrant collections, so v1 and v2 can be scored against each other by
eval_retrieval.py without touching the live collections.

    Elog_collection_1024_METADATA  ->  Elog_collection_1024_V2   (mxbai)
    Elog_collection_768_METADATA   ->  Elog_collection_768_V2    (mpnet)

Same dimensions, same cosine distance, same payload keys and the same point
ids (== ECL entry ids) as the live pair; only the embedded text differs. That
is the whole point: any score difference in the A/B is attributable to the
text change and nothing else.

Text comes from the local cache, not from ECL - see text_v2.py for why.

Usage (from anywhere, inside the venv):
    python DebugScripts/build_v2_collections.py                 # both models
    python DebugScripts/build_v2_collections.py --model mpnet   # the fast one
    python DebugScripts/build_v2_collections.py --limit 200     # smoke test
    python DebugScripts/build_v2_collections.py --drop          # delete and exit
"""

import os
import sys
import time
import argparse

ELOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, ELOG_DIR)

from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

from tqdm import tqdm
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, PointStruct, VectorParams

from build_text_cache import load_cache
from text_v2 import to_v2
from DataFetch.text_to_vector import TextProcessor

MODELS = {
    'mxbai': {
        'path': os.path.join(ELOG_DIR, 'models', 'mxbai-embed-large-v1'),
        'collection': 'Elog_collection_1024_V2',
        'dim': 1024,
    },
    'mpnet': {
        'path': os.path.join(ELOG_DIR, 'models', 'all-mpnet-base-v2'),
        'collection': 'Elog_collection_768_V2',
        'dim': 768,
    },
}


def build(client, name, cfg, records, batch_size):
    existing = {c.name for c in client.get_collections().collections}
    if cfg['collection'] in existing:
        client.delete_collection(cfg['collection'])
    client.create_collection(
        collection_name=cfg['collection'],
        vectors_config=VectorParams(size=cfg['dim'], distance=Distance.COSINE),
    )
    print(f"{name}: created '{cfg['collection']}' (dim={cfg['dim']}, cosine)")

    processor = TextProcessor(model_name=cfg['path'])
    start = time.perf_counter()
    for i in tqdm(range(0, len(records), batch_size), desc=f'{name} embed'):
        chunk = records[i:i + batch_size]
        texts = [to_v2(r) for r in chunk]
        # Batch encode - one model call per chunk instead of per entry.
        vectors = processor.model.encode(texts, convert_to_numpy=True,
                                         batch_size=batch_size,
                                         show_progress_bar=False)
        client.upsert(collection_name=cfg['collection'], points=[
            PointStruct(
                id=rec['id'],
                vector=vec.tolist(),
                payload={
                    'subject':       rec.get('subject', ''),
                    'top_category':  rec.get('top_category', ''),
                    'subcategory':   rec.get('subcategory', ''),
                    'creation_date': processor.date_to_rfc3339(rec.get('creation_date', '')),
                    'text':          text,
                },
            )
            for rec, vec, text in zip(chunk, vectors, texts)
        ])
    elapsed = time.perf_counter() - start
    count = client.count(cfg['collection'], exact=True).count
    print(f"{name}: {count} points in {elapsed / 60:.1f} min "
          f"({len(records) / elapsed:.1f} entries/s)")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--model', choices=sorted(MODELS), action='append',
                        help='Build only this model (repeatable). Default: both.')
    parser.add_argument('--limit', type=int, help='Only embed the first N entries (smoke test).')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--drop', action='store_true',
                        help='Delete the V2 collections and exit.')
    args = parser.parse_args()

    client = QdrantClient(host=os.environ.get('QDRANT_HOST', 'localhost'), port=6333)
    names = args.model or sorted(MODELS)

    if args.drop:
        existing = {c.name for c in client.get_collections().collections}
        for name in names:
            coll = MODELS[name]['collection']
            if coll in existing:
                client.delete_collection(coll)
                print(f"deleted '{coll}'")
            else:
                print(f"'{coll}' does not exist")
        return

    cache = load_cache()
    if not cache:
        sys.exit("No text cache - run DebugScripts/build_text_cache.py first.")
    records = [cache[i] for i in sorted(cache)]
    if args.limit:
        records = records[:args.limit]
    # Fail before loading any model if the cache and text_v2 disagree.
    for rec in records:
        to_v2(rec)
    print(f"{len(records)} entries to embed into {', '.join(names)}")

    for name in names:
        build(client, name, MODELS[name], records, args.batch_size)


if __name__ == '__main__':
    main()
