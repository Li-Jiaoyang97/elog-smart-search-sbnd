#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
verify_embeddings.py
--------------------
Sanity-check that each collection's stored vectors really came from its own
embedding model: re-fetch sample entries from ECL, re-embed with both models,
and compare cosine similarity against the stored vectors. The stored vector
should match its own model (~1.0) and be unrelated to the other model's
output (which can't even be compared - different dimensions).

Usage (from anywhere, inside the venv):
    python DebugScripts/verify_embeddings.py 2 801 11142 39276
"""

import os
import sys
import argparse

import numpy as np

ELOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, ELOG_DIR)
# run_updateSBND reads pwd.txt relative to the cwd at import time.
os.chdir(ELOG_DIR)

from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

from run_updateSBND import entry_to_point, USERNAME, PASSWORD
from DataFetch.ElogFetcherQdrantSBND import ElogFetcher
from DataFetch.text_to_vectorSBND import TextProcessor
from qdrant_client import QdrantClient

COLLECTIONS = {
    'Elog_collection_768_METADATA': os.path.join(ELOG_DIR, 'models', 'all-mpnet-base-v2'),
    'Elog_collection_1024_METADATA': os.path.join(ELOG_DIR, 'models', 'mxbai-embed-large-v1'),
}


def cosine(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('entry_ids', type=int, nargs='+')
    parser.add_argument('-u', '--username', default=USERNAME)
    parser.add_argument('-p', '--password', default=PASSWORD)
    args = parser.parse_args()

    client = QdrantClient(host=os.environ.get('QDRANT_HOST', 'localhost'), port=6333)
    fetcher = ElogFetcher()
    if not fetcher.create_authenticated_session(args.username, args.password):
        raise RuntimeError("ECL authentication failed.")

    processors = {name: TextProcessor(model_name=path) for name, path in COLLECTIONS.items()}

    all_ok = True
    for eid in args.entry_ids:
        entry = fetcher.get_entry(eid)
        if entry is None:
            print(f"ID {eid}: could not fetch from ECL - skipping")
            continue
        print(f"ID {eid}: '{entry.get('subject', '')[:60]}'")
        for name, processor in processors.items():
            stored = client.retrieve(collection_name=name, ids=[eid],
                                     with_vectors=True, with_payload=True)
            if not stored:
                print(f"  {name}: NOT PRESENT")
                all_ok = False
                continue
            expected = entry_to_point(entry, processor).vector
            sim = cosine(stored[0].vector, expected)
            ok = sim > 0.999
            all_ok &= ok
            print(f"  {name}: dim={len(stored[0].vector)}  "
                  f"cosine(stored, own-model re-embed) = {sim:.6f}  "
                  f"{'OK' if ok else 'MISMATCH (entry edited since crawl, or wrong model)'}")

    print("\nAll checks passed." if all_ok else "\nSome checks failed - see above.")


if __name__ == '__main__':
    main()
