#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
check_entry_in_db.py
---------------------
Check whether one or more ECL entry IDs have been upserted into the Qdrant
collection. Entries are stored with point id == entry id, so this is a
direct retrieve (no scan needed).

Usage:
    python check_entry_in_db.py 37875
    python check_entry_in_db.py 37875 40011 34648
"""

import os
import sys
import argparse
from dotenv import load_dotenv, find_dotenv
from qdrant_client import QdrantClient

#load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))
load_dotenv(find_dotenv())
collection_name = os.environ.get("COLLECTION_NAME")
if not collection_name:
    raise RuntimeError("Environment variable COLLECTION_NAME must be set.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('entry_ids', type=int, nargs='+', help='ECL entry ID(s) to check')
    args = parser.parse_args()

    client = QdrantClient(host=os.environ.get("QDRANT_HOST"), port=6333)

    found = client.retrieve(
        collection_name=collection_name,
        ids=args.entry_ids,
        with_payload=True,
    )
    found_by_id = {p.id: p for p in found}

    for entry_id in args.entry_ids:
        point = found_by_id.get(entry_id)
        if point is None:
            print(f"[{entry_id}] NOT found in collection '{collection_name}'.")
        else:
            payload = point.payload or {}
            print(f"[{entry_id}] FOUND — "
                  f"subject={payload.get('subject', '')!r}, "
                  f"category={payload.get('top_category', '')}/{payload.get('subcategory', '')}, "
                  f"creation_date={payload.get('creation_date', '')}")


if __name__ == "__main__":
    main()
