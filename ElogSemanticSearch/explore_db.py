#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
explore_db.py
-------------
Explore the SBND ECL Qdrant database interactively.

Modes (pick one):
  --query TEXT         Semantic search — embed TEXT and return top matches
  --similar ID         Find entries most similar to entry ID
  --list-categories    List all unique category values and their counts
  --id-range           Show the min/max indexed entry ID and the N most recent entries

Options:
  --category CAT       Filter results by top_category (combine with --query or --similar)
  --top N              Number of results to show (default: 10)

Examples:
  python explore_db.py --query "beam off target"
  python explore_db.py --query "CRT noise issue" --category CRT --top 5
  python explore_db.py --similar 8423
  python explore_db.py --list-categories
  python explore_db.py --id-range --top 10
"""

import os
import argparse
import logging
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

from DataFetch.text_to_vectorSBND import TextProcessor
from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, FieldCondition, MatchValue

logging.basicConfig(level=logging.WARNING)

model_path = os.environ.get("MODEL_PATH")
if not model_path:
    raise RuntimeError("Environment variable MODEL_PATH must be set.")

collection_name = os.environ.get("COLLECTION_NAME")
if not collection_name:
    raise RuntimeError("Environment variable COLLECTION_NAME must be set.")

ECL_BASE_URL = os.environ.get("ECL_URL", "https://dbweb1.fnal.gov:8443/ECL/sbnd/E/show?e=")


def _get_qdrant_client():
    return QdrantClient(host=os.environ.get("QDRANT_HOST"), port=6333)


def _entry_url(entry_id):
    return f"{ECL_BASE_URL}{entry_id}"


def _print_hit(rank, point, show_score=True):
    p = point.payload
    score = getattr(point, 'score', None)
    entry_id = point.id

    score_str = f"  score={score:.4f}" if (show_score and score is not None) else ""
    top_cat = p.get('top_category', '')
    subcat  = p.get('subcategory', '')
    category_str = f"{top_cat}/{subcat}" if subcat else top_cat

    print(f"  [{rank}] ID={entry_id}{score_str}")
    print(f"       Category : {category_str or '(none)'}")
    print(f"       Subject  : {p.get('subject', '') or '(no subject)'}")
    print(f"       Date     : {p.get('creation_date', '')}")
    print(f"       URL      : {_entry_url(entry_id)}")
    print()


def cmd_query(args, qdrant, processor):
    """Semantic search: embed the query text and find closest vectors."""
    query_clean  = processor.process_text(args.query)
    query_vector = processor.generate_embeddings(query_clean).tolist()

    query_filter = None
    if args.category:
        query_filter = Filter(must=[
            FieldCondition(key="top_category", match=MatchValue(value=args.category))
        ])

    results = qdrant.search(
        collection_name=collection_name,
        query_vector=query_vector,
        limit=args.top,
        query_filter=query_filter,
        with_payload=True,
    )

    cat_note = f" in category '{args.category}'" if args.category else ""
    print(f"\nTop {len(results)} results for: \"{args.query}\"{cat_note}\n")
    for i, hit in enumerate(results, 1):
        _print_hit(i, hit)


def cmd_similar(args, qdrant):
    """Find entries whose vectors are closest to the given entry ID."""
    records = qdrant.retrieve(
        collection_name=collection_name,
        ids=[args.similar],
        with_vectors=True,
        with_payload=True,
    )
    if not records:
        print(f"Entry ID {args.similar} not found in the collection.")
        return

    source = records[0]
    p = source.payload
    top_cat = p.get('top_category', '')
    subcat  = p.get('subcategory', '')
    print(f"\nEntries similar to ID={args.similar}")
    print(f"  Category : {top_cat}/{subcat}" if subcat else f"  Category : {top_cat}")
    print(f"  Subject  : {p.get('subject', '') or '(no subject)'}\n")

    query_filter = None
    if args.category:
        query_filter = Filter(must=[
            FieldCondition(key="top_category", match=MatchValue(value=args.category))
        ])

    hits = qdrant.search(
        collection_name=collection_name,
        query_vector=source.vector,
        limit=args.top + 1,  # fetch one extra to drop self
        query_filter=query_filter,
        with_payload=True,
    )
    hits = [h for h in hits if h.id != args.similar][: args.top]

    for i, hit in enumerate(hits, 1):
        _print_hit(i, hit)


def cmd_list_categories(qdrant):
    """Scroll through the collection and tally category counts."""
    print(f"\nScanning '{collection_name}' for categories...\n")

    top_counts  = {}
    full_counts = {}
    offset = None

    while True:
        batch, offset = qdrant.scroll(
            collection_name=collection_name,
            limit=1000,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        for point in batch:
            top = point.payload.get('top_category', '') or '(none)'
            sub = point.payload.get('subcategory', '') or ''
            top_counts[top] = top_counts.get(top, 0) + 1
            full = f"{top}/{sub}" if sub else top
            full_counts[full] = full_counts.get(full, 0) + 1
        if offset is None:
            break

    if not top_counts:
        print("No entries found.")
        return

    total = sum(top_counts.values())
    print(f"Total entries: {total}\n")

    # Top-level categories
    print(f"{'Top category':<35}  {'Count':>6}")
    print("-" * 45)
    for cat, count in sorted(top_counts.items(), key=lambda x: -x[1]):
        print(f"  {cat:<33}  {count:>6}")

    # Full category breakdown (only show if subcategories exist)
    if len(full_counts) > len(top_counts):
        print(f"\n{'Full category (Top/Sub)':<35}  {'Count':>6}")
        print("-" * 45)
        for cat, count in sorted(full_counts.items(), key=lambda x: -x[1]):
            if '/' in cat:
                print(f"  {cat:<33}  {count:>6}")
    print()


def cmd_id_range(qdrant, top_n=10):
    """Scroll the whole collection to find the ID range and the most recent entries (highest ID)."""
    print(f"\nScanning '{collection_name}' for ID range...\n")

    points = []
    offset = None
    while True:
        batch, offset = qdrant.scroll(
            collection_name=collection_name,
            limit=1000,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        points.extend(batch)
        if offset is None:
            break

    if not points:
        print("No entries found.")
        return

    ids = [p.id for p in points]
    print(f"Total entries indexed : {len(points)}")
    print(f"Min ID                : {min(ids)}")
    print(f"Max ID                : {max(ids)}")

    print(f"\nTop {top_n} most recent entries (highest ID):\n")
    newest = sorted(points, key=lambda p: p.id, reverse=True)[:top_n]
    for i, point in enumerate(newest, 1):
        _print_hit(i, point, show_score=False)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Explore the SBND ECL Qdrant database.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--query',   type=str, metavar='TEXT', help='Semantic search query.')
    mode.add_argument('--similar', type=int, metavar='ID',   help='Find entries similar to this entry ID.')
    mode.add_argument('--list-categories', action='store_true', help='List all categories and counts.')
    mode.add_argument('--id-range', action='store_true', help='Show the min/max indexed entry ID and the N most recent entries.')

    parser.add_argument('--category', type=str, default=None,
                        help='Filter results by top_category (use with --query or --similar).')
    parser.add_argument('--top', type=int, default=10,
                        help='Number of results to show (default: 10).')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    qdrant = _get_qdrant_client()

    if args.list_categories:
        cmd_list_categories(qdrant)
    elif args.id_range:
        cmd_id_range(qdrant, top_n=args.top)
    elif args.query:
        processor = TextProcessor(model_name=model_path)
        cmd_query(args, qdrant, processor)
    elif args.similar is not None:
        cmd_similar(args, qdrant)
