#!/usr/bin/env python3
"""
Interactive semantic search CLI.
Uses SemanticSearch (semantic_search.py) + Qdrant running in Docker.

Future: drop the perform_search() logic into an OpenWebUI Pipeline by
copying it into the pipe() method of a Pipeline class.
"""

import argparse
import webbrowser
from QdrantSearch.QdrantSearchSBND import SemanticSearch  # assumes same directory

# ── Configuration ─────────────────────────────────────────────────────────────
MODEL_PATH      = "all-mpnet-base-v2"                    # must match the model used during ingestion
COLLECTION_NAME = "Elog_collection_768_METADATA"         # the Qdrant collection name
QDRANT_HOST     = "localhost"
QDRANT_PORT     = 6333

BASE_URL        = "https://dbweb1.fnal.gov:8443/ECL/sbnd/E"

# How many raw candidates to fetch before adaptive filtering trims them down.
SEARCH_LIMIT    = 10

# Minimum cosine similarity score (0.0 = no floor; adaptive logic does the rest).
SCORE_THRESHOLD = 0.0
DEFAULT_CONDITIONS = []
# ── Helpers ────────────────────────────────────────────────────────────────────

def print_results(hit_ids: list[str]) -> None:
    if not hit_ids:
        print("\n  No results found.\n")
        return
    print(f"\n  Found {len(hit_ids)} result(s):")
    for i, id_ in enumerate(hit_ids, 1):
        print(f"    {i}. ID {id_}  →  {BASE_URL}/show?e={id_}")
    print()

def pick_and_open(searcher: SemanticSearch, hit_ids: list[str]) -> None:
    """Let the user open individual results by number, one at a time."""
    if not hit_ids:
        return
    while True:
        raw = input("  Open entry # (or press Enter to skip): ").strip()
        if not raw:
            break
        try:
            index = int(raw) - 1
            if 0 <= index < len(hit_ids):
                url = searcher.construct_url_and_open([hit_ids[index]], BASE_URL)
                print(f"  Opening: {url}")
                webbrowser.open(url)
            else:
                print(f"  Please enter a number between 1 and {len(hit_ids)}.")
        except ValueError:
            print("  Please enter a valid number.")


def run_search(searcher: SemanticSearch, query: str) -> list[str]:
    print(f"\n  Searching for: \"{query}\"")
    return searcher.perform_search(
        search_query=query,
        conditions=DEFAULT_CONDITIONS,
        limit=SEARCH_LIMIT,
        score_threshold=SCORE_THRESHOLD,
    )


def run_recommend(searcher: SemanticSearch, pos_ids: list[str], neg_ids: list[str]) -> list[str]:
    print(f"\n  Recommending — positive IDs: {pos_ids}, negative IDs: {neg_ids}")
    return searcher.recommend_results(
        pos_id=pos_ids,
        neg_id=neg_ids,
        conditions=DEFAULT_CONDITIONS,
        limit=SEARCH_LIMIT,
        score_threshold=SCORE_THRESHOLD,
    )


def run_scroll(searcher: SemanticSearch) -> list[str]:
    print("\n  Browsing all entries (no query) …")
    return searcher.scroll_results(
        conditions=DEFAULT_CONDITIONS,
        limit=SEARCH_LIMIT,
    )


# ── Interactive loop ───────────────────────────────────────────────────────────

HELP_TEXT = """
Commands:
  <any text>          → semantic search
  !recommend <id>     → find more entries like that ID
  !browse             → scroll through entries (no query)
  !help               → show this message
  !quit               → exit
"""


def interactive_loop(searcher: SemanticSearch) -> None:
    print(HELP_TEXT)

    while True:
        try:
            raw = input("Search> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not raw:
            continue

        if raw.lower() in ("!quit", "!exit", "exit", "quit"):
            print("Bye!")
            break

        if raw.lower() == "!help":
            print(HELP_TEXT)
            continue

        if raw.lower() == "!browse":
            hit_ids = run_scroll(searcher)
            print_results(hit_ids)
            if hit_ids:
                pick_and_open(searcher,hit_ids)
            continue

        if raw.lower().startswith("!recommend"):
            parts = raw.split()
            if len(parts) < 2:
                print("  Usage: !recommend <id> [<id2> ...]  [neg:<neg_id>]\n")
                continue
            pos_ids = [p for p in parts[1:] if not p.startswith("neg:")]
            neg_ids = [p[4:] for p in parts[1:] if p.startswith("neg:")]
            hit_ids = run_recommend(searcher, pos_ids, neg_ids)
            print_results(hit_ids)
            if hit_ids:
                pick_and_open(searcher,hit_ids)
            continue

        # Default: semantic search
        hit_ids = run_search(searcher, raw)
        print_results(hit_ids)
        if hit_ids:
            pick_and_open(searcher,hit_ids)


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Interactive semantic search CLI")
    parser.add_argument("--model",      default=MODEL_PATH,      help="SentenceTransformer model name or path")
    parser.add_argument("--collection", default=COLLECTION_NAME, help="Qdrant collection name")
    parser.add_argument("--host",       default=QDRANT_HOST,     help="Qdrant host")
    parser.add_argument("--port",       default=QDRANT_PORT,     type=int, help="Qdrant port")
    args = parser.parse_args()

    print(f"\nConnecting to Qdrant at {args.host}:{args.port}, collection '{args.collection}' …")
    searcher = SemanticSearch(
        model_path=args.model,
        collection_name=args.collection,
        qdrant_host=args.host,
        qdrant_port=args.port,
    )
    print("Ready.\n")
    interactive_loop(searcher)


if __name__ == "__main__":
    main()