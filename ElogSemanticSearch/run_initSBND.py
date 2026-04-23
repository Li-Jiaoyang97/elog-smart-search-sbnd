#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_initSBND.py
---------------
One-time historical load of all SBND ECL entries into Qdrant.

Strategy:
  - Auto-detect the current max entry ID across all categories
  - Walk every ID from 1 to max_id in parallel batches
  - Skip entries whose subject contains "checklist" (case-insensitive)
  - Embed and upsert everything else to Qdrant in batches
  - Checkpoint progress after every chunk so the script can be safely
    resumed after an interruption (just re-run the same command)

Usage:
  python run_initSBND.py                    # auto-detect max ID
  python run_initSBND.py --max_id 12000     # override max ID
  python run_initSBND.py --dry_run          # fetch + filter but skip Qdrant writes
  python run_initSBND.py --reset            # delete checkpoint and start over

@author: jiaoyang
"""

import os
import json
import logging
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

with open('pwd.txt', 'r') as f:
    ecl_pwd = f.read().strip()

USERNAME = os.environ.get('ECL_USERNAME', 'sbndprm')
PASSWORD = os.environ.get('ECL_PASSWORD', ecl_pwd)

from DataFetch.ElogFetcherQdrantSBND import ElogFetcher
from DataFetch.text_to_vectorSBND import TextProcessor
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, PointStruct, VectorParams

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

model_path = os.environ.get("MODEL_PATH")
if not model_path:
    raise RuntimeError("Environment variable MODEL_PATH must be set.")

collection_name = os.environ.get("COLLECTION_NAME")
if not collection_name:
    raise RuntimeError("Environment variable COLLECTION_NAME must be set.")

CHECKPOINT_FILE = os.path.join(os.path.dirname(__file__), "init_checkpoint.json")

# Entries whose subject contains any of these words (case-insensitive) are skipped.
SKIP_SUBJECT_KEYWORDS = ["checklist"]

# Entries whose full category (top_category/subcategory) matches are skipped.
# Use the same format as ECL: "Top" or "Top/Sub".
SKIP_CATEGORIES = ["DAQ/Automation"]

# -- Clear collection helper (for --clear flag) ----------------------------------
def clear_collection():
    """Delete all points from the configured Qdrant collection."""
    qdrant_client = _get_qdrant_client()
    qdrant_client.delete_collection(collection_name=collection_name)
    logger.info(f"Deleted collection '{collection_name}'.")
    return True


# ── Filtering ─────────────────────────────────────────────────────────────────

def should_skip(entry):
    subject = entry.get('subject', '').lower()
    text    = entry.get('text', '').lower()
    if any(kw in subject or kw in text for kw in SKIP_SUBJECT_KEYWORDS):
        return True

    top = entry.get('top_category', '')
    sub = entry.get('subcategory', '')
    full_category = f"{top}/{sub}" if sub else top
    if full_category in SKIP_CATEGORIES:
        return True

    return False


# ── Checkpoint helpers ────────────────────────────────────────────────────────

def load_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE) as f:
            data = json.load(f)
        logger.info(
            f"Resuming from checkpoint: last processed ID={data['last_id']}, "
            f"stored={data['stored']}, skipped={data['skipped']}"
        )
        return data
    return {"last_id": 0, "stored": 0, "skipped": 0}


def save_checkpoint(last_id, stored, skipped):
    with open(CHECKPOINT_FILE, 'w') as f:
        json.dump({"last_id": last_id, "stored": stored, "skipped": skipped}, f)


# ── Max ID detection ──────────────────────────────────────────────────────────

def detect_max_id(fetcher):
    """Find the current highest entry ID by checking the most recent entry across all categories."""
    logger.info("Auto-detecting max entry ID across all categories...")
    results = fetcher.check_most_recent_entries_allCategories(after='30days', limit=5)
    ids = [v for v in results.values() if v is not None]
    if not ids:
        raise RuntimeError("Could not detect any entry IDs. Check ECL authentication.")
    max_id = max(ids)
    logger.info(f"Detected max entry ID: {max_id}")
    return max_id


# ── Entry processing ──────────────────────────────────────────────────────────

def entry_to_point(entry, processor):
    """Embed a parsed entry dict and return a Qdrant PointStruct."""
    main_text = entry.get('text', '')
    comments_text = " ".join(c.get('text', '') for c in entry.get('comments', []))
    combined = f"{main_text} {comments_text}".strip()

    clean_text = processor.process_text(combined)
    augmented_text = clean_text

    logs = entry.get('logs', [])
    subcategory = entry.get('subcategory', '')
    cats = [subcategory] if subcategory else []

    if cats:
        augmented_text = processor.augment_with_cats(augmented_text, cats)
    augmented_text = processor.augment_with_logs(augmented_text, logs)

    vector        = processor.generate_embeddings(augmented_text).tolist()
    creation_date = processor.date_to_rfc3339(entry.get('creation_date', ''))

    return PointStruct(
        id=entry['id'],
        vector=vector,
        payload={
            'subject':       entry.get('subject', ''),
            'top_category':  entry.get('top_category', ''),
            'subcategory':   entry.get('subcategory', ''),
            'creation_date': creation_date,
        }
    )


# ── Qdrant helpers ────────────────────────────────────────────────────────────

def _get_qdrant_client():
    return QdrantClient(host=os.environ.get("QDRANT_HOST"), port=6333)


def ensure_collection(qdrant_client, vector_size=768):
    existing = {c.name for c in qdrant_client.get_collections().collections}
    if collection_name not in existing:
        qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )
        logger.info(f"Created collection '{collection_name}' (dim={vector_size}, cosine).")
    else:
        logger.info(f"Collection '{collection_name}' already exists — will upsert into it.")


# ── Main ──────────────────────────────────────────────────────────────────────

def run_init(username, password, max_id=None, max_workers=8, chunk_size=100,
             batch_size=500, dry_run=False):
    logger.info("Initializing text processor and ECL fetcher...")
    processor = TextProcessor(model_name=model_path)
    fetcher   = ElogFetcher()
    if not fetcher.create_authenticated_session(username, password):
        raise RuntimeError("ECL authentication failed.")

    if max_id is None:
        max_id = detect_max_id(fetcher)

    checkpoint = load_checkpoint()
    start_id   = checkpoint["last_id"] + 1
    stored     = checkpoint["stored"]
    skipped    = checkpoint["skipped"]

    if start_id > max_id:
        logger.info("Checkpoint already at max ID — nothing to do.")
        return

    total = max_id - start_id + 1
    logger.info(f"Processing IDs {start_id} → {max_id}  ({total} IDs to attempt).")

    if not dry_run:
        qdrant_client = _get_qdrant_client()
        ensure_collection(qdrant_client)

    pending_points = []  # accumulate PointStructs until batch_size reached

    for chunk_start in range(start_id, max_id + 1, chunk_size):
        chunk_ids = list(range(chunk_start, min(chunk_start + chunk_size, max_id + 1)))

        # ── Parallel fetch ────────────────────────────────────────────────────
        fetch_results = {}
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(fetcher.get_entry, eid): eid for eid in chunk_ids}
            for future in as_completed(futures):
                eid = futures[future]
                try:
                    fetch_results[eid] = future.result()
                except Exception as e:
                    logger.warning(f"ID {eid}: fetch error — {e}")
                    fetch_results[eid] = None

        # ── Process in ID order ───────────────────────────────────────────────
        for eid in chunk_ids:
            entry = fetch_results.get(eid)

            if entry is None:
                continue  # ID doesn't exist in ECL

            if should_skip(entry):
                skipped += 1
                logger.debug(f"ID {eid}: skipped  subject='{entry.get('subject', '')}'")
                continue

            if dry_run:
                logger.info(
                    f"[DRY RUN] ID {eid}: would store — "
                    f"category='{entry.get('top_category', '')}' "
                    f"subject='{entry.get('subject', '')}'"
                )
                stored += 1
                continue

            try:
                point = entry_to_point(entry, processor)
                pending_points.append(point)
            except Exception as e:
                logger.warning(f"ID {eid}: processing error — {e}")
                continue

            # Upsert when batch is full
            if len(pending_points) >= batch_size:
                qdrant_client.upsert(collection_name=collection_name, points=pending_points)
                stored += len(pending_points)
                logger.info(
                    f"Upserted {len(pending_points)} points  |  "
                    f"total stored={stored}  skipped={skipped}  last_id={eid}"
                )
                pending_points = []

        # Checkpoint at chunk boundary
        save_checkpoint(chunk_ids[-1], stored, skipped)
        logger.info(
            f"Chunk done: IDs {chunk_ids[0]}–{chunk_ids[-1]}  |  "
            f"stored={stored}  skipped={skipped}  pending={len(pending_points)}"
        )

    # Final partial batch
    if pending_points and not dry_run:
        qdrant_client.upsert(collection_name=collection_name, points=pending_points)
        stored += len(pending_points)
        save_checkpoint(max_id, stored, skipped)

    logger.info(f"Done.  Stored: {stored}  |  Skipped: {skipped}  |  Max ID: {max_id}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_arguments():
    parser = argparse.ArgumentParser(
        description='One-time historical SBND ECL → Qdrant initial load.'
    )
    parser.add_argument('-u', '--username',    type=str, default=USERNAME)
    parser.add_argument('-p', '--password',    type=str, default=PASSWORD)
    parser.add_argument('--max_id',     type=int, default=None,
                        help='Override auto-detected max entry ID.')
    parser.add_argument('--max_workers', type=int, default=8,
                        help='Parallel fetch threads (reduce if server resets connections).')
    parser.add_argument('--chunk_size',  type=int, default=100,
                        help='IDs fetched per parallel batch (also checkpoint frequency).')
    parser.add_argument('--batch_size',  type=int, default=500,
                        help='Qdrant upsert batch size.')
    parser.add_argument('--dry_run', action='store_true',
                        help='Fetch and filter but do not write to Qdrant.')
    parser.add_argument('--reset', action='store_true',
                        help='Delete the checkpoint file and start from ID 1.')
    parser.add_argument('--clear', action='store_true',
                        help='Delete all points from the Qdrant collection before updating.')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    if args.reset and os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)
        logger.info("Checkpoint deleted — starting from ID 1.")

    if args.clear:
        logger.warning(f"--clear flag set: deleting all points from collection '{collection_name}'.")
        clear_collection()

    run_init(
        username=args.username,
        password=args.password,
        max_id=args.max_id,
        max_workers=args.max_workers,
        chunk_size=args.chunk_size,
        batch_size=args.batch_size,
        dry_run=args.dry_run,
    )
