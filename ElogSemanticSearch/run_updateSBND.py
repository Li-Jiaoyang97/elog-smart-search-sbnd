
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 22 19:22:57 2024

@author: haydenh
@modified by jiaoyang for SBND ECL
"""

#!/usr/bin/env python

import os
import logging
import argparse
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))


with open('pwd.txt', 'r') as f:
    ecl_pwd = f.read().strip()  

# ── Credentials (set env vars or hardcode for local testing) ──────────────────
USERNAME = os.environ.get('ECL_USERNAME', 'sbndprm')
PASSWORD = os.environ.get('ECL_PASSWORD', ecl_pwd)

from DataFetch.ElogFetcherQdrantSBND import ElogFetcher
from DataFetch.text_to_vectorSBND import TextProcessor
from concurrent.futures import ThreadPoolExecutor, as_completed
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, PointStruct, VectorParams

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

model_path = os.environ.get("MODEL_PATH")
if not model_path:
    raise RuntimeError("Environment variable MODEL_PATH must be set.")

collection_name = os.environ.get("COLLECTION_NAME")
if not collection_name:
    raise RuntimeError("Environment variable COLLECTION_NAME must be set.")

SKIP_SUBJECT_KEYWORDS = ["checklist"]


def _print_entry(entry):
    print(f"  ID:            {entry['id']}")
    print(f"  Log:           {entry['logs']}")
    print(f"  Date:          {entry['creation_date']}")
    print(f"  Categories:    {entry['categories']}")
    print(f"  Subject:       {entry.get('subject', '')}")
    print(f"  Text preview:  {entry['text'][:120]}...")
    print(f"  Comments:      {len(entry['comments'])} comment(s)")
    for i, c in enumerate(entry['comments'], 1):
        print(f"    [{i}] {c['modified_date']} — {c['text'][:60]}...")


def should_skip(entry_data):
    subject = entry_data.get('subject', '').lower()
    text    = entry_data.get('text', '').lower()
    return any(kw in subject or kw in text for kw in SKIP_SUBJECT_KEYWORDS)


def entry_to_point(entry, processor):
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

    vector = processor.generate_embeddings(augmented_text).tolist()
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


def fetch_and_process_entries(client, processor, entry_ids, max_workers=10, verbose=False):
    """Fetch and process entry IDs in parallel, returning a list of PointStructs."""
    logger.info(f"Fetching {len(entry_ids)} entries with {max_workers} workers.")

    def fetch_one(entry_id):
        entry = client.get_entry(str(entry_id))
        if entry is None:
            logger.warning(f"No data found for entry ID {entry_id}")
            return None
        if should_skip(entry):
            if verbose:
                logger.info(f"Skipped entry ID {entry_id}: subject='{entry.get('subject', '')}'")
            return None
        if verbose:
            _print_entry(entry)
        try:
            return entry_to_point(entry, processor)
        except Exception as e:
            logger.warning(f"ID {entry_id}: processing error — {e}")
            return None

    points = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(fetch_one, eid): eid for eid in entry_ids}
        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                points.append(result)

    return points


def parse_arguments():
    parser = argparse.ArgumentParser(description='Run the SBND ECL update script.')
    parser.add_argument('--after', type=str, default='7days',
                        help='Time window for fetching recent entries (e.g. 7days, 30days).')
    parser.add_argument('--limit', type=int, default=50,
                        help='Max entries to fetch per category.')
    parser.add_argument('-u', '--username', type=str, default=USERNAME, help='Username for authentication.')
    parser.add_argument('-p', '--password', type=str, default=PASSWORD, help='Password for authentication.')
    parser.add_argument('--verbose', action='store_true',
                        help='Print entry details as they are processed.')
    return parser.parse_args()


def _get_qdrant_client():
    return QdrantClient(host=os.environ.get("QDRANT_HOST"), port=6333)


def ensure_collection_exists(qdrant_client, vector_size=768):
    """Create the Qdrant collection if it does not already exist."""
    existing = {c.name for c in qdrant_client.get_collections().collections}
    if collection_name not in existing:
        qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )
        logger.info(f"Created collection '{collection_name}' (dim={vector_size}, cosine).")
    else:
        logger.info(f"Collection '{collection_name}' already exists.")



def run_update(after='7days', limit=50, username="", password="", verbose=False):

    try:
        logger.info("Initializing text processor and ElogFetcher client.")
        processor = TextProcessor(model_name=model_path)
        client = ElogFetcher()
        client.create_authenticated_session(username, password)

        logger.info(f"Collecting entry IDs across all categories (after={after}, limit={limit} per category).")
        entry_ids = client.get_entry_ids_allCategories(after=after, limit=limit)

        all_entry_ids = []
        for cat, ids in entry_ids.items():
            all_entry_ids.extend(ids)
        all_entry_ids = sorted(set(all_entry_ids), reverse=True)

        if not all_entry_ids:
            logger.warning("No entry IDs found. Nothing to process.")
            return False

        logger.info(f"Found {len(all_entry_ids)} unique entries to process.")

        points = fetch_and_process_entries(client, processor, all_entry_ids, verbose=verbose)

        if not points:
            logger.warning("No points to upsert.")
            return False

        logger.info("Initializing Qdrant client.")
        qdrant_client = _get_qdrant_client()
        ensure_collection_exists(qdrant_client)

        batch_size = 500
        for i in range(0, len(points), batch_size):
            batch = points[i:i + batch_size]
            qdrant_client.upsert(collection_name=collection_name, points=batch)
            logger.info(f"Upserted batch of {len(batch)} entries.")

        logger.info(f"Data upsert complete. Total: {len(points)} entries.")
        return True

    except Exception as e:
        logger.error(f"Update failed: {e}", exc_info=True)
        return False


if __name__ == "__main__":
    args = parse_arguments()
    run_update(args.after, args.limit, args.username, args.password, args.verbose)
