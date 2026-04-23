
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
import pandas as pd
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

def should_skip(entry_data):
    subject = entry_data.get('subject', '').lower()
    return any(kw in subject for kw in SKIP_SUBJECT_KEYWORDS)


def fetch_and_process_entries(client, processor, entry_ids, max_workers=10):
    """Fetch and process a list of entry IDs in parallel, returning a list of DataFrames."""
    logger.info(f"Fetching {len(entry_ids)} entries with {max_workers} workers.")

    def fetch_one(entry_id):
        entry_data = client.get_entry(str(entry_id))
        if entry_data is None:
            logger.warning(f"No data found for entry ID {entry_id}")
            return None
        if should_skip(entry_data):
            logger.debug(f"Skipped entry ID {entry_id}: subject='{entry_data.get('subject', '')}'")
            return None
        return process_data_to_dataframe(entry_data, processor)

    dataframes = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(fetch_one, eid): eid for eid in entry_ids}
        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                dataframes.append(result)

    return dataframes


def parse_arguments():
    parser = argparse.ArgumentParser(description='Run the SBND ECL update script.')
    parser.add_argument('--after', type=str, default='7days',
                        help='Time window for fetching recent entries (e.g. 7days, 30days).')
    parser.add_argument('--limit', type=int, default=50,
                        help='Max entries to fetch per category.')
    parser.add_argument('-u', '--username', type=str, default=USERNAME, help='Username for authentication.')
    parser.add_argument('-p', '--password', type=str, default=PASSWORD, help='Password for authentication.')
    parser.add_argument('--clear', action='store_true',
                        help='Delete all points from the Qdrant collection before updating.')
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


def clear_collection():
    """Delete all points from the configured Qdrant collection."""
    qdrant_client = _get_qdrant_client()
    qdrant_client.delete_collection(collection_name=collection_name)
    logger.info(f"Deleted collection '{collection_name}'.")
    return True


def process_data_to_dataframe(entry_data, processor):
    rows = []
    main_text = entry_data.get('text', '')

    comments = entry_data.get('comments', [])
    comments_text = " ".join(comment.get('text', '') for comment in comments)

    combined_text = f"{main_text} {comments_text}".strip()

    logger.debug(f"Original Text for ID {entry_data.get('id')}: {combined_text[:200]}...")

    clean_text = processor.process_text(combined_text)
    augmented_text = clean_text

    logs = entry_data.get('logs', [])
    subcategory = entry_data.get('subcategory', '')
    cats = [subcategory] if subcategory else []

    if cats:
        augmented_text = processor.augment_with_cats(augmented_text, cats)

    augmented_text = processor.augment_with_logs(augmented_text, logs)

    # Print 2: After Augmentation**
    print(f"[DEBUG] Augmented Text: {augmented_text[:300]}...\n")

    vector = processor.generate_embeddings(augmented_text)

    creation_date = processor.date_to_rfc3339(entry_data.get('creation_date', ''))

    rows.append({
        'id':           entry_data.get('id'),
        'vector':       vector,
        'subject':      entry_data.get('subject', ''),
        'top_category': entry_data.get('top_category', ''),
        'subcategory':  entry_data.get('subcategory', ''),
        'creation_date': creation_date,
    })

    logger.info(f"Processed entry for ID {entry_data.get('id')}.")
    return pd.DataFrame(rows)


def run_update(after='7days', limit=50, username="", password=""):

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

        # grab the e-log with the most recent entry ID to get the category names and log names for augmentation
        for cat, ids in entry_ids.items():
            if ids:
                logger.info(f"Category '{cat}' has {len(ids)} entries. Most recent ID: {max(ids)}")
                most_recent_id = max(ids)
                most_recent_entry = client.get_entry(most_recent_id)
                client._print_entry(most_recent_entry)
            else:
                logger.info(f"Category '{cat}' has no recent entries.")
        
        dataframes = fetch_and_process_entries(client, processor, all_entry_ids)

        '''
        logger.info("Initializing Qdrant client.")
        qdrant_client = _get_qdrant_client()
        ensure_collection_exists(qdrant_client)

        batch_size = 500
        batch = []

        logger.info("Upserting data into Qdrant.")
        for df in dataframes:
            for _, row in df.iterrows():
                elog_id = row['id']
                vector = row['vector']
                creation_date = row['creation_date']

                payload = {
                    "subject":       row.get('subject', ''),
                    "top_category":  row.get('top_category', ''),
                    "subcategory":   row.get('subcategory', ''),
                    "creation_date": creation_date,
                }
                batch.append(PointStruct(id=elog_id, vector=vector, payload=payload))

                if len(batch) >= batch_size:
                    qdrant_client.upsert(collection_name=collection_name, points=batch)
                    logger.info(f"Upserted a batch of {len(batch)} entries.")
                    batch = []

        if batch:
            qdrant_client.upsert(collection_name=collection_name, points=batch)
            logger.info(f"Upserted the final batch of {len(batch)} entries.")
        '''
        logger.info("Data upsert complete.")
        return True

    except Exception as e:
        logger.error(f"Update failed: {e}", exc_info=True)
        return False


if __name__ == "__main__":
    args = parse_arguments()
    if args.clear:
        logger.warning(f"--clear flag set: deleting all points from collection '{collection_name}'.")
        clear_collection()
    run_update(args.after, args.limit, args.username, args.password)
