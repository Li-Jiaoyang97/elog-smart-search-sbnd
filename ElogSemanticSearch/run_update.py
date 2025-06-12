
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 22 19:22:57 2024

@author: haydenh
"""

base_url = "https://www-bd.fnal.gov/Elog/"


#!/usr/bin/env python

import os
import logging
import argparse  
from DataFetch.ElogFetcherQdrant import ElogFetcher
from DataFetch.text_to_vector import TextProcessor
import pandas as pd
import json
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct
import traceback

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def fetch_and_process_entries(client, processor, start_id, end_id):
    dataframes = []

    logger.info(f"Fetching entries from {start_id} to {end_id}.")

    for entry_id in range(start_id, end_id - 1, -1):  #Count from high to low
        logger.debug(f"Fetching entry ID: {entry_id}")

        entry_data = client.get_entry(str(entry_id))

        if entry_data:
            df = process_data_to_dataframe(entry_data, processor)
            dataframes.append(df)
        else:
            logger.warning(f"No data found for entry ID {entry_id}")

    return dataframes


def parse_arguments():
    parser = argparse.ArgumentParser(description='Run the update script with a specified update range.')
    parser.add_argument('-r', '--range', type=int, default=100, help='Number of entries to update.')
    parser.add_argument('-s', '--start_id', type=int, default=None, help='Start note ID for range selection.')
    parser.add_argument('-e', '--end_id', type=int, default=None, help='End note ID for range selection.')
    parser.add_argument('-u', '--username', type=str, required=True, help='Username for authentication.')
    parser.add_argument('-p', '--password', type=str, required=True, help='Password for authentication.')
    
    args = parser.parse_args()
    
    # Ensure at least one method is used (either range or ID range)
    if args.range is None and (args.start_id is None or args.end_id is None):
        parser.error("You must provide either an update range (-r) or a start and end note ID (-s and -e).")
    
    return args

def save_last_processed_id(elog_id):
    # Optionally implement saving to a file or database
    logger.info(f"Saved last processed ID: {elog_id}")

def get_last_processed_id(default_id):
    # If you wish to implement reading from a file later, you can uncomment this section
    # try:
    #     with open('last_processed_id.txt', 'r') as file:
    #         return int(file.read().strip())
    # except FileNotFoundError:
    return default_id

def process_data_to_dataframe(entry_data, processor):
    # Initialize an empty list for rows
    rows = []
    # Extract main text or default to empty string
    main_text = entry_data.get('text', '')

    # Combine all comment texts into one string
    comments = entry_data.get('comments', [])
    comments_text = " ".join(comment.get('text', '') for comment in comments)

    # Combine and clean the main text and comments
    combined_text = f"{main_text} {comments_text}".strip()
    
    # **Print 1: Before Augmentation**
    print(f"\n[DEBUG] Original Text for ID {entry_data.get('id')}: {combined_text[:200]}...")  # Truncate for readability
    # print(f"[DEBUG] Logs: {entry_data.get('logs', [])}\n")
    clean_text = processor.process_text(combined_text)
    augmented_text = clean_text
    # Augment text with logs before processing
    logs = entry_data.get('logs', [])  # Ensure logs exist
    cats = entry_data.get("categories", [])  # Extract categories
    
    
    # Only apply categories if they exist
    if cats:
        augmented_text = processor.augment_with_cats(augmented_text, cats)

    augmented_text = processor.augment_with_logs(augmented_text, logs)
    
    # **Print 2: After Augmentation**
    print(f"[DEBUG] Augmented Text: {augmented_text[:300]}...\n")

    
    vector = processor.generate_embeddings(augmented_text)

    # Convert and format creation date
    creation_date = processor.date_to_rfc3339(entry_data.get('creation_date', ''))

    # Serialize logs and categories as JSON
    logs_json = json.dumps(entry_data.get('logs', []))
    categories_json = json.dumps(entry_data.get('categories', []))

    # Append a dictionary for each processed entry to rows
    rows.append({
        'id': entry_data.get('id'),
        'vector': vector,
        'logs': logs_json,
        'categories': categories_json,
        'creation_date': creation_date
    })

    logger.info(f"Processed entry for ID {entry_data.get('id')}.")
    # Create a DataFrame from the list of rows and return
    df = pd.DataFrame(rows)
    return df

def run_update(update_range=None, start_id=None, end_id=None, username="", password=""):  

    
    try:
        

        # Instantiate the text processor and client
        logger.info("Initializing text processor and ElogFetcher client.")
        processor = TextProcessor(model_name="/app/ElogSemanticSearch/models/all-mpnet-base-v2")
        client = ElogFetcher()
        client.create_authenticated_session(username, password)


        # Ensure valid values for start_id and end_id
        if start_id is not None and start_id > 0:
            if end_id is None or end_id <= 0:  
                end_id = start_id - (update_range if update_range else 100)  # Default range 100 if not provided
                logger.info(f"End ID not provided. Calculating end_id = {start_id} - {update_range} = {end_id}")
            logger.info(f"Fetching entries from {start_id} to {end_id}.")
            most_recent_id = start_id  # Ensure `most_recent_id` has a value
            specified_id = end_id  # Ensure `specified_id` has a value
        else:
            # If start_id is not provided or invalid, get the most recent entry
            most_recent_id = client.check_most_recent_entry()
            specified_id = most_recent_id - (update_range if update_range else 100)  # Default 100 entries
            logger.info(f"Fetching the most recent {update_range} entries from {most_recent_id} to {specified_id}.")

        # Validate final start_id and end_id
        if most_recent_id <= 0 or specified_id <= 0:
            raise ValueError(f"Invalid start_id ({most_recent_id}) or end_id ({specified_id}). Ensure correct inputs.")


        # Fetch and process entries
        dataframes = fetch_and_process_entries(client, processor, most_recent_id, specified_id)
        
        logger.info("Initializing Qdrant client.")
        qdrant_client = QdrantClient(host="qdrant", port=6333)

        batch_size = 500  # Adjust the batch size as needed
        batch = []

        logger.info("Upserting data into Qdrant.")
        for df in dataframes:
            # Loop through each row in the DataFrame
            for index, row in df.iterrows():
                elog_id = row['id']
                vector = row['vector']
                # Deserialize categories and logs from JSON if stored as strings
                category_names = json.loads(row['categories']) if isinstance(row['categories'], str) else row['categories']
                log_names = json.loads(row['logs']) if isinstance(row['logs'], str) else row['logs']
                creation_date = row['creation_date']

                # Prepare payload for upsert
                payload = {
                    "category_names": category_names,
                    "log_names": log_names,
                    "creation_date": creation_date
                }
                # Append point structure to the batch
                batch.append(PointStruct(
                    id=elog_id,
                    vector=vector,
                    payload=payload
                ))

                # When batch size is reached, perform the upsert
                if len(batch) >= batch_size:
                    qdrant_client.upsert(collection_name="Elog_collection_768_METADATA", points=batch)
                    logger.info(f"Upserted a batch of {len(batch)} entries.")
                    batch = []  # Clear the batch for the next round

        # Upsert any remaining points that do not fill a complete batch
        if batch:
            qdrant_client.upsert(collection_name="Elog_collection_768_METADATA", points=batch)
            logger.info(f"Upserted the final batch of {len(batch)} entries.")

        logger.info("Data upsert complete.")
        return True  # Indicate success
    except Exception as e:
        logger.error(f"Update failed: {e}", exc_info=True)
        return False  # Indicate failure

if __name__ == "__main__":
    args = parse_arguments()
    run_update(args.range, args.start_id, args.end_id, args.username, args.password)

