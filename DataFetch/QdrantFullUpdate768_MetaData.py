#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 01:02:03 2024

@author: hayden
"""

base_url = "https://www-bd.fnal.gov/Elog/"

import os
from ElogFetcherQdrant import ElogFetcher
from text_to_vector import TextProcessor
import pandas as pd
import json
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct
from qdrant_client.models import VectorParams, Distance
 



'''
docker run -p 6333:6333 -p 6334:6334 \
    -v $(pwd)/qdrant_storage:/qdrant/storage:z \
    qdrant/qdrant
'''

    
def save_last_processed_id(elog_id):
    with open('last_processed_id.txt', 'w') as file:
        file.write(str(elog_id))
        print(f"Saved last processed ID: {elog_id}")
        
def get_last_processed_id(default_id):
    # try:
    #     with open('last_processed_id.txt', 'r') as file:                   #//This was commented out because it was not that useful. I'll put it in later when I find a way to implement it better.
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
    
    # Print 1: Before Augmentation**
    # print(f"\n[DEBUG] Original Text for ID {entry_data.get('id')}: {combined_text[:200]}...")  
    # print(f"[DEBUG] Logs: {entry_data.get('logs', [])}\n")
    
    # print(f"[DEBUG] Logs: {entry_data.get('categories', [])}\n")
    
    clean_text = processor.process_text(combined_text)
    # Augment text with logs before processing
    logs = entry_data.get('logs', [])  # Ensure logs exist
    cats = entry_data.get('categories', [])
    
    augmented_text = clean_text
    
    # Only augment with categories if they exist
    # if cats:
    #     augmented_text = processor.augment_with_cats(augmented_text, cats)
    
    augmented_text = processor.augment_with_logs(augmented_text, logs)

    
    # **Print 2: After Augmentation**
    # print(f"[DEBUG] Augmented Text: {augmented_text[:200]}...\n")

    
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

    print(f"Processed entry for ID {entry_data.get('id')}.")
    # Create a DataFrame from the list of rows and return
    df = pd.DataFrame(rows)
    return df

def main():
    # Instantiate the text processor
    # processor = TextProcessor('ElogSemanticSearch\models\mxbai-embed-large-v1')
    
    processor = TextProcessor('ElogSemanticSearch/models/all-mpnet-base-v2')
    # Instantiate the client to fetch entries
    client = ElogFetcher()
    # Create authenticated session
    

    
    client.create_authenticated_session(USERNAME, PASSWORD)
    
    custom_upper_range=273000
    #most_recent_id = client.check_most_recent_entry()
    
    update_range = 1000
    most_recent_id = custom_upper_range
    # default_start_id = 261050
    default_start_id = most_recent_id - update_range
   
    print(most_recent_id)
    # most_recent_id = 228000
    # Get the last processed ID or use the default
    specified_id = get_last_processed_id(default_start_id)
    # List to hold all dataframes
    dataframes = []

    for entry_id in range(most_recent_id, specified_id - 1, -1):
        entry_data = client.get_entry(str(entry_id))
        
        if entry_data:  # Check if the entry data is valid (not empty)
            df = process_data_to_dataframe(entry_data, processor)
            dataframes.append(df)
    
    
    #save_last_processed_id(most_recent_id)
        # Initialize Qdrant client
    
    Qclient = QdrantClient(host="localhost", port=6333)
    
    batch_size = 500  # Adjust the batch size as needed
    batch = []
    
    # Iterate through each DataFrame
    for df in dataframes:
        # Loop through each row in the DataFrame
        for index, row in df.iterrows():
            elog_id = row['id']
            vector = row['vector']  # Assuming you have a deserialize_vector function
    
            category_names = json.loads(row['categories']) if isinstance(row['categories'], str) else row['categories']
            log_names = json.loads(row['logs']) if isinstance(row['logs'], str) else row['logs']
            creation_date = row['creation_date']
            # print(elog_id)
           
            # Prepare payload for upsert
            payload = {
                "category_names": category_names,
                "log_names": log_names,
                "creation_date": creation_date
            }
            # print(payload)
            # Append point structure to the batch
            batch.append(PointStruct(
                id=elog_id,
                vector=vector,
                payload=payload
            ))
            
            # When batch size is reached, perform the upsert
            if len(batch) >= batch_size:
                # Qclient.upsert(collection_name="Elog_collection_1024_METADATA", points=batch)
                Qclient.upsert(collection_name="Elog_collection_METADATA_NoCats", points=batch)
                print(f"Upserted a batch of {len(batch)} entries.")
                batch = []  # Clear the batch for the next round
    
    # Upsert any remaining points that do not fill a complete batch
    if batch:
        # Qclient.upsert(collection_name="Elog_collection_1024_METADATA", points=batch)
        Qclient.upsert(collection_name="Elog_collection_METADATA_NoCats", points=batch)
        print(f"Upserted the final batch of {len(batch)} entries.")
    
    print("Data upsert complete.")


if __name__ == "__main__":
    USERNAME = os.environ.get('USERNAME')
    PASSWORD = os.environ.get('PASSWORD')
    main()













