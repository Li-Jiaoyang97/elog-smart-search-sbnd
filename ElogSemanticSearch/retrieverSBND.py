#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 04:26:57 2024

@author: hayden
"""
import os
import datetime
from ElogSemanticSearch.DataFetch.text_to_vectorSBND import TextProcessor

#from DataFetch.text_to_vector import TextProcessor
from ElogSemanticSearch.DataFetch.ElogFetcherQdrant import ElogFetcher
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging


"""
I created this retriver to work in a similar fashion to the LangChain ones. A retriever just takes metadata associated with a vector and returns text associated with that metadata.

You will need to adapt this retriever to whatever API your logbook uses to access data. The public release of this program will some pseudo code
"""


class ContextRetriever:
    def __init__(self):
        self.username = 'username'
        self.password = 'password'
        self.client = ElogFetcher()
        self.processor = TextProcessor(model_name="/app/ElogSemanticSearch/models/all-mpnet-base-v2")
        self.contexts = []  # List to store full entry details including metadata
        self.logger = logging.getLogger(__name__)

    def authenticate(self):
        """
        Authenticate the retriever using credentials from environment variables.
        """
        try:
            self.client.create_authenticated_session(self.username, self.password)
            self.logger.info("Authentication successful.")
        except Exception as e:
            self.logger.error(f"Failed to create authenticated session: {e}")
            raise Exception("Authentication Failed")

    def fetch_and_process_entry(self, entry_id, for_reranking=False):
        """
        Fetch and process a single entry.
        Args:
        - entry_id (int): The ID of the entry to fetch.
        - for_reranking (bool): Whether to process for reranking.
        Returns:
        - dict: Processed entry with 'id' and 'text'.
        """
        try:
            entry_data = self.client.get_entry(entry_id)
            if entry_data:
                text = (
                    self.process_entry_rerank(entry_data)
                    if for_reranking
                    else self.process_entry(entry_data)
                )
                return {"id": entry_id, "text": text}
            else:
                self.logger.warning(f"No data found for entry ID: {entry_id}")
        except Exception as e:
            self.logger.error(f"Failed to fetch or process entry {entry_id}: {e}")
        return None

    def fetch_and_process_entries_concurrently(self, entry_ids, for_reranking=False):
        """
        Fetch and process multiple entries concurrently using multithreading.
        Args:
        - entry_ids (list): List of entry IDs.
        - for_reranking (bool): Whether to process for reranking.
        Returns:
        - list: Processed entries.
        """
        results = []
        with ThreadPoolExecutor() as executor:
            future_to_id = {
                executor.submit(self.fetch_and_process_entry, entry_id, for_reranking): entry_id
                for entry_id in entry_ids
            }

            for future in as_completed(future_to_id):
                entry_id = future_to_id[future]
                try:
                    result = future.result()
                    if result:
                        results.append(result)
                        self.logger.debug(f"Processed entry ID: {entry_id}")
                except Exception as e:
                    self.logger.error(f"Error processing entry ID {entry_id}: {e}")

        self.logger.info(f"Processed {len(results)} entries concurrently.")
        return results

    def process_entry(self, entry_data):
        """
        Process a single entry to extract and format its metadata and text.
        Args:
        - entry_data (dict): The raw entry data.
        Returns:
        - str: Formatted entry text with metadata.
        """
        elog_id = entry_data.get('id')
        creation_date = self.processor.date_to_rfc3339(entry_data.get('creation_date', ''))
        main_text = entry_data.get('text', '')
        comments = entry_data.get('comments', [])

        # Ensure each update starts on a new line
        comments_text = "\n".join(
            f"Update {idx + 1}: {comment['text']}" for idx, comment in enumerate(comments)
        ) if comments else ""

        combined_text = f"{main_text}\n{comments_text}".strip()
        final_combined_text = self.processor.process_text(combined_text)

        # Format the entry with metadata
        entry_output = f"Elog ID: {elog_id}\nOccurred at: {creation_date}\nEntry Text: {final_combined_text}"
        return entry_output

    def process_entry_rerank(self, entry_data):
        """
        Process a single entry for reranking by extracting its combined text.
        Args:
        - entry_data (dict): The raw entry data.
        Returns:
        - str: Processed entry text for reranking.
        """
        main_text = entry_data.get('text', '')
        comments = entry_data.get('comments', [])

        # Ensure each update starts on a new line
        comments_text = "\n".join(
            f"Update {idx + 1}: {comment['text']}" for idx, comment in enumerate(comments)
        ) if comments else ""

        combined_text = f"{main_text}\n{comments_text}".strip()
        final_combined_text = self.processor.process_text(combined_text)
        return final_combined_text

    def get_combined_context(self):
        """
        Combine all fetched entries into a single formatted context string.
        Returns:
        - str: Combined entries with a header.
        """
        header = f"Current date and time of processing: {self.current_datetime}\n\n---\n\n"
        combined_entries = "\n\n---\n\n".join(self.contexts)
        return header + combined_entries
    
    def fetch_and_process_entry_rerank(self, entry_id):
        """
        Fetch and process an entry for reranking.
        Args:
        - entry_id (int): The ID of the entry to fetch.
        Returns:
        - dict: A dictionary with 'id' and 'text' (processed text for scoring).
        """
        try:
            entry_data = self.client.get_entry(entry_id)
            if entry_data:
                full_entry_text = self.process_entry_rerank(entry_data)
                logging.debug(f"Fetched and processed entry {entry_id} for reranking.")
                return {"id": entry_id, "text": full_entry_text}
            else:
                logging.warning(f"No data found for entry ID: {entry_id}")
                return None
        except Exception as e:
            logging.error(f"Failed to fetch or process data for entry {entry_id}: {e}")
            return None