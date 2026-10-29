#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 21:33:21 2024

@author: hayden
"""

#import spacy
import re
from datetime import datetime
from bs4 import BeautifulSoup
import html
#import torch
from sentence_transformers import SentenceTransformer

class TextProcessor:
    def __init__(self, model_name):
        self.model = SentenceTransformer(model_name)
        #self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # print(f"Using {'CUDA' if self.device.type == 'cuda' else 'CPU'} for processing.")
       
        #  # Enable GPU for spaCy
        # spacy.require_gpu()
        # self.nlp = spacy.load("en_core_web_trf")
        # self.exclusions = self.load_exclusions('exclusions.txt')


    def augment_with_logs(self, text, logs):
        """
        Augments text with log metadata using a structured format to improve clustering.
        - Removes 'Operations' if other logs are present to prevent overrepresentation.
        - Reinforces log-text relationships to improve soft clustering.
        """
        # Remove 'Operations' only if other logs exist
        if 'Operations' in logs and len(logs) > 1:
            logs = [log for log in logs if log != 'Operations']

        # Format logs in brackets
        logs_str = " ".join(f"[{log}]" for log in logs)

        # **Reinforce Log-Text Relationship**
        logs_natural_text = f"This entry is related to {', '.join(logs)}. " if logs else ""

        # Combine everything
        augmented_text = f"{logs_str} {logs_natural_text}{text}"

        return augmented_text  # No need for .strip() since logs are always present

    def augment_with_cats(self, text, categories):
        """
        Augments text with category metadata while keeping natural flow for embeddings.
        - Categories are introduced as subsystems of logs.
        - Uses structured yet readable phrasing.
        """
        cats_str = " ".join(f"[{cat}]" for cat in categories)

        if len(categories) == 1:
            cats_natural_text = f"This entry involves {categories[0]}. "
        elif len(categories) == 2:
            cats_natural_text = f"This entry involves {categories[0]} and {categories[1]}. "
        elif len(categories) > 2:
            cats_natural_text = f"This entry involves {', '.join(categories[:-1])}, and {categories[-1]}. "
        else:
            cats_natural_text = ""

        return f"{cats_str}{cats_natural_text}{text}"


    def clean_unicode_characters(self, text):
        """
        Removes specific unicode characters and cleans the text. 
        Sometimes unicode is left in the text from online logs, so this is a good tool to have in case.
        """
        replacements = {
            '\u00AD': '', '\u200C': '', '\u200D': '',
            '\u200E': '', '\u200F': '', '\uFEFF': ''
        }
        text = re.sub('|'.join(re.escape(key) for key in replacements.keys()), '', text)
        space_chars = ['\u2009', '\u200A', '\u2003', '\u2002']
        text = re.sub('|'.join(re.escape(key) for key in space_chars), ' ', text)
        text = re.sub(r'[\u0000-\u001F\u007F-\u009F]', '', text)
        return text

    def clean_text(self, text):
        """
        Decodes HTML entities, removes HTML tags, and cleans the text.
        Adjusts list items by replacing them with a period and a space.
        Good if you need to pull text right from html.
        """
        # Decode HTML entities
        text = html.unescape(text)
        
        # Parse HTML content
        soup = BeautifulSoup(text, "html.parser")
    
        # Handle <li> tags by replacing them with '. '
        for li in soup.find_all('li'):
            li.replace_with('. ' + li.text.strip() + '')  # Prepend a period and space before the list item text
    
        # Get clean text from modified HTML
        text = soup.get_text()
    
        # Additional cleaning steps
        text = re.sub(r'\u00a0|\u200b', ' ', text)  # Replace non-breaking spaces and zero-width spaces
        text = re.sub(r'http\S+', '[LINK]', text)  # Replace URLs with '[LINK]'
        text = re.sub(r'\s+', ' ', text).strip()  # Normalize whitespace
        text = re.sub(r'[\u0000-\u001F\u007F-\u009F\u00b0\u2019\u00a0\u200b]', ' ', text)
        # Replace multiple spaces with a single space
        text = re.sub(' +', ' ', text.replace('\n', ' ').replace('\r', ' '))
        return text

    def process_text(self, text):
        """
        Full processing pipeline from raw HTML to clean text.
        """
        text = self.clean_unicode_characters(text)
        text = self.clean_text(text)
        return text

    def generate_embeddings(self, text):
        """
        Generates embeddings for the given text using SentenceTransformers.
        This method handles tokenization and model inference internally.
        This is needed because QDrant does not support the actual embedding step.
        """
        
        return self.model.encode(text, convert_to_numpy=True)

        
        # The encode method of SentenceTransformer directly returns the embeddings as a tensor
        #embeddings = self.model.encode(text, convert_to_tensor=True, device=self.device)
        #return embeddings.cpu().numpy()  # Convert tensor to numpy array if needed. Sometimes you can get errors if the embedding is not a numpy array.

    def date_to_rfc3339(self, date_str):
        """
        Converts date from string format to RFC 3339 format. Qdrant does best storing time in this manner.
        """
        if not date_str:
            return None
        formats = [
            "%b %d, %Y %I:%M:%S %p",     # Mar 18, 2026 10:04:46 AM  (original AD elog)
            "%m/%d/%Y %H:%M:%S",         # 03/18/2026 10:04:46
            "%Y-%m-%dT%H:%M:%S",         # 2026-03-18T10:04:46
            "%Y-%m-%dT%H:%M:%S%z",       # 2026-03-18T10:04:46-05:00
            "%Y-%m-%d %H:%M:%S",         # 2026-03-18 10:04:46
            "%a, %d %b %Y %H:%M:%S %z",  # Wed, 18 Mar 2026 10:04:46 -0500
        ]
        for fmt in formats:
            try:
                dt = datetime.strptime(date_str, fmt)
                return dt.strftime("%Y-%m-%dT%H:%M:%SZ")
            except ValueError:
                continue
        import logging
        logging.getLogger(__name__).warning(f"Unrecognized date format: {date_str!r}")
        return None



# Example of using the TextProcessor class if needed. You don't need the print statements after intial testing.
# processor = TextProcessor()
# text = "<p>Hello, world! <a href='http://example.com'>Link</a></p>"
# clean_text = processor.process_text(text)
# embeddings = processor.generate_embeddings(clean_text)
# print("Cleaned Text:", clean_text)
# print("Embeddings Shape:", embeddings.shape)
