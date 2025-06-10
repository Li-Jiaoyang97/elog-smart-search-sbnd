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
import torch
from sentence_transformers import SentenceTransformer

class TextProcessor:
    def __init__(self, model_name):
        self.model = SentenceTransformer(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # print(f"Using {'CUDA' if self.device.type == 'cuda' else 'CPU'} for processing.")
       
        #  # Enable GPU for spaCy
        # spacy.require_gpu()
        #  #For anonymization
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
        #text = self.anonymize_text(text)
        return text

    def generate_embeddings(self, text):
        """
        Generates embeddings for the given text using SentenceTransformers.
        This method handles tokenization and model inference internally.
        This is needed because QDrant does not support the actual embedding step.
        """
        # The encode method of SentenceTransformer directly returns the embeddings as a tensor
        embeddings = self.model.encode(text, convert_to_tensor=True, device=self.device)
        return embeddings.cpu().numpy()  # Convert tensor to numpy array if needed. Sometimes you can get errors if the embedding is not a numpy array.

    def date_to_rfc3339(self, date_str):
        """
        Converts date from string format to RFC 3339 format. Qdrant does best storing time in this manner.
        """
        try:
            dt = datetime.strptime(date_str, "%b %d, %Y %I:%M:%S %p")
            return dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        except ValueError:
            return None


    # def load_exclusions(self, file_path):
    #     """Load terms that should not be anonymized from a text file."""
    #     with open(file_path, 'r') as file:
    #         exclusions = set(line.strip().lower() for line in file)
    #     return exclusions
    
    # def anonymize_text(self, text, save_entities=True, entities_file="anonymized_entities.txt"):
    #     """
    #     Anonymizes sensitive information in the text. Optionally saves original entities to a file.
    #     """
    #     # Regex patterns for sensitive data
    #     email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,7}\b')
    #     ssn_pattern = re.compile(r'\b\d{3}-\d{2}-\d{4}\b')
    #     credit_card_pattern = re.compile(r'\b(?:\d{4}[ -]?){3}\d{4}\b')
    #     driver_license_pattern = re.compile(r'\b[A-Z]{1,2}-\d{6,8}\b')
        
    #     date_keywords = {
    #         'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday',
    #         'jan', 'january', 'feb', 'february', 'mar', 'march', 'apr', 'april', 'may', 'jun',
    #         'june', 'jul', 'july', 'aug', 'august', 'sep', 'september', 'oct', 'october', 
    #         'nov', 'november', 'dec', 'december', 'today', 'tomorrow', 'yesterday', 'weekend',
    #         'mon', 'tues', 'wed', 'thu', 'fri', 'sat', 'sun'
    #     }
    
    #     doc = self.nlp(text)
    #     anonymized_text = text
    #     ner_anonymized_words = []
        
    #     # Define specific tokens for PERSON and LOCATION entities
    #     entity_tokens = {
    #         'PERSON': '[PERSON]',
    #         'GPE': '[LOCATION]'
    #     }
        
    #     for ent in reversed(doc.ents):
    #         entity_text_lower = ent.text.lower()
            
    #         if any(keyword in entity_text_lower for keyword in date_keywords) or any(char.isdigit() for char in ent.text):
    #             continue
            
    #         if ent.label_ in entity_tokens and ent.text[0].isupper() and entity_text_lower not in self.exclusions:
    #             replacement_token = entity_tokens[ent.label_]
    #             ner_anonymized_words.append(ent.text)  # Log original entity
    #             anonymized_text = anonymized_text[:ent.start_char] + replacement_token + anonymized_text[ent.end_char:]
    
    #     # Optionally save the original NER-identified entities to a file
    #     if save_entities and ner_anonymized_words:
    #         with open(entities_file, 'a') as file:
    #             for word in ner_anonymized_words:
    #                 file.write(word + "\n")
    
    #     # Apply regex patterns for anonymization, now uniformly using "[REDACTED]"
    #     anonymized_text = email_pattern.sub("[REDACTED]", anonymized_text)
    #     anonymized_text = ssn_pattern.sub("[REDACTED]", anonymized_text)
    #     anonymized_text = credit_card_pattern.sub("[REDACTED]", anonymized_text)
    #     anonymized_text = driver_license_pattern.sub("[REDACTED]", anonymized_text)

    #     return anonymized_text

    


# Example of using the TextProcessor class if needed. You don't need the print statements after intial testing.
# processor = TextProcessor()
# text = "<p>Hello, world! <a href='http://example.com'>Link</a></p>"
# clean_text = processor.process_text(text)
# embeddings = processor.generate_embeddings(clean_text)
# print("Cleaned Text:", clean_text)
# print("Embeddings Shape:", embeddings.shape)
