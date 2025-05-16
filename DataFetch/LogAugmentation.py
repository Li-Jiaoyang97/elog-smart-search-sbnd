import re
import torch
from sentence_transformers import SentenceTransformer
import spacy

class TextProcessor:
    def __init__(self, model_name='all-mpnet-base-v2'):
        self.model = SentenceTransformer(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using {'CUDA' if self.device.type == 'cuda' else 'CPU'} for processing.")
        
        # Enable GPU for spaCy
        spacy.require_gpu()
        self.nlp = spacy.load("en_core_web_trf")
        self.exclusions = self.load_exclusions('exclusions.txt')

    def load_exclusions(self, file_path):
        """Load terms that should not be anonymized from a text file."""
        try:
            with open(file_path, 'r') as file:
                exclusions = set(line.strip().lower() for line in file)
            return exclusions
        except FileNotFoundError:
            return set()

    def preprocess_text(self, text):
        """
        Adds spaces around special characters to improve name detection.
        """
        text = re.sub(r'([a-zA-Z])(\()', r'\1 (', text)  # Add space before parentheses
        text = re.sub(r'(\))([a-zA-Z])', r') \2', text)  # Add space after parentheses
        return text

    def load_previous_entities(self, entities_file="anonymized_entities.txt"):
        """Load previously anonymized entities from the given file."""
        try:
            with open(entities_file, 'r') as file:
                return [line.strip() for line in file.readlines()]
        except FileNotFoundError:
            return []

    def apply_previous_anonymization(self, text, previous_entities):
        """
        Apply previously anonymized entities to the current text.
        """
        for entity in previous_entities:
            # Replace any exact matches of previously anonymized entities
            text = re.sub(r'\b{}\b'.format(re.escape(entity)), '[PERSON]', text)
        return text

    def apply_regex_name_detection(self, text):
        """
        Apply regex to detect capitalized names that are close to non-alphabetic characters.
        """
        name_pattern = re.compile(r"\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)*(?=\W|\d)")
        names_found = re.findall(name_pattern, text)
        return names_found

    def anonymize_text(self, text, save_entities=True, entities_file="anonymized_entities.txt"):
        """
        Anonymizes sensitive information in the text. Includes recursive checks with previously anonymized entities.
        """
        # Preprocess the text to add spaces around parentheses and numbers
        text = self.preprocess_text(text)
        
        # Load previously anonymized entities
        previous_entities = self.load_previous_entities(entities_file)
        
        # Run NER to detect names
        doc = self.nlp(text)
        anonymized_text = text
        ner_anonymized_words = []

        # Use NER to anonymize detected names
        for ent in reversed(doc.ents):
            if ent.label_ == 'PERSON' and ent.text not in self.exclusions:
                replacement_token = '[PERSON]'
                ner_anonymized_words.append(ent.text)
                anonymized_text = anonymized_text[:ent.start_char] + replacement_token + anonymized_text[ent.end_char:]
        
        # Use regex to capture additional names (e.g., missed by NER)
        regex_names = self.apply_regex_name_detection(anonymized_text)
        for name in regex_names:
            anonymized_text = anonymized_text.replace(name, '[PERSON]')

        # Optionally save the original NER-identified entities to a file, only if they aren't already present
        if save_entities:
            with open(entities_file, 'a') as file:
                for word in ner_anonymized_words + regex_names:
                    if word not in previous_entities:
                        file.write(word + "\n")

        # Perform second pass using previous anonymized names
        anonymized_text = self.apply_previous_anonymization(anonymized_text, previous_entities)

        return anonymized_text

    def generate_embeddings(self, text):
        """
        Generates embeddings for the given text using SentenceTransformers.
        """
        embeddings = self.model.encode(text, convert_to_tensor=True, device=self.device)
        return embeddings.cpu().numpy()

    def process_text(self, text, logs=None):
        """
        Full processing pipeline from raw text to clean text, augmented with logs metadata.
        """
        # Clean the main text
        text = self.clean_unicode_characters(text)
        text = self.clean_text(text)

        # Augment with logs metadata
        if logs:
            text = self.augment_with_logs(text, logs)
        
        # Anonymize the text with NER, regex, and recursive entity checks
        text = self.anonymize_text(text)
        return text

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

        return augmented_text  

    def clean_unicode_characters(self, text):
        """
        Removes specific unicode characters and cleans the text.
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
        """
        from bs4 import BeautifulSoup
        import html
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
