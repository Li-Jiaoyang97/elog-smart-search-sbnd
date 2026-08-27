#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ElogFetcherQdrantSBND.py
---------------------
Drop-in replacement for the original ElogFetcherQdrant.py.
Uses ecl_api to fetch entries from the SBND ECL elog system
and parses the returned XML into the dict format the pipeline expects.

@author: hayden
         jiaoyang updated for SBND ECL
"""

import os
import re
import time
import logging
from ECLConfig import ECL_CATEGORIES

from ecl_api import ECL
import xml.etree.ElementTree as ET
from urllib.parse import quote
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)


class EntryFetchError(Exception):
    """
    A fetch failed for a reason that is *not* "this entry doesn't exist".

    Raised for transport-level problems (connection reset, timeout, HTTP
    error). These are usually transient and worth retrying, so callers can
    tell them apart from a genuine gap in the ID sequence — which is signalled
    by get_entry() returning None.
    """


class EntryParseError(EntryFetchError):
    """
    ECL returned data but it could not be parsed into an entry.

    Deterministic: retrying sends the identical request and gets the identical
    unparseable bytes back, so the retry loop deliberately does not catch this.
    Subclasses EntryFetchError so callers that only care about "did the fetch
    fail" can catch the parent.
    """


# ECL sometimes serializes a raw '&' in subject/field text instead of escaping
# it as '&amp;' (e.g. subject="PDS X-ARAPUCA & PMT tests"), which makes the
# XML not well-formed. Escape only the offending bare ampersands so parsing
# doesn't blow up on otherwise-valid entries.
_UNESCAPED_AMP_RE = re.compile(r'&(?!amp;|lt;|gt;|apos;|quot;|#\d+;|#x[0-9a-fA-F]+;)')

# ECL also serializes raw '"' inside attribute values instead of escaping them
# as '&quot;' (e.g. subject="... removing "87" from the dictionary ..."),
# which terminates the attribute value early and breaks well-formedness.
# Every attribute in ECL's response starts at the beginning of a line as
# name="value". A quote is treated as the *real* closing delimiter only if
# what follows it looks like the next attribute starting or the tag closing
# (rather than assuming it's simply the last quote on the line) — anything
# else between the opening quote and that point is stray content to escape.
_ATTR_LINE_RE = re.compile(
    r'^(?P<indent>[ \t]*)(?P<name>[A-Za-z_][\w.:-]*)="(?P<value>.*?)"'
    r'(?=[ \t]+[A-Za-z_][\w.:-]*="|[ \t]*/?>[ \t]*$|[ \t]*$)',
    re.MULTILINE,
)


def _escape_inner_quotes(match):
    value = match.group('value').replace('"', '&quot;')
    return f"{match.group('indent')}{match.group('name')}=\"{value}\""


def _sanitize_xml(xml_string):
    xml_string = _ATTR_LINE_RE.sub(_escape_inner_quotes, xml_string)
    return _UNESCAPED_AMP_RE.sub('&amp;', xml_string)


class ElogFetcher:
    def __init__(self):
        self.user     = os.environ.get('ECL_USERNAME')
        self.password = os.environ.get('ECL_PASSWORD')
        self.base_url = os.environ.get('ECL_URL', 'https://dbweb1.fnal.gov:8443/ECL/sbnd/E')
        self.ecl      = None

    def create_authenticated_session(self, username, password):
        """
        Create an authenticated ECL session.
        Must be called before get_entry() or check_most_recent_entries_allCategories().
        """
        try:
            self.ecl = ECL(url=self.base_url, user=username, password=password)
            self.user     = username
            self.password = password

            # ── Verify the session actually works (equivalent of status_code == 200)
            test = self.ecl.search(limit=1)
            if not test:
                raise Exception("Authentication succeeded but session returned no data — likely bad credentials.")

            logger.info("Authentication successful.")
            return True

        except Exception as e:
            logger.error(f"Authentication failed: {e}")
            self.ecl = None
            return False
        
    def get_entry(self, entry_id, retries=3, backoff=2.0):
        """
        Fetch a single elog entry by ID and return it as a dict.

        Returns a dict with keys:
            id            - int
            text          - str  (form fields flattened into one string)
            logs          - list of str  (e.g. ["CRT"])
            categories    - list of str  (e.g. ["CRT"])
            creation_date - str  (e.g. "03/18/2026 10:04:46")
            comments      - list of dicts with keys 'text' and 'modified_date'

        Returns None ONLY when ECL has no entry at this ID — i.e. a genuine gap
        in the sequence, which is normal and expected.

        Raises:
            EntryParseError - ECL returned data that could not be parsed.
            EntryFetchError - the request itself failed (transport, auth).

        This distinction matters: callers count a None as a harmless gap, so
        folding failures into it silently discards real entries. Transport
        errors are retried up to `retries` times with exponential backoff
        (matching get_entry_ids_allCategories); parse errors are not, since
        the same bytes would come back.
        """
        if self.ecl is None:
            raise EntryFetchError(
                "Session not authenticated. Call create_authenticated_session() first."
            )

        delay = backoff
        for attempt in range(1, retries + 1):
            try:
                xml_string = self.ecl.get_entry(entry_id=int(entry_id))
            except Exception as e:
                if attempt < retries:
                    logger.warning(
                        f"Entry {entry_id}: fetch attempt {attempt}/{retries} failed "
                        f"({type(e).__name__}: {e}), retrying in {delay}s..."
                    )
                    time.sleep(delay)
                    delay *= 2
                    continue
                raise EntryFetchError(
                    f"Failed to fetch entry {entry_id} after {retries} attempts: "
                    f"{type(e).__name__}: {e}"
                ) from e

            if not xml_string:
                logger.debug(f"No data returned for entry ID {entry_id} (nonexistent).")
                return None

            # Outside the try: a parse failure must not be retried, and must not
            # be reported as a transport error.
            return self._parse_entry(entry_id, xml_string)

    def check_most_recent_entry_perCategory(self, category='Shift', after="1days", limit=1, debug=False):
        """
        Return the ID of the most recent entry in the elog.
        Useful for determining the range to ingest.
        """
        if self.ecl is None:
            logger.error("Session not authenticated. Call create_authenticated_session() first.")
            return None
        
        try:
            encoded_cat = quote(category, safe='/-')
            if debug:
                print(f"\n  [{category}] Encoded as: {encoded_cat}")
            rawtext = self.ecl.search(category=encoded_cat, limit=limit, after=after)
            xml     = ET.fromstring(_sanitize_xml(rawtext))
            entries = xml.findall('./entry')

            if entries:
                # Sort by ID descending to ensure we get the highest (most recent)
                entries_sorted = sorted(entries, key=lambda e: int(e.attrib.get('id', 0)), reverse=True)
                entry_id = entries_sorted[0].attrib.get('id')
                if entry_id:
                    return int(entry_id)

            logger.warning("Could not determine most recent entry ID.")
            return None

        except Exception as e:
            logger.error(f"Failed to get most recent entry: {e}")
            return None

    def check_most_recent_entries_allCategories(self, categories=None, after='7days', limit=5, debug=False):
        if self.ecl is None:
            logger.error("Session not authenticated.")
            return None

        if categories is None:
            categories = ECL_CATEGORIES

        results = {}  # category -> most recent id
        for category_name in categories:
            try:
                encoded = quote(category_name, safe='/-')
                text = self.ecl.search(category=encoded, limit=limit, after=after)
                xml = ET.fromstring(_sanitize_xml(text))
                entries = xml.findall('./entry')
                ids = [int(e.attrib['id']) for e in entries if e.attrib.get('id')]
                if ids:
                    results[category_name] = max(ids)
                    if debug:
                        print(f"  [{category_name}] Most recent ID: {results[category_name]}")
                else:
                    results[category_name] = None
                    if debug:
                        print(f"  [{category_name}] No entries found")
            except Exception as e:
                results[category_name] = None
                logger.warning(f"  [{category_name}] Exception: {type(e).__name__}: {e}")

        return results

    def get_entry_ids_perCategory(self, category, after='7days', limit=50, debug=False):
        """
        Return a descending-sorted list of entry IDs for a single category.
        """
        if self.ecl is None:
            logger.error("Session not authenticated.")
            return []

        try:
            encoded = quote(category, safe='/-')
            text = self.ecl.search(category=encoded, limit=limit, after=after)
            xml = ET.fromstring(_sanitize_xml(text))
            entries = xml.findall('./entry')
            ids = sorted(
                [int(e.attrib['id']) for e in entries if e.attrib.get('id')],
                reverse=True
            )
            if debug:
                print(f"  [{category}] Found {len(ids)} entries")
            return ids
        except Exception as e:
            logger.warning(f"  [{category}] Exception: {type(e).__name__}: {e}")
            return []

    def get_entry_ids_allCategories(self, categories=None, after='7days', limit=50, debug=False, max_workers=10):
        """
        Return a dictionary with categories as keys and lists of entry IDs as values.
        Each list is sorted in descending order (most recent first).
        Category searches run in parallel (max_workers threads).
        """
        if self.ecl is None:
            logger.error("Session not authenticated.")
            return {}

        if categories is None:
            categories = ECL_CATEGORIES

        def fetch_category(category_name, retries=3, backoff=2.0):
            import time
            encoded = quote(category_name, safe='/-')
            for attempt in range(retries):
                try:
                    text = self.ecl.search(category=encoded, limit=limit, after=after)
                    xml = ET.fromstring(_sanitize_xml(text))
                    entries = xml.findall('./entry')
                    ids = sorted(
                        [int(e.attrib['id']) for e in entries if e.attrib.get('id')],
                        reverse=True
                    )
                    if debug:
                        print(f"  [{category_name}] Found {len(ids)} entries")
                    return category_name, ids
                except Exception as e:
                    if attempt < retries - 1:
                        logger.warning(f"  [{category_name}] Attempt {attempt+1} failed ({type(e).__name__}), retrying in {backoff}s...")
                        time.sleep(backoff)
                        backoff *= 2
                    else:
                        logger.error(f"  [{category_name}] Exception after {retries} attempts: {type(e).__name__}: {e}")
                        return category_name, []

        result = {}
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(fetch_category, cat): cat for cat in categories}
            for future in as_completed(futures):
                category_name, ids = future.result()
                result[category_name] = ids

        return result
    
    

    # ── Internal XML parser ───────────────────────────────────────────────────
    def _parse_entry(self, entry_id, xml_string):
        """
        Parse an ECL XML string into the dict format the pipeline expects.

        Handles two entry types:
          - Form-based entries  (e.g. Alarm Acknowledged)
          - Free-text entries   (e.g. BNB narrow bunch study)
        """
        try:
            root = ET.fromstring(_sanitize_xml(xml_string))
        except ET.ParseError as e:
            # Raise rather than return None: ECL *did* return data for this ID,
            # so the entry exists and silently treating it as a gap would lose it.
            raise EntryParseError(
                f"Failed to parse XML for entry {entry_id}: {e}"
            ) from e

        # ── Header fields from XML attributes ────────────────────────────────
        #author        = root.attrib.get('author', '')
        category      = root.attrib.get('category', '')
        timestamp     = root.attrib.get('timestamp', '')
        subject       = root.attrib.get('subject', '')

        # ── Build text string ─────────────────────────────────────────────────
        text_parts = []

        # Include subject if present
        if subject and subject.strip():
            text_parts.append(f"Subject: {subject.strip()}.")

        # Form-based entry — flatten all fields into label: value sentences
        form = root.find('form')
        if form is not None:
            form_name = form.attrib.get('name', '')
            if form_name:
                text_parts.append(f"Form: {form_name}.")
            for field in form.findall('field'):
                name  = field.attrib.get('name', '').strip()
                value = (field.text or '').strip()
                if name and value:
                    text_parts.append(f"{name}: {value}.")

        # Free-text entry — use the text element directly
        text_el = root.find('text')
        if text_el is not None and text_el.text:
            text_parts.append(text_el.text.strip())

        # If neither form nor text element found, try root text
        if not text_parts and root.text and root.text.strip():
            text_parts.append(root.text.strip())

        # ── Comments ──────────────────────────────────────────────────────────
        comments = []
        for comment in root.findall('comment'):
            comment_text = (comment.text or '').strip()
            if comment_text:
                comments.append({
                    'text':          comment_text,
                    'modified_date': comment.attrib.get('timestamp', '')
                })
            else:
                comments.append({
                    'text':          'No Comment Text Provided',
                    'modified_date': 'No Date Provided'
                })

        # ── Assemble final dict ───────────────────────────────────────────────
        parts = category.split('/', 1) if category else []
        top_category = parts[0] if parts else ''
        subcategory  = parts[1] if len(parts) > 1 else ''
        return {
            'id':            int(entry_id),
            'text':          ' '.join(text_parts),
            'logs':          [top_category] if top_category else [],
            'categories':    [category]     if category     else [],
            'subject':       (subject or '').strip(),
            'top_category':  top_category or '',
            'subcategory':   subcategory or '',
            'creation_date': timestamp or '',
            'comments':      comments,
        }
    
    def _print_entry(self, entry):
        """
        Pretty-print an entry dict after parsing.
        """
        print(f"  ID:            {entry['id']}")
        print(f"  Log:           {entry['logs']}")
        print(f"  Date:          {entry['creation_date']}")
        print(f"  Categories:    {entry['categories']}")
        print(f"  Subject:       {entry.get('subject', '')}")
        print(f"  Text preview:  {entry['text'][:120]}...")
        print(f"  Comments:      {len(entry['comments'])} comment(s)")

        for i, c in enumerate(entry['comments'], 1):
            print(f"    [{i}] {c['modified_date']} — {c['text'][:60]}...")
    