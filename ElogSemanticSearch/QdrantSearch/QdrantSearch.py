#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 04:13:08 2024

@author: hayden
@modified by jiaoyang for SBND Smart Search
"""


from qdrant_client import models, QdrantClient
from sentence_transformers import SentenceTransformer, CrossEncoder
from datetime import datetime, timedelta, timezone
import webbrowser
import logging
import re
import numpy as np

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


# "2026", "2026-03", "2026-03-18" (month/day may be one digit)
_PARTIAL_DATE = re.compile(r"^(\d{4})(?:-(\d{1,2}))?(?:-(\d{1,2}))?$")


def _parse_date_bound(value, is_end):
    """
    Turn one start/end date string into a (range_key, naive UTC datetime) bound
    on creation_date (stored as e.g. "2015-10-20T15:02:39Z"). A partial date
    spans its whole period: as a start it means the first instant of it, as an
    end it means "before the next period starts" (lt), so end "2026-03" keeps
    March 31 entries. A full datetime is used as-is (gte/lte). Raises ValueError.
    """
    value = value.strip()
    m = _PARTIAL_DATE.match(value)
    if m:
        year, month, day = int(m[1]), int(m[2] or 1), int(m[3] or 1)
        start = datetime(year, month, day)
        if m[3]:
            after = start + timedelta(days=1)
        elif m[2]:
            after = datetime(year + month // 12, month % 12 + 1, 1)
        else:
            after = datetime(year + 1, 1, 1)
        return ("lt", after) if is_end else ("gte", start)
    bound = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if bound.tzinfo:
        bound = bound.astimezone(timezone.utc).replace(tzinfo=None)
    return ("lte" if is_end else "gte"), bound


def date_range(start_date=None, end_date=None):
    """
    Optional start/end dates -> creation_date range bounds as RFC3339 strings,
    e.g. ("2026", "2026") -> {"gte": "2026-01-01T00:00:00Z", "lt": "2027-01-01T00:00:00Z"}.
    Accepts "2026", "2026-03", "2026-03-18" or a full ISO datetime for either end.
    Returns None if neither is set; raises ValueError (with a message meant for
    whoever sent the dates) on an unreadable or reversed range.
    """
    bounds = {}
    for value, is_end in ((start_date, False), (end_date, True)):
        if not value or not str(value).strip():
            continue
        try:
            bounds[is_end] = _parse_date_bound(str(value), is_end)
        except ValueError:
            raise ValueError(f"Could not read date '{value}'. Use e.g. '2026', '2026-03', "
                             f"'2026-03-18' or '2026-03-18T10:00:00Z'.")
    if not bounds:
        return None
    if False in bounds and True in bounds:
        (_, start), (end_key, end) = bounds[False], bounds[True]
        if start > end or (start == end and end_key == "lt"):
            raise ValueError(f"start_date '{start_date}' is not before end_date '{end_date}'.")
    return {key: f"{bound:%Y-%m-%dT%H:%M:%SZ}" for key, bound in bounds.values()}


def entry_body(text):
    """Strip the '[Cat] This entry is related to ... Form: default. text:' preamble
    that run_update.entry_to_text puts in front of the entry's own text."""
    for marker in ('text: ', 'Form: '):
        if marker in text:
            return text.split(marker, 1)[1]
    return text


def rerank_text(rec):
    """What the reranker reads for an entry: "[category] subject. body". `rec` is a
    Qdrant payload or a text-cache record - both have top_category/subject/text."""
    cat, subject = rec.get('top_category') or '', rec.get('subject') or ''
    head = (f"[{cat}] {subject}" if cat else subject).strip()
    return f"{head}. {entry_body(rec.get('text') or '')}"


class Reranker:
    """
    Cross-encoder (e.g. bge-reranker-base) that re-scores search candidates by
    reading the query and each entry's text together - slower than the vector
    search but much better at ordering, so it only re-sorts the top candidates.
    Used by /api/search and by DebugScripts/eval_retrieval.py, so production
    reranks exactly the way the evaluation measured.
    """

    def __init__(self, model_path):
        self.model = CrossEncoder(model_path, device='cpu', max_length=512)
        logging.info(f"Reranker loaded from {model_path}.")

    def score(self, query, texts):
        """Relevance score of each text for the query (higher = more relevant)."""
        return np.asarray(self.model.predict([(query, t) for t in texts],
                                             batch_size=16, show_progress_bar=False))

    def rerank(self, query, hits, top_k=None):
        """Reorder Qdrant hits (with payload, incl. 'text') best-first; keep top_k."""
        if not hits:
            return []
        missing = [h.id for h in hits if not (h.payload or {}).get('text')]
        if missing:
            # Still scored on category + subject; run DebugScripts/backfill_text_payload.py.
            logging.warning(f"Reranking without entry text for {len(missing)} hits: {missing[:10]}")
        scores = self.score(query, [rerank_text(h.payload or {}) for h in hits])
        ranked = [hits[k] for k in np.argsort(-scores, kind='stable')]
        return ranked if top_k is None else ranked[:top_k]


class SemanticSearch:
    def __init__(self, model_path, collection_name, qdrant_host, qdrant_port, query_prefix=""):

        # Initialization: Sets up the Qdrant client and the SentenceTransformer model.
        # Prints a message confirming the setup details.

        # query_prefix: prepended to search queries only (not documents). Needed
        # for asymmetric models like mxbai-embed-large-v1, which expects
        # "Represent this sentence for searching relevant passages: " on queries.
        self.client = QdrantClient(qdrant_host, port=qdrant_port)
        self.encoder = SentenceTransformer(model_path)
        self.collection_name = collection_name
        self.query_prefix = query_prefix
        logging.info(f"SemanticSearch initialized with model {model_path} on {qdrant_host}:{qdrant_port} for collection '{collection_name}'.")
        

    def set_filter_conditions(self, conditions):
       
        # Setting Filter Conditions: Converts a list of conditions into a format that Qdrant understands.
        # Supports 'match' and 'range' conditions, and prints errors for missing information.
       
        must_conditions = []
        for condition in conditions:
            if condition['type'] == 'match':      #The "match" portion is for the different categories and logs.
                must_conditions.append(models.FieldCondition(key=condition['field'], match=models.MatchValue(value=condition['value'])))
            elif condition['type'] == 'range':    #This is for the date and time condition
                if 'field' in condition and 'value' in condition:
                    # Any of gt/gte/lt/lte; a missing bound leaves that side open.
                    bounds = {k: v for k, v in condition['value'].items() if k in ('gt', 'gte', 'lt', 'lte') and v}
                    date_range = models.DatetimeRange(**bounds)
                    must_conditions.append(models.FieldCondition(key=condition['field'], range=date_range))
                else:
                    print(f"Error: Missing field or value for range condition: {condition}")
                    continue  
    
        print(f"Filter conditions set: {must_conditions}")  #Very good for checking that the conditions you expected are being set.
        return models.Filter(must=must_conditions)

    def adaptive_threshold_max(self, scores: list[float]) -> float:
        if not scores:
            return 0.0

        sorted_scores = sorted(scores, reverse=True)
        top_score = sorted_scores[0]

        # Strategy 1: 70% of top score (lenient)
        score_frac_cutoff = top_score * 0.70

        # Strategy 2: Mean of top 20 - 1.25 * std (lenient)
        top_n = sorted_scores[:min(20, len(sorted_scores))]
        stats_cutoff = np.mean(top_n) - 1.25 * np.std(top_n)

        # Strategy 3: First large drop in score
        drop_cutoff = top_score
        for i in range(len(sorted_scores) - 1):
            if sorted_scores[i] - sorted_scores[i + 1] > 0.02:
                drop_cutoff = sorted_scores[i + 1]
                break

        
        print(f"Score cutoff (70% top): {score_frac_cutoff:.4f}")
        print(f"Stats cutoff (mean - 1.25*std): {stats_cutoff:.4f}")
        print(f"Drop cutoff (first big drop): {drop_cutoff:.4f}")
        
        return min(stats_cutoff, (stats_cutoff + drop_cutoff) / 2)
    
    
    def adaptive_threshold_min(self, scores: list[float]) -> float:
        if not scores:
            return 0.0

        sorted_scores = sorted(scores, reverse=True)
        top_score = sorted_scores[0]

        # Strategy 1: 70% of top score (lenient)
        score_frac_cutoff = top_score * 0.70

        # Strategy 2: Mean of top 20 - 1.25 * std (lenient)
        top_n = sorted_scores[:min(20, len(sorted_scores))]
        stats_cutoff = np.mean(top_n) - 1.25 * np.std(top_n)

        # Strategy 3: First large drop in score
        drop_cutoff = top_score
        for i in range(len(sorted_scores) - 1):
            if sorted_scores[i] - sorted_scores[i + 1] > 0.02:
                drop_cutoff = sorted_scores[i + 1]
                break

        print(f"Score cutoff (70% top): {score_frac_cutoff:.4f}")
        print(f"Stats cutoff (mean - 1.25*std): {stats_cutoff:.4f}")
        print(f"Drop cutoff (first big drop): {drop_cutoff:.4f}")

        return min(score_frac_cutoff, stats_cutoff, drop_cutoff)

    def search_candidates(self, search_query, conditions, limit):
        """
        Raw nearest-neighbour hits WITH payload and without the adaptive cutoff -
        the candidate list for a Reranker. Same query encoding and search
        parameters as perform_search.
        """
        query_vector = self.encoder.encode(self.query_prefix + search_query).tolist()
        return self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            search_params=models.SearchParams(hnsw_ef=512, exact=True),
            query_filter=self.set_filter_conditions(conditions),
            limit=limit,
            with_payload=True,
        )

    def perform_search(self, search_query, conditions, limit, score_threshold):
            
            # Performing Search: Encodes the search query, applies filter conditions, and searches the collection.
            # The filter conditions are being passed by the GUI. I want to get more of this on the backend.
            # The limit refers to the limit on returned hits for the base search, not the reranked search.
            # Score threshold makes it 
            
            # Returns a list of hit IDs and prints them. Handles exceptions with error messages.
            
            try:
                print(f"Starting search with query: {search_query}")
                search_query = self.query_prefix + search_query
                print(f"Performing search with query: {search_query}, conditions: {conditions}, limit: {limit}, score_threshold: {score_threshold}")
                filter_conditions = self.set_filter_conditions(conditions)
                query_vector = self.encoder.encode(search_query).tolist()
                print(f"Encoded query vector: {query_vector[:5]}") 
                #logging.info(f"Encoded query vector: {query_vector}")
                hits = self.client.search(
                    collection_name=self.collection_name, 
                    query_vector=query_vector, 
                    search_params=models.SearchParams(hnsw_ef=512, exact=True),
                    query_filter=filter_conditions, 
                    limit=limit,
                    score_threshold=score_threshold
                )
                
                
                
                
                scores = [hit.score for hit in hits]
                for hit in hits:
                    print(f"ID {hit.id} Score: {hit.score}")

                if scores:
                    threshold = self.adaptive_threshold_min(scores)
                    print(f"Top score: {scores[0]:.4f}, Adaptive threshold: {threshold:.4f}")
                    filtered_hits = [(str(hit.id), hit.score) for hit in hits if hit.score >= threshold]
                else:
                    filtered_hits = []

                # Print IDs and scores
                for fid, fscore in filtered_hits:
                    print(f"ID {fid} Score: {fscore}")

                # Return only the IDs
                return [fid for fid, _ in filtered_hits]
                
                # for hit in hits:
                #     print(f"ID {hit.id} Score: {hit.score}")
                # hit_ids = [str(hit.id) for hit in hits]
                # #print(f"Search hits IDs: {hit_ids}") 
                # #logging.info(f"Search response: {hits}")

                
                
                # # Step 2: Adaptive threshold filtering
                # if hits:
                #     top_score = hits[0].score
                #     adaptive_threshold = top_score * 0.75
                #     print(f"Top score: {top_score:.4f}, Adaptive threshold: {adaptive_threshold:.4f}")

                #     filtered_hits = [hit for hit in hits if hit.score >= adaptive_threshold]
                # else:
                #     filtered_hits = []

                # for hit in filtered_hits:
                #     print(f"ID {hit.id} Score: {hit.score}")

                # hit_ids = [str(hit.id) for hit in filtered_hits]
                
                # return hit_ids
                
                
                
            except Exception as e:
                print(f"Error performing search: {e}")
                logging.error(f"Error performing search: {e}")
                return []     

    def construct_url_and_open(self, hit_ids, base_url):
        
        # Constructing URL and Opening: Creates a URL from the hit IDs and opens it in a web browser.
        # Prints the constructed URL and handles cases where no hits are found.
        if not hit_ids:
            print("No hits found.")
            return None
        else:
            return f"{base_url}/show?e={hit_ids[0]}"


    def recommend_results(self, pos_id, neg_id, conditions, limit, score_threshold):
        
        # Recommending Results: Uses positive and negative example IDs to recommend similar items.
        # Applies filter conditions and returns a list of recommended hit IDs. Handles exceptions with error messages.
        # I want to add in a system for saving queries and results for Reinforcement learning. This is probably the most useful part of the program.
        
        try:
            print(f"Starting search with positive IDs {pos_id} and negative IDs {neg_id}")
            filter_conditions = self.set_filter_conditions(conditions)
            print(f"Encoded vector for {pos_id}")
            hits = self.client.recommend(
                    collection_name=self.collection_name,
                    positive=pos_id,
                    negative=neg_id,
                    # positive=[],  #Hardcoded in ID's for testing
                    # negative=[], 
                    strategy=models.RecommendStrategy.BEST_SCORE,
                    query_filter=filter_conditions, 
                    limit=limit,
                    score_threshold=score_threshold
                    )

            scores = [hit.score for hit in hits]
            for hit in hits:
                print(f"ID {hit.id} Score: {hit.score}")

            if scores:
                threshold = self.adaptive_threshold_max(scores)
                print(f"Top score: {scores[0]:.4f}, Adaptive threshold: {threshold:.4f}")
                filtered_hits = [(str(hit.id), hit.score) for hit in hits if hit.score >= threshold]
            else:
                filtered_hits = []

            # Print IDs and scores
            for fid, fscore in filtered_hits:
                print(f"ID {fid} Score: {fscore}")


            # Return only the IDs, with pos_id first
            final_ids = list(map(str, pos_id)) + [fid for fid, _ in filtered_hits]
            return final_ids
            
        except Exception as e:
            print(f"Error performing search: {e}")
            return []

    

    def scroll_results(self, conditions, limit):
       
        # Scrolling Results: Retrieves a list of items from the collection based on filter conditions.
        # Returns a list of hit IDs and prints them. Handles exceptions with error messages. 
       
       
        try:
            print(f"Starting scrolling results")
            filter_conditions = self.set_filter_conditions(conditions)
            hits = self.client.scroll(
                    collection_name=self.collection_name,
                    scroll_filter=filter_conditions, 
                    limit=limit,
                    
                    )
    
    
            ##Uncomment if you need to see the score
            # for hit in hits:
            #     print(f"Payload for ID {hit.id}: {hit.payload}, Score: {hit.score}")      
            hit_ids = [str(hit.id) for hit in hits]
            print(f"Search hits IDs: {hit_ids}")  
            
            return hit_ids
        except Exception as e:
            print(f"Error performing search: {e}")
            return []
