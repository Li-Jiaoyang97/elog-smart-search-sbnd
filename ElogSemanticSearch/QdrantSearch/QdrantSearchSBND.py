#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 04:13:08 2024

@author: hayden
@modified by jiaoyang for SBND Smart Search
"""


from qdrant_client import models, QdrantClient
from sentence_transformers import SentenceTransformer
import webbrowser
import logging
import numpy as np

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class SemanticSearch:
    def __init__(self, model_path, collection_name, qdrant_host, qdrant_port):
        
        # Initialization: Sets up the Qdrant client and the SentenceTransformer model.
        # Prints a message confirming the setup details.
        
        self.client = QdrantClient(qdrant_host, port=qdrant_port)
        self.encoder = SentenceTransformer(model_path)
        self.collection_name = collection_name
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
                    date_range = models.DatetimeRange(gte=condition['value']['gte'], lte=condition['value']['lte'])
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

    def perform_search(self, search_query, conditions, limit, score_threshold):
            
            # Performing Search: Encodes the search query, applies filter conditions, and searches the collection.
            # The filter conditions are being passed by the GUI. I want to get more of this on the backend.
            # The limit refers to the limit on returned hits for the base search, not the reranked search.
            # Score threshold makes it 
            
            # Returns a list of hit IDs and prints them. Handles exceptions with error messages.
            
            try:
                print(f"Starting search with query: {search_query}")  
                # retrieval_prompt = "Represent this sentence for searching relevant passages: "
                # search_query = retrieval_prompt + search_query
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
