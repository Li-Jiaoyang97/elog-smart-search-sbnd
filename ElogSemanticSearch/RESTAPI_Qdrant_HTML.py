#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 21 19:15:03 2024

@author: haydenh
"""


'''



'''

base_url = "https://www-bd.fnal.gov/Elog/"
import os
import subprocess
import logging
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import List, Dict, Optional
import ast
from datetime import datetime, timedelta
from ElogSemanticSearch.QdrantSearch.QdrantSearch import SemanticSearch
# from ElogSemanticSearch.reranker import CustomReRanker
from fastapi.staticfiles import StaticFiles
from fastapi import Query
from fastapi import Depends, Request
from ElogSemanticSearch.session_fetcher import SessionFetcher 
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Allow frontend requests from localhost:3000
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)





session_fetcher = SessionFetcher()


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Mount static files directory
app.mount("/static", StaticFiles(directory="/app/ElogSemanticSearch/static"), name="static")

# searcher_configs = {
#     "Mixed-Bread-1024": {
#         "model_path": "/app/ElogSemanticSearch/models/mxbai-embed-large-v1",
#         "collection_name": "Elog_collection_1024_METADATA",
#         "host": "qdrant",  # Docker service name for Qdrant
#         "port": 6333
#     }
# }

searcher_configs = {
    "All-MPNet-v2-Update": {
        "model_path": "/app/ElogSemanticSearch/models/all-mpnet-base-v2",
        "collection_name": "Elog_collection_768_METADATA",
        "host": "qdrant",  # Docker service name for Qdrant
        "port": 6333
    }
}

searcher = None

active_sessions = {}


class LoginRequest(BaseModel):
    username: str
    password: str

# Pydantic models for requests
class SearchRequest(BaseModel):
    query: str
    limit: int
    score_threshold: float
    conditions_list: List[Dict] = []

class ReRankRequest(BaseModel):
    query: str
    limit: int
    score_threshold: float
    rank_limit: int
    conditions_list: List[Dict] = []

class RecommendRequest(BaseModel):
    pos_input: str
    neg_input: str
    limit: int
    score_threshold: float
    conditions_list: List[Dict] = []

class UpdateRequest(BaseModel):
    update_range: Optional[int] = None
    start_id: Optional[int] = None 
    end_id: Optional[int] = None
    username: str
    password: str

def get_current_user(request: Request):
    """
    Checks if the user is authenticated by validating their session.
    If the session is invalid or missing, raises an HTTPException.
    """
    username = request.headers.get("X-Username")  
    
    if not username or username not in active_sessions:
        raise HTTPException(status_code=403, detail="Unauthorized: Session expired or missing")

    session_data = active_sessions[username]
    
    if datetime.utcnow() > session_data["expires_at"]:
        del active_sessions[username]  #Delete expired session after an hour
        raise HTTPException(status_code=403, detail="Session expired. Log in again.")
    
    return username  

def initialize_default_searcher():
    model_key = "All-MPNet-v2-Update"
    config = searcher_configs[model_key]
    global searcher
    searcher = SemanticSearch(
        model_path=config["model_path"],
        collection_name=config["collection_name"],
        qdrant_host=config["host"],
        qdrant_port=config["port"]
    )
    logging.debug(
        f"Searcher initialized with model: {config['model_path']} on host: {config['host']}:{config['port']} for collection: {config['collection_name']}"
    )
    return f"Searcher initialized with default model: {config['model_path']} and collection: {config['collection_name']}"

default_searcher_message = initialize_default_searcher()
print(default_searcher_message)

def construct_conditions_from_list(conditions_list):
    conditions = []
    for cond in conditions_list:
        if cond["type"] in ["log", "category"]:
            field_name = "log_names" if cond["type"] == "log" else "category_names"
            conditions.append({
                "field": field_name,
                "type": "match",
                "value": cond["value"]
            })
        elif cond["type"] == "range":
            field_name = cond.get("field", "creation_date")
            conditions.append({
                "field": field_name,
                "type": "range",
                "value": cond["value"]
            })
    print("Final search conditions:", conditions)
    return conditions

def call_search_function_rerank(query, limit, score_threshold, rank_limit, conditions):
    print("Starting rerank with query:", query)
    full_url = None

    hit_ids = searcher.perform_search(query, conditions, int(limit), float(score_threshold))

    if hit_ids:
        reranker = CustomReRanker('/app/ElogSemanticSearch/models/ms-marco-MiniLM-L-12-v2')
        reranked_documents = reranker.search_and_rerank(query, hit_ids, ranklimit=rank_limit)

        if reranked_documents:
            for doc in reranked_documents:
                print(f"ID: {doc['id']}, Score: {doc['score']:.4f}")

            reranked_hit_ids = [doc['id'] for doc in reranked_documents]
            # Sort the IDs in ascending order
            sorted_hit_ids = sorted(reranked_hit_ids, key=int)

            # Print the sorted list of IDs
            print("Sorted Hit IDs:", sorted_hit_ids)

            # Pass base_url to construct_url_and_open
            full_url = searcher.construct_url_and_open(reranked_hit_ids, base_url)
            print(f"Attempting to open URL: {full_url}")
            if full_url and isinstance(full_url, str):
                print("URL properly formed")
            else:
                print("Failed to open URL because it was not properly formed.")
        else:
            print("Reranking returned no results.")
    else:
        print("No data available for reranking after initialization.")

    return full_url


@app.post("/login")
async def login(request: LoginRequest):
    """
    Endpoint to authenticate a user and create an Elog session.
    """
    try:
        auth_success = session_fetcher.create_authenticated_session(request.username, request.password)
         
        if not auth_success:
            raise HTTPException(status_code=401, detail="Authentication failed.")
        
        
        active_sessions[request.username] = {
            "authenticated": True,
            "expires_at": datetime.utcnow() + timedelta(hours=1)
        }

        
        print(f"Active sessions after login: {active_sessions}")
        
        
        
        return {"message": "Login successful"}
    
    except Exception as e:
        print("login failed:", str(e))
        raise HTTPException(status_code=401, detail=str(e))


@app.post("/search")
def search(request: SearchRequest, username: str = Depends(get_current_user)):
    
    try:   
    
        print("Starting search with query:", request.query)
        
        #construct conditions 
        conditions = construct_conditions_from_list(request.conditions_list)
        
        #Uses searcher we intialized above to preform search
        hit_ids = searcher.perform_search(
            request.query, conditions, request.limit, request.score_threshold
        )
        
        if not hit_ids:
                logging.warning("No results found for the given search criteria.")
                raise HTTPException(status_code=404, detail="No results found. Try adding more detail or lower the Relvance threshold in Advanced Settings")
        
        # if hit_ids:
            #print(f"Search hits IDs: {hit_ids}")
        full_url = searcher.construct_url_and_open(hit_ids, base_url)
        print(f"Attempting to open URL")
            
        if not full_url or not isinstance(full_url, str):
            logging.error("Failed to open URL because it was not properly formed.")
            raise HTTPException(status_code=500, detail="Failed to construct URL.")
            
            
        logging.debug("URL formed properly")
        print("URL formed properly")
        # response_data = {"url": full_url, "hit_ids": hit_ids}
        #print(f"Response data: {response_data}")
        return {"success": True, "url":full_url, "hit_ids": hit_ids}

    except HTTPException as http_error:
         # ✅ Return API-level errors (e.g., 404, 500)
        raise http_error
   
    except Exception as e:
        # ✅ Catch unexpected errors and return a 500 error
        logging.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred.")
   
   

    # else:
    #     print(f"Failed to open URL because it was not properly formed: {full_url}")
    #     return {"message": "Failed to construct URL"}
    # else:
    #     print("No results found for the given search criteria.")
    #     return {"message": "No results found"}

@app.post("/rerank")
def rerank(request: ReRankRequest):
    raise HTTPException(status_code=403, detail="Reranking is disabled")
    conditions = construct_conditions_from_list(request.conditions_list)
    reranked_url = call_search_function_rerank(
        request.query, request.limit, request.score_threshold, request.rank_limit, conditions
    )
    if reranked_url:
        return {"url": reranked_url}
    else:
        return {"message": "No results found"}

@app.post("/recommend")
def recommend(request: RecommendRequest, username: str = Depends(get_current_user)):

    pos_ids = ast.literal_eval(request.pos_input)
    if isinstance(pos_ids, int):
        pos_ids = [pos_ids]
    neg_ids = ast.literal_eval(request.neg_input) if request.neg_input else []
    if isinstance(neg_ids, int):
        neg_ids = [neg_ids]

    conditions = construct_conditions_from_list(request.conditions_list)
    hit_ids = searcher.recommend_results(
        pos_ids, neg_ids, conditions, request.limit, request.score_threshold
    )
    if hit_ids:
        recommend_url = searcher.construct_url_and_open(hit_ids, base_url)
        return {"url": recommend_url, "hit_ids": hit_ids}
    else:
        print("No results found for the given search criteria.")
        return {"message": "No results found"}

# Endpoint to trigger the update process

# async def update(background_tasks: BackgroundTasks, update_range: int = Query(100, description="Number of entries to update.")):

@app.post("/update")
async def update(
    background_tasks: BackgroundTasks,
    update_request: UpdateRequest,

):

    if (update_request.update_range is None and (update_request.start_id is None or update_request.end_id is None)):
        raise HTTPException(status_code=400, detail="Either update_range or start_id & end_id must be provided.")


    background_tasks.add_task(
        run_update_script, 
        update_request.update_range,
        update_request.start_id,
        update_request.end_id,
        update_request.username,
        update_request.password
    )
    return {"message": f"Update process has been started."}

def run_update_script(update_range=None, start_id=None, end_id=None, username="", password=""):
    try:
        
        
        command = [
            "python", "/app/ElogSemanticSearch/run_update.py"
        ]
        
        if update_range is not None:
            command.extend(["--range", str(update_range)])
        
        if start_id is not None and end_id is not None:
            command.extend(["--start_id", str(start_id), "--end_id", str(end_id)])
        
        command.extend(["--username", username, "--password", password])
        
        #start the update subprocess using the dynamically altered command line
        
        process =  subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        
        

        # Read the output each line and log it
        with process.stdout:
            for line in iter(process.stdout.readline, ''):
                logger.info(line.strip())

        # Wait for the process to finish and get the return code
        returncode = process.wait()

        if returncode == 0:
            logger.info("Update script completed successfully.")
        else:
            logger.error(f"Update script exited with return code {returncode}.")

    except Exception as e:
        logger.error(f"Exception while running update script: {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
