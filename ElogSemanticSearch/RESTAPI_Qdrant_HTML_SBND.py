#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SBND port of RESTAPI_Qdrant_HTML.py.

Same login-gated design as the original (session via X-Username header, checked
against active_sessions set up by /login) - kept for parity rather than dropping
auth, per project decision. Wired to the SBND-specific components instead of the
generic ones: QdrantSearchSBND.SemanticSearch, session_fetcherSBND.SessionFetcherSBND,
and the SBND ECL base URL.

Note: /update calls run_updateSBND.py, whose CLI takes --after/--limit (a time
window + max entries per category), not the generic script's --range/--start_id/
--end_id - the UpdateRequest model below matches run_updateSBND.py's actual args.
"""

base_url = "https://dbweb1.fnal.gov:8443/ECL/sbnd/E"
import os
from dotenv import load_dotenv
import subprocess
import logging
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import List, Dict, Optional
from datetime import datetime, timedelta
from ElogSemanticSearch.QdrantSearch.QdrantSearchSBND import SemanticSearch
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi import Depends, Request
from ElogSemanticSearch.session_fetcherSBND import SessionFetcherSBND
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="SBND Elog Semantic Search",
    description=(
        "Semantic search over the SBND ECL (electronic logbook). "
        "Use /api/search to find logbook entries relevant to a natural-language query."
    ),
    version="1.0",
)

session_fetcher = SessionFetcherSBND()
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Mount static files directory (production/Docker path - skipped locally if absent
# so this app can still start for local dev/testing)
_PROD_STATIC_DIR = "/app/ElogSemanticSearch/static"
if os.path.isdir(_PROD_STATIC_DIR):
    app.mount("/static", StaticFiles(directory=_PROD_STATIC_DIR), name="static")

# Plain-HTML UI for local/manual testing (see static_sbnd/index.html) - served
# from the same FastAPI app so browser fetch() calls are same-origin (no CORS
# setup needed for this).
_SBND_UI_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static_sbnd")
app.mount("/static_sbnd", StaticFiles(directory=_SBND_UI_DIR), name="static_sbnd")


@app.get("/", include_in_schema=False)
def serve_sbnd_ui():
    return FileResponse(os.path.join(_SBND_UI_DIR, "index.html"))


def required_env(var_name: str) -> str:
    if var_name in os.environ:
        return os.environ[var_name]
    raise RuntimeError(f"Environment variable {var_name} must be set.")

# Two embedding models, loaded side by side for comparison. "mxbai" is the
# default (matches the old single-searcher behavior via MODEL_PATH /
# COLLECTION_NAME); "mpnet" is optional - it's only loaded if MPNET_MODEL_PATH
# is set, so the API still starts in single-model deployments.
searcher_configs = {
    "mxbai": {
        "model_path": required_env("MODEL_PATH"),
        "collection_name": required_env("COLLECTION_NAME"),
        "host": os.environ["QDRANT_HOST"],
        "port": 6333,
        # mxbai-embed-large-v1 is asymmetric: queries (not documents) need this
        # prefix for the model to score retrieval properly.
        "query_prefix": "Represent this sentence for searching relevant passages: ",
    },
}
if os.environ.get("MPNET_MODEL_PATH"):
    searcher_configs["mpnet"] = {
        "model_path": os.environ["MPNET_MODEL_PATH"],
        "collection_name": required_env("MPNET_COLLECTION_NAME"),
        "host": os.environ["QDRANT_HOST"],
        "port": 6333,
        "query_prefix": "",  # all-mpnet-base-v2 is symmetric, no prefix
    }

DEFAULT_MODEL = "mxbai"

searchers = {}
searcher = None  # default searcher, used by the legacy HTML-UI endpoints

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

class RecommendRequest(BaseModel):
    pos_input: str
    neg_input: str
    limit: int
    score_threshold: float
    conditions_list: List[Dict] = []

class UpdateRequest(BaseModel):
    after: Optional[str] = "7days"
    limit: Optional[int] = 50
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

def initialize_searchers():
    global searcher
    for key, config in searcher_configs.items():
        searchers[key] = SemanticSearch(
            model_path=config["model_path"],
            collection_name=config["collection_name"],
            qdrant_host=config["host"],
            qdrant_port=config["port"],
            query_prefix=config["query_prefix"],
        )
        logger.info(
            f"Searcher '{key}' initialized: model={config['model_path']}, collection={config['collection_name']}"
        )
    searcher = searchers[DEFAULT_MODEL]
    return f"Searchers initialized: {list(searchers)} (default: {DEFAULT_MODEL})"

print(initialize_searchers())

def construct_conditions_from_list(conditions_list):
    # SBND's Qdrant payload only has subject/top_category/subcategory/creation_date
    # (see run_initSBND.py's entry_to_point) - no log_names/category_names field
    # like the generic system's payload, so "category" maps to top_category here.
    conditions = []
    for cond in conditions_list:
        if cond["type"] == "category":
            conditions.append({
                "field": "top_category",
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


@app.post("/login", include_in_schema=False)
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

    except HTTPException:
        raise
    except Exception as e:
        print("login failed:", str(e))
        raise HTTPException(status_code=401, detail=str(e))


@app.post("/search", include_in_schema=False)
def search(request: SearchRequest, username: str = Depends(get_current_user)):

    try:
        print("Starting search with query:", request.query)

        conditions = construct_conditions_from_list(request.conditions_list)

        hit_ids = searcher.perform_search(
            request.query, conditions, request.limit, request.score_threshold
        )

        if not hit_ids:
            logging.warning("No results found for the given search criteria.")
            raise HTTPException(status_code=404, detail="No results found. Try adding more detail or lower the Relvance threshold in Advanced Settings")

        full_url = searcher.construct_url_and_open(hit_ids, base_url)
        print(f"Attempting to open URL")

        if not full_url or not isinstance(full_url, str):
            logging.error("Failed to open URL because it was not properly formed.")
            raise HTTPException(status_code=500, detail="Failed to construct URL.")

        logging.debug("URL formed properly")
        return {"success": True, "url": full_url, "hit_ids": hit_ids}

    except HTTPException as http_error:
        raise http_error

    except Exception as e:
        logging.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred.")


@app.post("/recommend", include_in_schema=False)
def recommend(request: RecommendRequest, username: str = Depends(get_current_user)):
    import ast

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


class ToolSearchRequest(BaseModel):
    query: str
    model: str = DEFAULT_MODEL  # "mxbai" or "mpnet" - which embedding model to search with
    limit: int = 10


@app.post(
    "/api/search",
    summary="Search the SBND logbook",
    description=(
        "Semantic search over SBND ECL logbook entries. Give a natural-language "
        "description of what you're looking for (e.g. 'TPC power supply trips', "
        "'purity monitor calibration issues'). Returns the most relevant entries "
        "with their subject, category, date, and a link to read the full entry. "
        "The optional 'model' field selects the embedding model: 'mxbai' "
        "(default) or 'mpnet' - useful for comparing the two."
    ),
)
def tool_search(request: ToolSearchRequest):
    if request.model not in searchers:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model '{request.model}'. Available: {list(searchers)}",
        )
    s = searchers[request.model]

    # score_threshold=0.0: rely on the adaptive threshold inside perform_search
    # rather than a fixed cutoff (the two models' score scales differ).
    hit_ids = s.perform_search(request.query, [], request.limit, 0.0)
    if not hit_ids:
        return {"model": request.model, "results": [], "message": "No relevant entries found."}

    points = s.client.retrieve(
        collection_name=s.collection_name,
        ids=[int(i) for i in hit_ids],
        with_payload=True,
    )
    payload_by_id = {str(p.id): p.payload for p in points}

    results = []
    for eid in hit_ids:  # retrieve() doesn't preserve order; keep ranking from search
        payload = payload_by_id.get(eid, {})
        results.append({
            "entry_id": eid,
            "subject": payload.get("subject", ""),
            "category": payload.get("top_category", ""),
            "subcategory": payload.get("subcategory", ""),
            "date": payload.get("creation_date", ""),
            "url": f"{base_url}/show?e={eid}",
        })
    return {"model": request.model, "results": results}


@app.post("/update", include_in_schema=False)
async def update(
    background_tasks: BackgroundTasks,
    update_request: UpdateRequest,
):
    background_tasks.add_task(
        run_update_script,
        update_request.after,
        update_request.limit,
        update_request.username,
        update_request.password
    )
    return {"message": "Update process has been started."}

def run_update_script(after="7days", limit=50, username="", password=""):
    try:
        command = [
            "python", "/app/ElogSemanticSearch/run_updateSBND.py",
            "--after", str(after),
            "--limit", str(limit),
            "--username", username,
            "--password", password,
        ]

        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )

        with process.stdout:
            for line in iter(process.stdout.readline, ''):
                logger.info(line.strip())

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
