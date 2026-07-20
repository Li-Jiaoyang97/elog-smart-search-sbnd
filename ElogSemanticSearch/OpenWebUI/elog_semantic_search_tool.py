"""
title: SBND Elog Semantic Search
author: Jiaoyang Li
version: 0.1.0
license: MIT
description: Semantic search (and "find similar") over the SBND electronic logbook (ECL), backed by the existing Qdrant vector index built in this repo's ingestion pipeline.
requirements: qdrant-client,sentence-transformers,numpy

Notes:
- This file is self-contained on purpose. OpenWebUI executes Tools inside its own
  Python process/venv, so it cannot import ElogSemanticSearch/QdrantSearch/QdrantSearchSBND.py
  directly. The search + adaptive-threshold logic below is a minimal copy of
  SemanticSearch.perform_search / .recommend_results / .adaptive_threshold_* from that file.
  If you change the scoring logic there, mirror the change here too.
- No Fermilab ECL credentials are used or needed. Search/recommend only read the
  already-built Qdrant index and a local embedding model; they never touch the elog's
  own login system. Only the offline ingestion job (run_initSBND.py / run_updateSBND.py)
  needs ECL credentials, and that runs outside of OpenWebUI entirely.
- Results include the ID, subject, creation date, and elog link per hit - all of which are
  already stored in Qdrant's payload from ingestion, so this costs no extra fetch/latency/
  credentials. Full entry body text is NOT included (and can't be cheaply): it isn't stored
  in the payload, only used transiently to build the embedding at ingestion time. Pulling
  full text would require a live, authenticated re-fetch from the ECL system - a possible
  future enhancement, not done here. Without at least the subject line, the calling LLM has
  no way to verify *why* a hit matched and tends to guess/hallucinate about entry content -
  see the "none of these mention Grafana v13.1" incident this file's history was fixed after.
- Optional category / date-range filtering mirrors RESTAPI_Qdrant_HTML.py's
  construct_conditions_from_list() + QdrantSearchSBND.SemanticSearch.set_filter_conditions(),
  simplified to flat keyword args (category, start_date, end_date) since LLM function-calling
  handles flat args far more reliably than a nested list-of-dicts schema.
"""

from typing import List

import numpy as np
from pydantic import BaseModel, Field
from qdrant_client import models, QdrantClient
from sentence_transformers import SentenceTransformer


class Tools:
    class Valves(BaseModel):
        QDRANT_HOST: str = Field(
            default="localhost", description="Hostname of the Qdrant instance."
        )
        QDRANT_PORT: int = Field(
            default=6333, description="Port of the Qdrant instance."
        )
        COLLECTION_NAME: str = Field(
            default="Elog_collection_768_METADATA",
            description="Qdrant collection name to search.",
        )
        MODEL_PATH: str = Field(
            default="all-mpnet-base-v2",
            description="SentenceTransformer model name or local path. Must match the model used during ingestion.",
        )
        BASE_URL: str = Field(
            default="https://dbweb1.fnal.gov:8443/ECL/sbnd/E",
            description="Base URL used to construct links to individual elog entries.",
        )
        SEARCH_LIMIT: int = Field(
            default=10,
            description="Max raw candidates to fetch before adaptive filtering trims them down.",
        )

    def __init__(self):
        self.valves = self.Valves()
        self._client = None
        self._encoder = None

    # ── lazy singletons ──────────────────────────────────────────────────────

    def _get_client(self) -> QdrantClient:
        if self._client is None:
            self._client = QdrantClient(
                self.valves.QDRANT_HOST, port=self.valves.QDRANT_PORT
            )
        return self._client

    def _get_encoder(self) -> SentenceTransformer:
        if self._encoder is None:
            self._encoder = SentenceTransformer(self.valves.MODEL_PATH)
        return self._encoder

    # ── adaptive thresholds (mirrors QdrantSearchSBND.SemanticSearch) ───────

    @staticmethod
    def _adaptive_threshold_min(scores: List[float]) -> float:
        if not scores:
            return 0.0
        sorted_scores = sorted(scores, reverse=True)
        top_score = sorted_scores[0]
        score_frac_cutoff = top_score * 0.70
        top_n = sorted_scores[: min(20, len(sorted_scores))]
        stats_cutoff = float(np.mean(top_n) - 1.25 * np.std(top_n))
        drop_cutoff = top_score
        for i in range(len(sorted_scores) - 1):
            if sorted_scores[i] - sorted_scores[i + 1] > 0.02:
                drop_cutoff = sorted_scores[i + 1]
                break
        return min(score_frac_cutoff, stats_cutoff, drop_cutoff)

    @staticmethod
    def _adaptive_threshold_max(scores: List[float]) -> float:
        if not scores:
            return 0.0
        sorted_scores = sorted(scores, reverse=True)
        top_score = sorted_scores[0]
        top_n = sorted_scores[: min(20, len(sorted_scores))]
        stats_cutoff = float(np.mean(top_n) - 1.25 * np.std(top_n))
        drop_cutoff = top_score
        for i in range(len(sorted_scores) - 1):
            if sorted_scores[i] - sorted_scores[i + 1] > 0.02:
                drop_cutoff = sorted_scores[i + 1]
                break
        return min(stats_cutoff, (stats_cutoff + drop_cutoff) / 2)

    # ── filtering (mirrors RESTAPI_Qdrant_HTML.construct_conditions_from_list) ─

    @staticmethod
    def _build_filter(category: str = "", start_date: str = "", end_date: str = ""):
        must = []
        if category:
            must.append(
                models.FieldCondition(
                    key="top_category", match=models.MatchValue(value=category)
                )
            )
        if start_date or end_date:
            must.append(
                models.FieldCondition(
                    key="creation_date",
                    range=models.DatetimeRange(
                        gte=start_date or None, lte=end_date or None
                    ),
                )
            )
        return models.Filter(must=must) if must else None

    # ── formatting (mirrors run_search.py's print_results, plus subject/date) ─

    def _format_hits(self, hits) -> str:
        if not hits:
            return "No results found."

        lines = [f"Found {len(hits)} result(s):"]
        for i, hit in enumerate(hits, 1):
            payload = hit.payload or {}
            subject = payload.get("subject") or "(no subject)"
            date = payload.get("creation_date") or ""
            date_str = f" ({date})" if date else ""
            url = f"{self.valves.BASE_URL}/show?e={hit.id}"
            lines.append(f"{i}. ID {hit.id} — {subject}{date_str} → {url}")
        return "\n".join(lines)

    # ── tools exposed to the LLM ─────────────────────────────────────────────

    def search_elog(
        self,
        query: str,
        category: str = "",
        start_date: str = "",
        end_date: str = "",
    ) -> str:
        """
        Search the SBND electronic logbook (ECL) for entries semantically related to a
        natural-language query. Use this whenever the user asks about SBND detector
        status, shift logs, run conditions, hardware issues, or anything else that
        might be recorded in the logbook.
        :param query: Natural-language description of what to search for.
        :param category: Optional top-level elog category to restrict results to (leave empty to search all categories).
        :param start_date: Optional RFC3339 date/time (e.g. "2026-01-01T00:00:00Z") - only return entries created on or after this date.
        :param end_date: Optional RFC3339 date/time - only return entries created on or before this date.
        :return: A numbered list of matching elog entries: ID, subject, date, and link.
        """
        try:
            client = self._get_client()
            encoder = self._get_encoder()
            vector = encoder.encode(query).tolist()
            query_filter = self._build_filter(category, start_date, end_date)

            hits = client.search(
                collection_name=self.valves.COLLECTION_NAME,
                query_vector=vector,
                search_params=models.SearchParams(hnsw_ef=512, exact=True),
                query_filter=query_filter,
                limit=self.valves.SEARCH_LIMIT,
                with_payload=True,
            )
            if not hits:
                return "No matching elog entries were found."

            scores = [hit.score for hit in hits]
            threshold = self._adaptive_threshold_min(scores)
            filtered = [hit for hit in hits if hit.score >= threshold]
            return self._format_hits(filtered)
        except Exception as e:
            return f"Elog search failed: {e}"

    def find_similar_elog_entries(
        self,
        entry_id: str,
        category: str = "",
        start_date: str = "",
        end_date: str = "",
    ) -> str:
        """
        Find elog entries similar to a given, already-known elog entry ID. Use this
        when the user references a specific elog entry (by its ID) and wants more
        entries like it.
        :param entry_id: The elog entry ID to use as the positive example.
        :param category: Optional top-level elog category to restrict results to (leave empty to search all categories).
        :param start_date: Optional RFC3339 date/time (e.g. "2026-01-01T00:00:00Z") - only return entries created on or after this date.
        :param end_date: Optional RFC3339 date/time - only return entries created on or before this date.
        :return: A numbered list of similar elog entries: ID, subject, date, and link.
        """
        try:
            client = self._get_client()
            query_filter = self._build_filter(category, start_date, end_date)
            hits = client.recommend(
                collection_name=self.valves.COLLECTION_NAME,
                positive=[entry_id],
                negative=[],
                strategy=models.RecommendStrategy.BEST_SCORE,
                query_filter=query_filter,
                limit=self.valves.SEARCH_LIMIT,
                with_payload=True,
            )
            if not hits:
                return "No similar elog entries were found."

            scores = [hit.score for hit in hits]
            threshold = self._adaptive_threshold_max(scores)
            filtered = [hit for hit in hits if hit.score >= threshold]
            return self._format_hits(filtered)
        except Exception as e:
            return f"Elog recommend failed: {e}"
