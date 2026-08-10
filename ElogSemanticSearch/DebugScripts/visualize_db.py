#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
visualize_db.py
---------------
Pull all vectors from Qdrant, reduce to 2D with UMAP, and produce an
interactive Plotly scatter plot saved as an HTML file.

Points are colored by top_category.  Hover over any point to see its
entry ID, subject, date, and a direct ECL link.

Dependencies (install once):
  pip install umap-learn plotly

Usage:
  python visualize_db.py                     # saves elog_map.html
  python visualize_db.py --out my_map.html   # custom output file
  python visualize_db.py --category Shift    # show only one top-category
  python visualize_db.py --neighbors 20 --min_dist 0.1   # tune UMAP
"""

import os
import argparse
import logging
import numpy as np
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, FieldCondition, MatchValue

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

collection_name = os.environ.get("COLLECTION_NAME")
if not collection_name:
    raise RuntimeError("Environment variable COLLECTION_NAME must be set.")

ECL_BASE_URL = os.environ.get("ECL_URL", "https://dbweb1.fnal.gov:8443/ECL/sbnd/E/show?e=")


def _get_qdrant_client():
    return QdrantClient(host=os.environ.get("QDRANT_HOST"), port=6333)


def _entry_url(entry_id):
    return f"{ECL_BASE_URL}{entry_id}"


def fetch_all_vectors(qdrant, category_filter=None):
    """Scroll through the entire collection and return vectors + metadata."""
    logger.info("Fetching vectors from Qdrant...")

    scroll_filter = None
    if category_filter:
        scroll_filter = Filter(must=[
            FieldCondition(key="top_category", match=MatchValue(value=category_filter))
        ])

    ids, vectors, subjects, top_cats, subcats, dates = [], [], [], [], [], []
    offset = None

    while True:
        batch, offset = qdrant.scroll(
            collection_name=collection_name,
            scroll_filter=scroll_filter,
            limit=500,
            offset=offset,
            with_payload=True,
            with_vectors=True,
        )
        for point in batch:
            if point.vector is None:
                continue
            ids.append(point.id)
            vectors.append(point.vector)
            p = point.payload
            subjects.append(p.get('subject', '') or '(no subject)')
            top_cats.append(p.get('top_category', '') or '(none)')
            subcats.append(p.get('subcategory', '') or '')
            dates.append(p.get('creation_date', '') or '')
        if offset is None:
            break

    logger.info(f"Fetched {len(ids)} entries.")
    return (
        np.array(ids),
        np.array(vectors, dtype=np.float32),
        subjects,
        top_cats,
        subcats,
        dates,
    )


def run_umap(vectors, n_neighbors=15, min_dist=0.05):
    """Reduce 768-D vectors to 2D using UMAP."""
    try:
        import umap
    except ImportError:
        raise ImportError("umap-learn is required: pip install umap-learn")

    logger.info(f"Running UMAP on {len(vectors)} points (n_neighbors={n_neighbors}, min_dist={min_dist})...")
    reducer = umap.UMAP(n_components=2, n_neighbors=n_neighbors, min_dist=min_dist, random_state=42)
    embedding = reducer.fit_transform(vectors)
    logger.info("UMAP done.")
    return embedding


def build_plot(ids, embedding, subjects, top_cats, subcats, dates, out_path):
    """Build an interactive Plotly scatter and save as HTML."""
    try:
        import plotly.graph_objects as go
    except ImportError:
        raise ImportError("plotly is required: pip install plotly")

    x = embedding[:, 0]
    y = embedding[:, 1]

    # Assign a color per top_category
    unique_cats = sorted(set(top_cats))
    palette = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
        "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5",
        "#c49c94", "#f7b6d2", "#c7c7c7", "#dbdb8d", "#9edae5",
    ]
    color_map = {cat: palette[i % len(palette)] for i, cat in enumerate(unique_cats)}

    fig = go.Figure()

    for cat in unique_cats:
        mask = [tc == cat for tc in top_cats]
        cat_ids      = [ids[i] for i, m in enumerate(mask) if m]
        cat_x        = [x[i]   for i, m in enumerate(mask) if m]
        cat_y        = [y[i]   for i, m in enumerate(mask) if m]
        cat_subjects = [subjects[i] for i, m in enumerate(mask) if m]
        cat_subcats  = [subcats[i]  for i, m in enumerate(mask) if m]
        cat_dates    = [dates[i]    for i, m in enumerate(mask) if m]

        hover = [
            (
                f"ID: {eid}<br>"
                f"Category: {cat}/{sub}" if sub else f"ID: {eid}<br>Category: {cat}"
            ) + f"<br>Subject: {subj[:80]}<br>Date: {d}<br><i>Click point to open in ECL</i>"
            for eid, sub, subj, d in zip(cat_ids, cat_subcats, cat_subjects, cat_dates)
        ]
        cat_urls = [_entry_url(eid) for eid in cat_ids]

        fig.add_trace(go.Scatter(
            x=cat_x,
            y=cat_y,
            mode='markers',
            name=cat,
            marker=dict(size=5, color=color_map[cat], opacity=0.75),
            text=hover,
            hovertemplate="%{text}<extra></extra>",
            customdata=cat_urls,
        ))

    n_total = len(ids)
    fig.update_layout(
        title=f"SBND ECL — {n_total} entries  (UMAP 2D projection, colored by top category)",
        xaxis_title="UMAP-1",
        yaxis_title="UMAP-2",
        hovermode="closest",
        legend=dict(title="Top category", itemsizing="constant"),
        width=1200,
        height=800,
    )

    # Click a point → open the ECL entry in a new tab
    click_js = """
    var plotDiv = document.getElementById('{plot_id}');
    plotDiv.on('plotly_click', function(data) {
        var url = data.points[0].customdata;
        if (url) { window.open(url, '_blank'); }
    });
    """

    div_id = "elog-map"
    fig.write_html(out_path, div_id=div_id, post_script=click_js.replace("{plot_id}", div_id))
    logger.info(f"Plot saved to: {out_path}")
    print(f"\nVisualization saved → {out_path}")
    print("Open in a browser. Hover to preview, click any point to open the entry in ECL.")


def parse_arguments():
    parser = argparse.ArgumentParser(description="Visualize SBND ECL Qdrant embeddings in 2D.")
    parser.add_argument('--out',       type=str,   default="elog_map.html",
                        help="Output HTML file (default: elog_map.html).")
    parser.add_argument('--category',  type=str,   default=None,
                        help="Only plot entries from this top_category.")
    parser.add_argument('--neighbors', type=int,   default=15,
                        help="UMAP n_neighbors (default: 15; higher = more global structure).")
    parser.add_argument('--min_dist',  type=float, default=0.05,
                        help="UMAP min_dist (default: 0.05; lower = tighter clusters).")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    qdrant = _get_qdrant_client()

    ids, vectors, subjects, top_cats, subcats, dates = fetch_all_vectors(
        qdrant, category_filter=args.category
    )

    if len(ids) == 0:
        print("No entries found. Check your COLLECTION_NAME and QDRANT_HOST.")
        raise SystemExit(1)

    embedding = run_umap(vectors, n_neighbors=args.neighbors, min_dist=args.min_dist)

    build_plot(ids, embedding, subjects, top_cats, subcats, dates, out_path=args.out)
