#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
token_stats.py
--------------
How much of each entry do the embedding models actually see? Tokenizes the
cached entry text (build_text_cache.py) with each model's own tokenizer and
reports the length distribution, how many entries exceed the model's
max_seq_length (everything past it is silently truncated before embedding),
and how many chunks they'd need if we split them instead.

Usage (from anywhere, inside the venv):
    python DebugScripts/token_stats.py
    python DebugScripts/token_stats.py --show 15       # also list the longest entries
"""

import os
import sys
import argparse

import numpy as np

ELOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from transformers.utils import logging as hf_logging
from sentence_transformers import SentenceTransformer

from build_text_cache import load_cache, CACHE_FILE

hf_logging.set_verbosity_error()  # silence "sequence length is longer than..." per entry

MODELS = {
    'mpnet': os.path.join(ELOG_DIR, 'models', 'all-mpnet-base-v2'),
    'mxbai': os.path.join(ELOG_DIR, 'models', 'mxbai-embed-large-v1'),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--show', type=int, default=0,
                        help='List the N longest entries per model.')
    args = parser.parse_args()

    cache = load_cache()
    if not cache:
        sys.exit(f"No cache at {CACHE_FILE} - run build_text_cache.py first.")
    ids = sorted(cache)
    texts = [cache[i]['text'] for i in ids]
    print(f"{len(texts)} cached entries\n")

    for name, path in MODELS.items():
        model = SentenceTransformer(path, device='cpu')
        limit = model.max_seq_length
        lengths = np.array([len(x) for x in model.tokenizer(texts)['input_ids']])
        over = lengths > limit
        lost = np.clip(lengths - limit, 0, None)
        chunks = np.ceil(lengths / limit).astype(int)

        print(f"── {name}  (max_seq_length = {limit}) " + "─" * 30)
        p = np.percentile(lengths, [50, 75, 90, 95, 99])
        print(f"  tokens: median {p[0]:.0f}  p75 {p[1]:.0f}  p90 {p[2]:.0f}  "
              f"p95 {p[3]:.0f}  p99 {p[4]:.0f}  max {lengths.max()}")
        print(f"  truncated entries:  {over.sum()} / {len(lengths)}  ({100 * over.mean():.1f}%)")
        print(f"  tokens never embedded: {100 * lost.sum() / lengths.sum():.1f}% of all tokens")
        if over.any():
            seen = limit / lengths[over]
            print(f"  among truncated entries, median fraction embedded: {100 * np.median(seen):.0f}%")
        print("  chunks needed:  " + "  ".join(
            f"{label}: {n}" for label, n in [
                ('1', (chunks == 1).sum()), ('2', (chunks == 2).sum()),
                ('3-4', ((chunks >= 3) & (chunks <= 4)).sum()), ('5+', (chunks >= 5).sum()),
            ]))
        if args.show:
            print(f"  longest {args.show}:")
            for k in np.argsort(-lengths)[:args.show]:
                rec = cache[ids[k]]
                print(f"    {ids[k]:>6}  {lengths[k]:>6} tok  {rec['subject'][:70]}")
        print()


if __name__ == '__main__':
    main()
