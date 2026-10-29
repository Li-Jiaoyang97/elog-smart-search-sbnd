#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
text_v2.py
----------
"Text cleanup v2": the candidate replacement for the prefix that
run_update.entry_to_text puts in front of every embedded entry.

v1 (what is in the live collections) states each category twice - once
bracketed, once in a natural-language sentence - and prepends "Form: default."
to 4,506 of 9,129 entries, a string that is identical everywhere it appears and
so carries no signal:

    [Cryogenics] This entry is related to Cryogenics. [Cryostat]This entry
    involves Cryostat. Subject: ... Form: default. text: ...

v2 states each category once and drops only the 'default' form name, keeping
the real ones ('Alarm Acknowledged' and friends, which do mark a document type):

    [Cryogenics / Cryostat] Subject: ... text: ...

This module derives v2 from a v1 text-cache record rather than re-fetching from
ECL: the v1 prefix is a pure function of (top_category, subcategory), so it can
be reconstructed and stripped exactly - verified against all 9,129 cached
entries by `python DebugScripts/text_v2.py --check`.

Why derive instead of re-fetch: a re-crawl of 9k entries is ~an hour of ECL
load for an experiment that may not win, and it would also pick up entries
edited since the crawl, confounding the A/B with content drift.
"""

import os
import sys
import json
import argparse

ELOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, ELOG_DIR)

FORM_NOISE = 'Form: default. '


def v1_prefix(top_category, subcategory):
    """Reproduce exactly what augment_with_logs + augment_with_cats prepend.

    Mirrors TextProcessor in DataFetch/text_to_vector.py. The branches there for
    multiple logs/categories are unreachable: ElogFetcherQdrant sets
    logs=[top_category] and run_update passes cats=[subcategory], so both lists
    hold at most one element.
    """
    prefix = ''
    if subcategory:
        prefix = f"[{subcategory}]This entry involves {subcategory}. "
    if top_category:
        # augment_with_logs wraps the cats-augmented text, so it lands outermost.
        prefix = (f"[{top_category}] This entry is related to {top_category}. "
                  + prefix)
    return prefix


def v2_prefix(top_category, subcategory):
    """One bracketed tag, each category named once."""
    if top_category and subcategory:
        return f"[{top_category} / {subcategory}] "
    tag = top_category or subcategory
    return f"[{tag}] " if tag else ''


def to_v2(rec):
    """v1 cache record -> the v2 text to embed. Raises if the v1 prefix is not
    where it should be, rather than silently embedding a mangled string."""
    text = rec['text']
    old = v1_prefix(rec.get('top_category') or '', rec.get('subcategory') or '')
    if old and not text.startswith(old):
        raise ValueError(f"id {rec['id']}: expected v1 prefix {old!r}, got {text[:80]!r}")
    body = text[len(old):].replace(FORM_NOISE, '', 1)
    return v2_prefix(rec.get('top_category') or '', rec.get('subcategory') or '') + body


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--check', action='store_true',
                        help='Derive v2 for every cached entry and report size change.')
    parser.add_argument('--show', type=int, nargs='*', default=None,
                        help='Print v1 and v2 side by side for these entry IDs.')
    args = parser.parse_args()

    from build_text_cache import load_cache
    cache = load_cache()
    if not cache:
        sys.exit("No text cache - run DebugScripts/build_text_cache.py first.")

    if args.show is not None:
        for eid in (args.show or sorted(cache)[:3]):
            rec = cache.get(eid)
            if rec is None:
                print(f"{eid}: not in cache")
                continue
            print(f"--- {eid} ---")
            print(f"v1: {rec['text'][:220]}")
            print(f"v2: {to_v2(rec)[:220]}\n")
        return

    if args.check:
        failures, v1_chars, v2_chars, dropped_form = [], 0, 0, 0
        for eid, rec in cache.items():
            try:
                new = to_v2(rec)
            except ValueError as e:
                failures.append(str(e))
                continue
            v1_chars += len(rec['text'])
            v2_chars += len(new)
            dropped_form += FORM_NOISE in rec['text']
        print(f"{len(cache)} cached entries, {len(failures)} prefix mismatches")
        for f in failures[:10]:
            print(f"  {f}")
        print(f"'Form: default. ' removed from {dropped_form} entries")
        print(f"total chars {v1_chars} -> {v2_chars} ({100 * (v2_chars - v1_chars) / v1_chars:+.1f}%)")
        if failures:
            sys.exit(1)
        return

    parser.print_help()


if __name__ == '__main__':
    main()
