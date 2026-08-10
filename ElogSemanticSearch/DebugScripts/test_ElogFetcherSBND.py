#!/usr/bin/env python3
"""
test_elog_fetcher.py
Quick sanity-check for ElogFetcherQdrantSBND.ElogFetcher
"""

import os
from DataFetch.ElogFetcherQdrantSBND import ElogFetcher

with open('../pwd.txt', 'r') as f:
    ecl_pwd = f.read().strip()  

# ── Credentials (set env vars or hardcode for local testing) ──────────────────
USERNAME = os.environ.get('ECL_USERNAME', 'sbndprm')
PASSWORD = os.environ.get('ECL_PASSWORD', ecl_pwd)

def test_ElogFetcherSBND():

    # Initialize the fetcher
    fetcher = ElogFetcher()

    # 1. Test authentication
    print("=" * 50)
    print("TEST 1: Authentication")
    ok = fetcher.create_authenticated_session(USERNAME, PASSWORD)
    assert ok, "Authentication failed — check credentials/URL"
    print("PASSED\n")

    # 2. Test getting most recent entry ID
    print("=" * 50)
    category = "CRT"
    after = "365days"
    print(f"TEST 2: Most recent entry ID for category '{category}' in the last {after}")
    recent_id = fetcher.check_most_recent_entry_perCategory(category=category, after=after, limit=5) #category='DAQ Cold Electronics', 
    assert recent_id is not None, "Could not retrieve most recent entry ID"
    print(f"Most recent entry ID: {recent_id}")
    entry = fetcher.get_entry(recent_id)
    assert entry is not None, f"Could not fetch entry {recent_id}"
    fetcher._print_entry(entry)
    print("PASSED\n")

    # 3. Test getting most recent entry ID across all categories
    print("=" * 50)
    print("TEST 3: Most recent entry ID across multiple categories")
    #ECL_CATEGORIES_TEST = [
    #    'BNB',
    #    'Building Management'
    #]
    results = fetcher.check_most_recent_entries_allCategories(after="7days", limit=3)#categories=ECL_CATEGORIES_TEST, 
    # Retrive the most recent entry ID across all categories
    for cat, ids in results.items():
        if ids:
            print(f"  [{cat}] Most recent entry ID: {ids}")
            entry = fetcher.get_entry(ids)
            if entry:
                fetcher._print_entry(entry)
        else:
            print(f"  [{cat}] No recent entries found.")

    print("PASSED\n")

    # 4. Fetch a few entries just before the most recent one
    print("=" * 50)
    print("TEST 4: Fetch 3 earlier entries")
    for entry_id in range(max(1, recent_id - 2), recent_id + 1):
        e = fetcher.get_entry(entry_id)
        if e:
            print(f"  Entry {entry_id}: [{', '.join(e['categories'])}] "
                  f"{e['creation_date']} — {e['text'][:80]}...")
        else:
            print(f"  Entry {entry_id}: not found or skipped")
    print("PASSED\n")

    # 5. Test a dedicated entry ID 
    print("=" * 50)
    print("TEST 5: Fetch an existent entry with known content")
    entry = fetcher.get_entry(34646)
    assert entry is not None, "Expected entry not found"
    fetcher._print_entry(entry)
    print("Correctly fetched the entry")
    print("PASSED\n")

    # 6. Test bad entry ID (should return None, not crash)
    print("=" * 50)
    print("TEST 6: Fetch non-existent entry (expect None)")
    bad_entry = fetcher.get_entry(999999999)
    assert bad_entry is None, "Expected None for non-existent entry"
    print("Correctly returned None")
    print("PASSED\n")

    # 7. Test unauthenticated fetcher (should return None gracefully)
    print("=" * 50)
    print("TEST 7: Unauthenticated fetcher (expect None)")
    unauth = ElogFetcher()
    result2 = unauth.check_most_recent_entry_perCategory(category="CRT", after="7days", limit=5)
    assert result2 is None, "Expected None from unauthenticated check"
    print("Correctly returned None for both calls")
    print("PASSED\n")

    print("=" * 50)
    print("All tests passed!")


if __name__ == '__main__':
    test_ElogFetcherSBND()