#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
debug_xml_parse.py
-------------------
Throwaway diagnostic: fetch the raw XML for a given ECL category's search
listing and, if it fails to parse, pinpoint the exact line/character that
breaks well-formedness (usually a raw unescaped '&' in a subject/field).

Usage:
    python debug_xml_parse.py "Photon Detection/XARAPUCA"
    python debug_xml_parse.py "DAQ" --after 7days --limit 100
"""

import os
import sys
import argparse
from urllib.parse import quote
import xml.etree.ElementTree as ET
from dotenv import load_dotenv, find_dotenv

#load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))
load_dotenv(find_dotenv())

with open(os.path.join(os.path.dirname(__file__), '..', 'pwd.txt')) as f:
    ecl_pwd = f.read().strip()

from ecl_api import ECL

USERNAME = os.environ.get('ECL_USERNAME', 'sbndprm')
BASE_URL = os.environ.get('ECL_URL', 'https://dbweb1.fnal.gov:8443/ECL/sbnd/E')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('category', help='e.g. "Photon Detection/XARAPUCA"')
    parser.add_argument('--after', default='30days')
    parser.add_argument('--limit', type=int, default=100)
    args = parser.parse_args()

    ecl = ECL(url=BASE_URL, user=USERNAME, password=ecl_pwd)
    encoded = quote(args.category, safe='/-')
    raw = ecl.search(category=encoded, limit=args.limit, after=args.after)

    print(f"Fetched {len(raw)} chars for category={args.category!r} after={args.after!r}")

    try:
        ET.fromstring(raw)
        print("Parsed OK — no well-formedness issue right now.")
        return
    except ET.ParseError as e:
        print(f"ParseError: {e}")
        lineno, col = e.position  # expat positions are 1-based line, 0-based column

    lines = raw.splitlines()
    idx = lineno - 1

    print(f"\n--- Offending line {lineno} ---")
    if 0 <= idx < len(lines):
        line_text = lines[idx]
        print(repr(line_text))
        if col < len(line_text):
            print(f"Character at column {col}: {line_text[col]!r} (U+{ord(line_text[col]):04X})")

    # Show a few lines of context so you can see which <entry> it's inside
    print("\n--- Context (3 lines before/after) ---")
    for i in range(max(0, idx - 3), min(len(lines), idx + 4)):
        marker = ">>" if i == idx else "  "
        print(f"{marker} {i+1}: {lines[i]}")

    out_path = os.path.join(os.path.dirname(__file__), 'xml_debug_dump.xml')
    with open(out_path, 'w', encoding='utf-8', errors='backslashreplace') as f:
        f.write(raw)
    print(f"\nFull raw response saved to {out_path}")


if __name__ == '__main__':
    main()
