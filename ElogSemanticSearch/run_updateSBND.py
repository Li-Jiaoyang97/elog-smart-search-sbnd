
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_updateSBND.py
-----------------
Sync SBND ECL entries into Qdrant. Serves two modes via the same
checkpointed ID-walk engine:

  - Initial backfill: `--reset` (or a large `--max_id`) to walk from ID 1.
  - Steady-state update (e.g. cron): run with defaults; the checkpoint file
    (init_checkpoint_<COLLECTION_NAME>.json, one per collection) makes each
    run pick up only new entries since the last run.

Strategy:
  - Auto-detect the current max entry ID across all categories (via --after/
    --limit), unless --max_id overrides it
  - Walk every ID from the checkpointed last_id+1 to max_id in parallel
    chunks
  - Skip entries whose subject contains "checklist" or whose category is in
    SKIP_CATEGORIES
  - Embed and upsert everything else to Qdrant in batches
  - Checkpoint progress after every chunk so the script can be safely
    resumed after an interruption (just re-run the same command)

  - Repair: `--repair` re-probes IDs absent from the collection and re-ingests
    the ones that were lost to transient failures, without touching last_id.

Counters:
  stored/skipped/dropped keep their original meanings. `dropped` is split for
  reporting into `absent` (ECL has no such ID — expected sequence gaps, tens of
  thousands of them) and `failed` (a real entry we could not ingest). Only
  `failed` IDs are recorded in the checkpoint's `dropped_ids` and retried on the
  next run.

Usage:
  python run_updateSBND.py                  # steady-state update, auto-detect max ID
  python run_updateSBND.py --max_id 12000   # override max ID
  python run_updateSBND.py --dry_run        # fetch + filter but skip Qdrant writes and checkpoint advance
  python run_updateSBND.py --reset          # delete checkpoint and start over (initial backfill)
  python run_updateSBND.py --repair --dry_run          # preview what repair would recover
  python run_updateSBND.py --repair --repair_from 40438 --repair_to 40836

@author: haydenh
@modified by jiaoyang for SBND ECL
"""

import os
import json
import logging
import argparse
import tempfile
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

# Resolve pwd.txt relative to this file, falling back to the cwd-relative path
# this module has always used. Superset of the old behaviour: it keeps working
# from ElogSemanticSearch/ and additionally lets tools import this module from
# anywhere without the os.chdir() dance the DebugScripts do.
_PWD_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pwd.txt')
with open(_PWD_PATH if os.path.exists(_PWD_PATH) else 'pwd.txt', 'r') as f:
    ecl_pwd = f.read().strip()

USERNAME = os.environ.get('ECL_USERNAME', 'sbndprm')
PASSWORD = os.environ.get('ECL_PASSWORD', ecl_pwd)

from DataFetch.ElogFetcherQdrantSBND import ElogFetcher, EntryFetchError, EntryParseError
from DataFetch.text_to_vectorSBND import TextProcessor
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, PointStruct, VectorParams

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def add_file_logging(log_file=None):
    """
    Mirror all log output to a file alongside the existing stderr handler.

    Without this, the per-ID reason a fetch failed is printed once and lost,
    which is precisely what made the earlier dropped entries impossible to
    diagnose after the fact. Attaches to the ROOT logger so the fetcher's
    messages (DataFetch.ElogFetcherQdrantSBND) are captured too.
    """
    if log_file is None:
        log_dir = os.path.join(os.path.dirname(__file__), 'logs')
        os.makedirs(log_dir, exist_ok=True)
        stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = os.path.join(log_dir, f'update_{collection_name}_{stamp}.log')
    else:
        parent = os.path.dirname(os.path.abspath(log_file))
        if parent:
            os.makedirs(parent, exist_ok=True)

    handler = logging.FileHandler(log_file)
    handler.setFormatter(
        logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    )
    logging.getLogger().addHandler(handler)
    logger.info(f"Logging to file: {log_file}")
    return log_file

model_path = os.environ.get("MODEL_PATH")
if not model_path:
    raise RuntimeError("Environment variable MODEL_PATH must be set.")

collection_name = os.environ.get("COLLECTION_NAME")
if not collection_name:
    raise RuntimeError("Environment variable COLLECTION_NAME must be set.")

CHECKPOINT_FILE = os.path.join(os.path.dirname(__file__), f"init_checkpoint_{collection_name}.json")

# Entries whose subject contains any of these words (case-insensitive) are skipped.
SKIP_SUBJECT_KEYWORDS = ["checklist"]

# Entries whose full category (top_category/subcategory) matches are skipped.
# Use the same format as ECL: "Top" or "Top/Sub".
SKIP_CATEGORIES = ["DAQ/Automation"]


# ── Clear collection helper (for --clear flag) ────────────────────────────────
def clear_collection():
    """Delete all points from the configured Qdrant collection."""
    qdrant_client = _get_qdrant_client()
    qdrant_client.delete_collection(collection_name=collection_name)
    logger.info(f"Deleted collection '{collection_name}'.")
    return True


# ── Filtering ─────────────────────────────────────────────────────────────────

def should_skip(entry):
    subject = entry.get('subject', '').lower()
    text    = entry.get('text', '').lower()
    if any(kw in subject or kw in text for kw in SKIP_SUBJECT_KEYWORDS):
        return True

    top = entry.get('top_category', '')
    sub = entry.get('subcategory', '')
    full_category = f"{top}/{sub}" if sub else top
    if full_category in SKIP_CATEGORIES:
        return True

    return False


# ── Checkpoint helpers ────────────────────────────────────────────────────────

def _atomic_write_json(path, obj):
    """
    Write JSON via a temp file in the same directory, then os.replace().

    A plain open(path, 'w') truncates before writing, so an interrupt mid-write
    leaves a truncated file that won't parse — and the natural operator reaction
    to an unreadable checkpoint is --reset, which re-walks tens of thousands of
    IDs. os.replace() is atomic on POSIX, so the file is either the old content
    or the new one, never a fragment.
    """
    directory = os.path.dirname(os.path.abspath(path)) or '.'
    fd, tmp = tempfile.mkstemp(dir=directory, prefix='.tmp_', suffix='.json')
    try:
        with os.fdopen(fd, 'w') as f:
            json.dump(obj, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except Exception:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def _empty_checkpoint():
    return {"last_id": 0, "stored": 0, "skipped": 0, "dropped": 0,
            "dropped_ids": {"error": [], "parse": []}}


def load_checkpoint_raw():
    """Load the checkpoint with defaults applied, without logging. {} if absent."""
    if not os.path.exists(CHECKPOINT_FILE):
        return dict(_empty_checkpoint())
    try:
        with open(CHECKPOINT_FILE) as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise RuntimeError(
            f"Checkpoint file {CHECKPOINT_FILE} is not valid JSON ({e}). "
            f"Inspect it by hand — do NOT --reset unless you intend to re-walk "
            f"every ID from 1."
        ) from e

    # Older checkpoint files predate these keys.
    data.setdefault("dropped", 0)
    dropped_ids = data.setdefault("dropped_ids", {})
    dropped_ids.setdefault("error", [])
    dropped_ids.setdefault("parse", [])
    return data


def load_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        data = load_checkpoint_raw()
        n_dropped_ids = sum(len(v) for v in data["dropped_ids"].values())
        logger.info(
            f"Resuming from checkpoint: last processed ID={data['last_id']}, "
            f"stored={data['stored']}, skipped={data['skipped']}, dropped={data['dropped']}"
            + (f", recorded dropped IDs pending retry={n_dropped_ids}" if n_dropped_ids else "")
        )
        return data
    return _empty_checkpoint()


def save_checkpoint(last_id, stored, skipped, dropped, dropped_ids=None):
    """
    Persist progress. Read-modify-write so that keys written by another mode
    (e.g. --repair) are preserved rather than silently dropped by a plain
    4-key overwrite.

    dropped_ids, when given, is a {"error": [...], "parse": [...]} dict that
    REPLACES the stored one (callers own the full current set). Nonexistent IDs
    are deliberately never recorded here — there are ~32k of them and they are
    not actionable; they belong in the repair sidecar.
    """
    data = load_checkpoint_raw()
    data.update({"last_id": last_id, "stored": stored,
                 "skipped": skipped, "dropped": dropped})
    if dropped_ids is not None:
        data["dropped_ids"] = {
            "error": sorted(set(dropped_ids.get("error", []))),
            "parse": sorted(set(dropped_ids.get("parse", []))),
        }
    _atomic_write_json(CHECKPOINT_FILE, data)


# ── Max ID detection ──────────────────────────────────────────────────────────

def detect_max_id(fetcher, after='7days', limit=100):
    """Find the current highest entry ID by checking the most recent entry across all categories."""
    logger.info(f"Auto-detecting max entry ID across all categories (after={after}, limit={limit})...")
    results = fetcher.check_most_recent_entries_allCategories(after=after, limit=limit)
    ids = [v for v in results.values() if v is not None]
    if not ids:
        raise RuntimeError("Could not detect any entry IDs. Check ECL authentication.")
    max_id = max(ids)
    logger.info(f"Detected max entry ID: {max_id}")
    return max_id


# ── Entry processing ──────────────────────────────────────────────────────────

def entry_to_point(entry, processor):
    """Embed a parsed entry dict and return a Qdrant PointStruct."""
    main_text = entry.get('text', '')
    comments_text = " ".join(c.get('text', '') for c in entry.get('comments', []))
    combined = f"{main_text} {comments_text}".strip()

    clean_text = processor.process_text(combined)
    augmented_text = clean_text

    logs = entry.get('logs', [])
    subcategory = entry.get('subcategory', '')
    cats = [subcategory] if subcategory else []

    if cats:
        augmented_text = processor.augment_with_cats(augmented_text, cats)
    augmented_text = processor.augment_with_logs(augmented_text, logs)

    vector        = processor.generate_embeddings(augmented_text).tolist()
    creation_date = processor.date_to_rfc3339(entry.get('creation_date', ''))

    return PointStruct(
        id=entry['id'],
        vector=vector,
        payload={
            'subject':       entry.get('subject', ''),
            'top_category':  entry.get('top_category', ''),
            'subcategory':   entry.get('subcategory', ''),
            'creation_date': creation_date,
        }
    )


# ── Qdrant helpers ────────────────────────────────────────────────────────────

def _get_qdrant_client():
    return QdrantClient(host=os.environ.get("QDRANT_HOST"), port=6333)


def ensure_collection(qdrant_client, vector_size):
    existing = {c.name for c in qdrant_client.get_collections().collections}
    if collection_name not in existing:
        qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )
        logger.info(f"Created collection '{collection_name}' (dim={vector_size}, cosine).")
    else:
        # .env carries MODEL_PATH and COLLECTION_NAME as a commented-out pair, so
        # flipping one and forgetting the other is easy — and would write vectors
        # from the wrong model into a live collection. Same probe as
        # DebugScripts/sync_missing_entries.py.
        actual = qdrant_client.get_collection(collection_name).config.params.vectors.size
        if actual != vector_size:
            raise RuntimeError(
                f"Model/collection mismatch: model at MODEL_PATH produces "
                f"{vector_size}-dim vectors but collection '{collection_name}' "
                f"expects {actual}. Check MODEL_PATH and COLLECTION_NAME in .env."
            )
        logger.info(
            f"Collection '{collection_name}' already exists (dim={actual}) — will upsert into it."
        )


# ── Repair mode ───────────────────────────────────────────────────────────────
#
# Recovers entries that earlier runs lost silently, back when a failed fetch was
# indistinguishable from a nonexistent ID and the checkpoint advanced past it
# without recording anything.
#
# Self-contained by design: the only ground truth is "which IDs are absent from
# THIS collection". It never consults a second collection, so it keeps working
# once the 768/1024 A/B test collapses to a single collection.
#
# Absence is dominated by correctly-skipped entries (~80%), so re-probing every
# absent ID on every run would be enormous waste. The sidecar caches the two
# terminal classifications — nonexistent and skipped — so each ID is probed at
# most once.

REPAIR_STATE_FILE = os.path.join(
    os.path.dirname(__file__), f"repair_state_{collection_name}.json"
)


def skip_rules_fingerprint():
    """
    Hash of the active skip rules. If they change, cached `skipped`
    classifications are stale — an ID skipped under the old rules may now
    qualify — so the repair drops that bucket and re-probes.
    """
    import hashlib
    payload = json.dumps(
        {"subjects": sorted(SKIP_SUBJECT_KEYWORDS), "categories": sorted(SKIP_CATEGORIES)},
        sort_keys=True,
    )
    return "sha256:" + hashlib.sha256(payload.encode()).hexdigest()[:16]


def load_repair_state(path=None):
    path = path or REPAIR_STATE_FILE
    fingerprint = skip_rules_fingerprint()
    if not os.path.exists(path):
        return {"version": 1, "collection": collection_name,
                "skip_fingerprint": fingerprint,
                "nonexistent": [], "skipped": [], "repaired": []}
    with open(path) as f:
        state = json.load(f)
    state.setdefault("nonexistent", [])
    state.setdefault("skipped", [])
    state.setdefault("repaired", [])
    if state.get("skip_fingerprint") != fingerprint:
        logger.warning(
            "Skip rules changed since the repair cache was written — discarding "
            f"{len(state['skipped'])} cached 'skipped' classifications so they are "
            "re-evaluated. Cached 'nonexistent' entries are unaffected."
        )
        state["skipped"] = []
        state["skip_fingerprint"] = fingerprint
    return state


def save_repair_state(state, path=None):
    path = path or REPAIR_STATE_FILE
    state["nonexistent"] = sorted(set(state["nonexistent"]))
    state["skipped"] = sorted(set(state["skipped"]))
    state["repaired"] = sorted(set(state["repaired"]))
    state["updated_utc"] = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    _atomic_write_json(path, state)


def scroll_present_ids(qdrant_client, id_from=None, id_to=None, page=1000):
    """Every point ID currently in the collection (optionally clipped to a range)."""
    present = set()
    offset = None
    while True:
        points, offset = qdrant_client.scroll(
            collection_name=collection_name, limit=page, offset=offset,
            with_payload=False, with_vectors=False,
        )
        for p in points:
            pid = int(p.id)
            if (id_from is None or pid >= id_from) and (id_to is None or pid <= id_to):
                present.add(pid)
        if offset is None:
            return present


def run_repair(username, password, id_from, id_to, recheck=False,
               max_workers=8, chunk_size=100, batch_size=100, dry_run=False,
               state_path=None):
    """
    Re-probe IDs absent from the collection and ingest only the genuine drops.

    Never touches the checkpoint's last_id. Skipped entries are re-filtered
    through should_skip() on fresh data, so a checklist can never be resurrected
    by the repair — the cache can only ever REMOVE an ID from the candidate
    list, never cause one to be ingested.
    """
    processor = TextProcessor(model_name=model_path)
    qdrant_client = _get_qdrant_client()
    ensure_collection(
        qdrant_client, vector_size=processor.model.get_sentence_embedding_dimension()
    )

    state = load_repair_state(state_path)
    present = scroll_present_ids(qdrant_client, id_from, id_to)

    # If someone ran --clear, previously-repaired IDs are gone from the
    # collection and must become candidates again. Only re-evaluate IDs inside
    # this run's range: `present` is clipped to it, so intersecting the whole
    # set would discard repairs recorded for other ranges.
    in_range = {i for i in state["repaired"] if id_from <= i <= id_to}
    state["repaired"] = sorted((set(state["repaired"]) - in_range) | (in_range & present))

    candidates = set(range(id_from, id_to + 1)) - present
    cached_nonexistent = set(state["nonexistent"])
    cached_skipped = set(state["skipped"])
    if not recheck:
        candidates -= cached_nonexistent
        candidates -= cached_skipped
    candidates = sorted(candidates)

    logger.info(
        f"Repair scope IDs {id_from}–{id_to}: {len(present)} present, "
        f"{len(candidates)} to probe "
        f"(cache: {len(cached_nonexistent)} nonexistent, {len(cached_skipped)} skipped"
        f"{'; IGNORED via --repair_recheck' if recheck else ''})."
    )
    if not candidates:
        logger.info("Nothing to probe — collection is complete for this range.")
        return

    fetcher = ElogFetcher()
    if not fetcher.create_authenticated_session(username, password):
        raise RuntimeError("ECL authentication failed.")

    tally = {"repaired": 0, "skipped": 0, "nonexistent": 0, "unresolved": 0}
    pending = []

    def flush():
        if pending and not dry_run:
            qdrant_client.upsert(collection_name=collection_name, points=pending)
            state["repaired"].extend(p.id for p in pending)
            tally["repaired"] += len(pending)
            logger.info(f"Repaired {len(pending)} entry/entries (total {tally['repaired']}).")
            pending.clear()
        elif pending:
            tally["repaired"] += len(pending)
            pending.clear()

    for chunk_start in range(0, len(candidates), chunk_size):
        chunk = candidates[chunk_start:chunk_start + chunk_size]

        results, errors = {}, {}
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(fetcher.get_entry, eid): eid for eid in chunk}
            for future in as_completed(futures):
                eid = futures[future]
                try:
                    results[eid] = future.result()
                except EntryFetchError as e:
                    results[eid] = None
                    errors[eid] = str(e)
                except Exception as e:
                    results[eid] = None
                    errors[eid] = f"{type(e).__name__}: {e}"

        for eid in chunk:
            entry = results.get(eid)

            if entry is None:
                if eid in errors:
                    tally["unresolved"] += 1
                    logger.warning(f"ID {eid}: unresolved — {errors[eid]}")
                else:
                    tally["nonexistent"] += 1
                    state["nonexistent"].append(eid)
                continue

            if should_skip(entry):
                # The guard. Re-evaluated on freshly fetched data every time.
                tally["skipped"] += 1
                state["skipped"].append(eid)
                logger.debug(f"ID {eid}: skipped — subject={entry.get('subject','')!r}")
                continue

            if dry_run:
                logger.info(
                    f"[DRY RUN] ID {eid}: WOULD REPAIR — "
                    f"category='{entry.get('top_category','')}/{entry.get('subcategory','')}' "
                    f"subject={entry.get('subject','')!r}"
                )
                tally["repaired"] += 1
                continue

            try:
                pending.append(entry_to_point(entry, processor))
            except Exception as e:
                tally["unresolved"] += 1
                logger.warning(f"ID {eid}: unresolved — processing — {type(e).__name__}: {e}")
                continue

            if len(pending) >= batch_size:
                flush()

        # Flush before persisting: the cache may never claim an ID whose point
        # is still only in memory.
        flush()
        if not dry_run:
            save_repair_state(state, state_path)

    flush()
    if not dry_run:
        save_repair_state(state, state_path)

    logger.info(
        f"Repair done ({'DRY RUN — nothing written' if dry_run else 'live'}).  "
        f"Repaired: {tally['repaired']}  |  Skipped: {tally['skipped']}  |  "
        f"Nonexistent: {tally['nonexistent']}  |  Unresolved: {tally['unresolved']}"
    )
    if tally["unresolved"]:
        logger.warning(
            f"{tally['unresolved']} ID(s) could not be resolved this pass and were NOT "
            f"cached — re-run the repair to retry them."
        )


# ── Retry of previously-dropped IDs ───────────────────────────────────────────

def _retry_dropped(retry_ids, fetcher, processor, qdrant_client, batch_size=100):
    """
    Re-attempt IDs recorded as dropped by an earlier run.

    Returns (recovered, still_failing) where recovered is
    {"stored": n, "skipped": n} and still_failing is the {"error": [...],
    "parse": [...]} dict to carry forward.

    should_skip() is re-applied here, so an ID that now matches the skip rules
    is retired as skipped rather than ingested — a dropped entry never becomes
    a way to smuggle a checklist into the collection.
    """
    recovered = {"stored": 0, "skipped": 0}
    still_failing = {"error": [], "parse": []}
    pending = []

    def flush():
        if pending:
            qdrant_client.upsert(collection_name=collection_name, points=pending)
            recovered["stored"] += len(pending)
            pending.clear()

    for eid in retry_ids:
        try:
            entry = fetcher.get_entry(eid)
        except EntryParseError as e:
            still_failing["parse"].append(eid)
            logger.warning(f"ID {eid}: retry failed — parse — {e}")
            continue
        except EntryFetchError as e:
            still_failing["error"].append(eid)
            logger.warning(f"ID {eid}: retry failed — error — {e}")
            continue

        if entry is None:
            # Now reads as a genuine gap. Retire it: it is not actionable and
            # keeping it would make us re-fetch it forever.
            logger.info(f"ID {eid}: retry — ECL has no entry at this ID; retiring from retry list")
            continue

        if should_skip(entry):
            recovered["skipped"] += 1
            logger.info(f"ID {eid}: retry — matches skip rules; retiring as skipped")
            continue

        try:
            pending.append(entry_to_point(entry, processor))
        except Exception as e:
            still_failing["error"].append(eid)
            logger.warning(f"ID {eid}: retry failed — processing — {type(e).__name__}: {e}")
            continue

        if len(pending) >= batch_size:
            flush()

    flush()
    return recovered, still_failing


# ── Main ──────────────────────────────────────────────────────────────────────

def run_updateSBND(username, password, after='7days', limit=100, max_id=None,
                   max_workers=8, chunk_size=100, batch_size=500, dry_run=False):
    logger.info("Initializing text processor and ECL fetcher...")
    processor = TextProcessor(model_name=model_path)
    fetcher   = ElogFetcher()
    if not fetcher.create_authenticated_session(username, password):
        raise RuntimeError("ECL authentication failed.")

    if max_id is None:
        max_id = detect_max_id(fetcher, after=after, limit=limit)

    checkpoint = load_checkpoint()
    start_id   = checkpoint["last_id"] + 1
    stored     = checkpoint["stored"]
    skipped    = checkpoint["skipped"]
    dropped    = checkpoint["dropped"]

    # Per-run sub-counters splitting `dropped` into its two causes. `dropped`
    # keeps its historical meaning (everything that produced no point) so old
    # checkpoint values stay comparable.
    absent = 0   # ECL has no entry at this ID — expected sequence gaps
    failed = 0   # a real entry we could not ingest — actionable

    # IDs whose failure is worth retrying next run. Nonexistent IDs are never
    # added: there are ~32k of them and re-fetching them is pure waste.
    dropped_ids = {
        "error": list(checkpoint["dropped_ids"].get("error", [])),
        "parse": list(checkpoint["dropped_ids"].get("parse", [])),
    }

    # ── Retry previously-dropped IDs before walking new ground ────────────────
    # This is what makes a drop self-healing: a transient failure recorded by an
    # earlier run gets another chance here, so entries stop falling through the
    # gap the advancing checkpoint used to create.
    retry_ids = sorted(set(dropped_ids["error"]) | set(dropped_ids["parse"]))
    if retry_ids and not dry_run:
        logger.info(f"Retrying {len(retry_ids)} previously-dropped ID(s) before new IDs...")
        qdrant_client = _get_qdrant_client()
        ensure_collection(
            qdrant_client, vector_size=processor.model.get_sentence_embedding_dimension()
        )
        recovered, still_failing = _retry_dropped(
            retry_ids, fetcher, processor, qdrant_client
        )
        stored  += recovered["stored"]
        skipped += recovered["skipped"]
        dropped -= (recovered["stored"] + recovered["skipped"])
        dropped_ids = still_failing
        save_checkpoint(checkpoint["last_id"], stored, skipped, dropped,
                        dropped_ids=dropped_ids)
        logger.info(
            f"Retry pass: recovered={recovered['stored']} "
            f"now-skipped={recovered['skipped']} "
            f"still-failing={sum(len(v) for v in still_failing.values())}"
        )

    if start_id > max_id:
        logger.info("Checkpoint already at max ID — nothing to do.")
        return

    total = max_id - start_id + 1
    logger.info(f"Processing IDs {start_id} → {max_id}  ({total} IDs to attempt).")

    if not dry_run:
        qdrant_client = _get_qdrant_client()
        ensure_collection(qdrant_client, vector_size=processor.model.get_sentence_embedding_dimension())

    pending_points = []  # accumulate PointStructs until batch_size reached

    for chunk_start in range(start_id, max_id + 1, chunk_size):
        chunk_ids = list(range(chunk_start, min(chunk_start + chunk_size, max_id + 1)))

        # ── Parallel fetch ────────────────────────────────────────────────────
        fetch_results = {}
        fetch_errors  = {}   # eid -> (kind, reason); kind is "error" or "parse"
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(fetcher.get_entry, eid): eid for eid in chunk_ids}
            for future in as_completed(futures):
                eid = futures[future]
                try:
                    fetch_results[eid] = future.result()
                except EntryParseError as e:
                    fetch_results[eid] = None
                    fetch_errors[eid] = ("parse", str(e))
                except EntryFetchError as e:
                    fetch_results[eid] = None
                    fetch_errors[eid] = ("error", str(e))
                except Exception as e:
                    fetch_results[eid] = None
                    fetch_errors[eid] = ("error", f"{type(e).__name__}: {e}")

        # ── Process in ID order ───────────────────────────────────────────────
        for eid in chunk_ids:
            entry = fetch_results.get(eid)

            if entry is None:
                dropped += 1
                failure = fetch_errors.get(eid)
                if failure is None:
                    # get_entry() returns None ONLY for a genuine gap in the ID
                    # sequence. Normal and expected — tens of thousands of these,
                    # so debug level and deliberately NOT recorded for retry.
                    absent += 1
                    logger.debug(f"ID {eid}: absent — ECL has no entry at this ID")
                else:
                    kind, reason = failure
                    failed += 1
                    dropped_ids[kind].append(eid)
                    logger.warning(
                        f"ID {eid}: DROPPED — {kind} — {reason} "
                        f"(recorded for retry on the next run)"
                    )
                continue

            if should_skip(entry):
                skipped += 1
                logger.debug(f"ID {eid}: skipped  subject='{entry.get('subject', '')}'")
                continue

            if dry_run:
                logger.info(
                    f"[DRY RUN] ID {eid}: would store — "
                    f"category='{entry.get('top_category', '')}' "
                    f"subject='{entry.get('subject', '')}'"
                )
                stored += 1
                continue

            try:
                point = entry_to_point(entry, processor)
                pending_points.append(point)
            except Exception as e:
                dropped += 1
                failed += 1
                dropped_ids["error"].append(eid)
                logger.warning(
                    f"ID {eid}: DROPPED — processing error — {type(e).__name__}: {e} "
                    f"(recorded for retry on the next run)"
                )
                continue

            # Upsert when batch is full
            if len(pending_points) >= batch_size:
                qdrant_client.upsert(collection_name=collection_name, points=pending_points)
                stored += len(pending_points)
                logger.info(
                    f"Upserted {len(pending_points)} points  |  "
                    f"total stored={stored}  skipped={skipped}  dropped={dropped}  last_id={eid}"
                )
                pending_points = []

        # Checkpoint at chunk boundary (skip entirely during --dry_run so a
        # preview run never permanently marks IDs as processed). Any pending
        # points must be flushed first: the checkpoint may never advance past
        # entries that are still only in memory, or an interruption would lose
        # them permanently (the resume skips past their IDs).
        #
        # Advancing last_id past a dropped ID is safe now that dropped_ids
        # records which ones failed — the next run retries them up front.
        if not dry_run:
            if pending_points:
                qdrant_client.upsert(collection_name=collection_name, points=pending_points)
                stored += len(pending_points)
                logger.info(
                    f"Upserted {len(pending_points)} points  |  "
                    f"total stored={stored}  skipped={skipped}  dropped={dropped}  last_id={chunk_ids[-1]}"
                )
                pending_points = []
            save_checkpoint(chunk_ids[-1], stored, skipped, dropped, dropped_ids=dropped_ids)
        logger.info(
            f"Chunk done: IDs {chunk_ids[0]}–{chunk_ids[-1]}  |  "
            f"stored={stored}  skipped={skipped}  dropped={dropped} "
            f"(absent={absent} failed={failed})  pending={len(pending_points)}"
        )

    # Final partial batch
    if pending_points and not dry_run:
        qdrant_client.upsert(collection_name=collection_name, points=pending_points)
        stored += len(pending_points)
        save_checkpoint(max_id, stored, skipped, dropped, dropped_ids=dropped_ids)

    n_pending_retry = sum(len(v) for v in dropped_ids.values())
    logger.info(
        f"Done.  Stored: {stored}  |  Skipped: {skipped}  |  Dropped: {dropped}  |  Max ID: {max_id}"
    )
    logger.info(
        f"This run: absent={absent} (no such ID in ECL)  failed={failed} (real entries not ingested)"
    )
    if n_pending_retry:
        logger.warning(
            f"{n_pending_retry} ID(s) recorded in the checkpoint for retry on the next run: "
            f"{sorted(set(dropped_ids['error']) | set(dropped_ids['parse']))[:20]}"
        )


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_arguments():
    parser = argparse.ArgumentParser(description='Run the SBND ECL update script.')
    parser.add_argument('--after', type=str, default='7days',
                        help='Time window used only for auto-detecting the max entry ID (e.g. 7days, 30days).')
    parser.add_argument('--limit', type=int, default=100,
                        help='Per-category limit used only for auto-detecting the max entry ID.')
    parser.add_argument('-u', '--username', type=str, default=USERNAME, help='Username for authentication.')
    parser.add_argument('-p', '--password', type=str, default=PASSWORD, help='Password for authentication.')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable debug-level logging.')
    parser.add_argument('--max_id', type=int, default=None,
                        help='Override auto-detected max entry ID.')
    parser.add_argument('--max_workers', type=int, default=8,
                        help='Parallel fetch threads (reduce if server resets connections).')
    parser.add_argument('--chunk_size', type=int, default=100,
                        help='IDs fetched per parallel batch (also checkpoint frequency).')
    parser.add_argument('--batch_size', type=int, default=500,
                        help='Qdrant upsert batch size.')
    parser.add_argument('--dry_run', action='store_true',
                        help='Fetch and filter but do not write to Qdrant or advance the checkpoint.')
    parser.add_argument('--reset', action='store_true',
                        help='Delete the checkpoint file and start from ID 1.')
    parser.add_argument('--clear', action='store_true',
                        help='Delete all points from the Qdrant collection before updating.')
    parser.add_argument('--log_file', type=str, default=None,
                        help='Write logs here (default: logs/update_<COLLECTION>_<timestamp>.log).')

    repair = parser.add_argument_group('repair mode')
    repair.add_argument('--repair', action='store_true',
                        help='Re-probe IDs absent from the collection and ingest genuine drops. '
                             'Never advances the checkpoint. Combine with --dry_run to preview.')
    repair.add_argument('--repair_from', type=int, default=None,
                        help='First ID to consider (default: lowest ID present in the collection).')
    repair.add_argument('--repair_to', type=int, default=None,
                        help="Last ID to consider (default: the checkpoint's last_id).")
    repair.add_argument('--repair_recheck', action='store_true',
                        help='Ignore the cached nonexistent/skipped classifications and re-probe '
                             'every absent ID in range.')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    if args.verbose:
        # Root level, not just this module's: the fetcher logs under its own
        # logger name and its debug lines would otherwise stay invisible.
        logging.getLogger().setLevel(logging.DEBUG)

    add_file_logging(args.log_file)

    if args.repair:
        if args.reset or args.clear:
            raise SystemExit("--repair cannot be combined with --reset or --clear.")

        checkpoint = load_checkpoint()
        qdrant_client = _get_qdrant_client()

        repair_to = args.repair_to if args.repair_to is not None else checkpoint["last_id"]
        if args.repair_from is not None:
            repair_from = args.repair_from
        else:
            present = scroll_present_ids(qdrant_client)
            repair_from = min(present) if present else 1

        if repair_to > checkpoint["last_id"]:
            # Beyond last_id is the ingest's territory; repairing there would
            # create points a later normal run re-walks and re-counts.
            logger.warning(
                f"--repair_to {repair_to} is beyond the checkpoint's last_id "
                f"({checkpoint['last_id']}); clamping. Run a normal update for newer IDs."
            )
            repair_to = checkpoint["last_id"]
        if repair_from > repair_to:
            raise SystemExit(f"Empty repair range: {repair_from} > {repair_to}.")

        run_repair(
            username=args.username,
            password=args.password,
            id_from=repair_from,
            id_to=repair_to,
            recheck=args.repair_recheck,
            max_workers=args.max_workers,
            chunk_size=args.chunk_size,
            batch_size=min(args.batch_size, 100),
            dry_run=args.dry_run,
        )
        raise SystemExit(0)

    if args.reset and os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)
        logger.info("Checkpoint deleted — starting from ID 1.")

    if args.clear:
        logger.warning(f"--clear flag set: deleting all points from collection '{collection_name}'.")
        clear_collection()

    run_updateSBND(
        username=args.username,
        password=args.password,
        after=args.after,
        limit=args.limit,
        max_id=args.max_id,
        max_workers=args.max_workers,
        chunk_size=args.chunk_size,
        batch_size=args.batch_size,
        dry_run=args.dry_run,
    )
