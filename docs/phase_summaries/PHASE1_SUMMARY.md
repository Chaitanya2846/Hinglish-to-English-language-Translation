# Phase 1 Handoff Summary

Status: Phase 1 (Data Collection) — FROZEN / Complete

## Corpus locations and filename patterns

- Active filtered corpus directory: `data/raw/`
- Active files: `raw_filtered_*.jsonl` (pattern: `raw_filtered_<videoid>_<category>_<TIMESTAMP>.jsonl`)
- Archives (pre-phase1 raw and any trimmed/archive files): `data/raw/raw_archive_phase1a/`

## Totals

- Total code-mixed comments (active filtered corpus): 6,583
- Holdout (unbiased, reserved for evaluation only): `data/raw/holdout_unbiased_sample.jsonl` — 400 comments (DO NOT use for training)

## Category breakdown (active filtered corpus)

- Educational: 50.7% (≈3,341 comments)
- Casual (big creators + other vloggers): 49.3% (≈3,242 comments)

## Per-video cap enforcement

- Per-video cap: 200 comments per video enforced.
- Verification: post-trim report confirms no active video file exceeds 200 comments.

## Data sources

- Primary: YouTube (via `scraper/youtube_scraper.py`) using seed channels and search queries.
  - Educational seed examples: Physics Wallah (Alakh Pandey), Khan Sir; queries: "Physics Wallah lecture", "Khan Sir class", "JEE NEET Hindi English", "UPSC Hindi English mix".
  - Casual / big-creator seeds: CarryMinati, BB Ki Vines, Ashish Chanchlani, Technical Guruji, Slayy Point; queries: "CarryMinati video", "Technical Guruji review", "vlog India Hinglish".
- Reddit: deferred / pending approval (not collected for Phase 1).

## PII handling

- PII handling: PII hashing applied at save time inside `scraper/youtube_scraper.py` (raw batch saving step).

## Audit and quality checks

- Audit performed: `scripts/audit_and_clean.py` ran a random 100-sample strict check.
- Observed false-positive rate: 1.0% (1/100) — below the 5% threshold; no cleaning required.

## Known limitations

- Filtering is marker/token-based and can produce occasional false positives (~1% observed in audit sample).
- Roman-Hindi markers list is heuristic and may miss rarer or orthographically variant tokens.
- Holdout is a reservoir sample (N=400) preserved unchanged for evaluation — ensure it is not used for training.

## Relevant files and scripts

- Collection: `scraper/youtube_scraper.py`
- Filtered collection workflow: `scripts/filtered_collect.py`
- Holdout creation: `scripts/filtered_collect.py` (reservoir sample)
- Trimming / cap enforcement: `scripts/trim_caps.py`
- Reporting: `scripts/report_filtered_stats.py`
- Audit: `scripts/audit_and_clean.py`
- Verification / quick scans: `scripts/verify_filtered_corpus.py`

---

If you'd like, I can now produce a compact CSV with per-video counts and category splits, or prepare Phase 1 handoff slides. No Phase 2 actions will be started until you give explicit go-ahead.

Prepared by: automation scripts in this workspace
Date: 2026-09-08
