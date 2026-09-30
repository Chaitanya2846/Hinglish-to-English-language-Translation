# Phase 2 Handoff Summary

Status: Phase 2 (Preprocessing) — FROZEN / Complete; do not proceed to Phase 3 without explicit go-ahead.

## Architecture and track split

The preprocessing stage is intentionally split into two explicit tracks to preserve both demo/teaching value and model-quality requirements.

### Track A — Term-work demo track
- Purpose: demonstrate the syllabus-required NLP steps in a classroom-friendly form
- Sequence: noise removal → mixed-script tokenization → script validation → stopword removal → lemmatization/stemming
- Output: cleaned token stream for demonstration and teaching analysis, e.g. `{sentence_id, filtered_tokens[], lemmatized[]}`
- Use: analysis and term-work examples only

### Track B — Model-input track
- Purpose: preserve the original grammatical meaning of a sentence before translation/model training
- Sequence: noise removal → mixed-script tokenization → script validation; stop here
- Output: preserved sentence representation, e.g. `{sentence_id, cleaned_text, tokens[], script_tags[]}`
- Use: production-facing contract for later Model 3 translation work

### Why the split matters
The key rule is that negation and question words must not be stripped before translation. Removing words such as `nahi`, `kya`, `kahan`, `kaun`, `kyun`, `kaise`, `not`, `what`, `where`, `who`, `why`, and `how` would distort sentence meaning and polarity. The demo track can use aggressive stopword filtering for teaching purposes, but the model-input track is the authoritative representation used later in the pipeline.

## Corpus and processing scope

- Active filtered corpus used for Phase 2: `data/raw/raw_filtered_*.jsonl`
- Total processed comments: 6,583
- Source dataset: frozen Phase 1 filtered corpus, holdout preserved separately and not used in preprocessing
- Final processing target: all 6,583 active comments were processed through the fixed-order preprocessing pipeline

## Final corpus statistics

The full preprocessed corpus was checked against the pipeline logic and produced the following script-tag totals across the active dataset:

- Latin tokens: 79,693
- Devanagari tokens: 895
- Garbage/ambiguous tokens dropped: 2,419
- Total tokens observed: 83,007

This means the active corpus is predominantly Latin-script mixed text, with a minority of Devanagari tokens; garbage tokens are explicitly dropped rather than silently guessed or normalized.

### Track-specific output notes
- Track A output: valid cleaned tokens, stopword-filtered for demonstration, then lemmatized/stemmed where possible
- Track B output: preserved cleaned text + tags without stopword removal or lemmatization, matching the model-input contract
- No silent fallback was introduced for script validation: ambiguous or invalid tokens are tagged as `garbage` and excluded from downstream processing

## Test status

Validation was performed using the project regression suite:

- Test file: `preprocessing/test_preprocessing.py`
- Command run: `./.venv/Scripts/python -m pytest preprocessing/test_preprocessing.py -q`
- Result: 7/7 tests passed

The passing checks cover:
- noise stripping
- mixed-script tokenization
- script validation
- stopword behavior
- lemmatization behavior
- negation/question preservation
- model-input track contract

## Stopword fix confirmation

The stopword bug was fixed and explicitly validated: negation and question words are preserved rather than stripped, which prevents semantic distortion before translation.

### Before / After examples

1. Before: `['nahi', 'kya', 'kahan', 'kaun', 'kyun', 'kaise']` -> After: these tokens remain in the filtered output when the stopword pass runs.
2. Before: `['not', 'what', 'where', 'who', 'why', 'how']` -> After: all remain present in the kept token list, preserving polarity and question semantics.
3. Before: `['nahi kya bhai ye class hai']` -> After: the model-input track returns the cleaned sentence unchanged after script tagging, without stripping `nahi` or `kya`.

This fix was required because removing these terms would change the meaning of the sentence, especially for negation and interrogation.

## Indic NLP status

### Current status: blocked but documented as an accepted limitation
The Indic NLP route was attempted once as requested and the resource repository was cloned successfully to the project workspace at `indic_nlp_resources/`.

However, the runtime still did not expose the required model/resource bundle in an actually usable path for Devanagari morphological processing. In practice, the environment had the `indic-nlp-library` package installed but not the resource bundle needed for runtime stemming/segmentation. This is therefore treated as a known limitation, not an open technical question.

### Scope of the block
- Devanagari token count in the filtered corpus: 895 tokens
- Approximate affected share of total tokens: 895 / 83,007 ≈ 1.08%
- The fallback in use is rule-based stemming/normalization (`_hindi_suffix_stem` in `preprocessing/pipeline.py`) rather than claiming real Indic NLP runtime functionality

This is an accepted limitation for the current frozen Phase 2 state. No further dependency chasing was done beyond the single targeted repo attempt, and the project is intentionally frozen before Phase 3.

## Relevant files

- Preprocessing pipeline: `preprocessing/pipeline.py`
- Validation tests: `preprocessing/test_preprocessing.py`
- Architecture update: `ARCHITECTURE.md` (the §2.2 preprocessing section)
- Roman-Hindi markers export: `data/roman_hindi_markers.json`

## Final status

Phase 2 preprocessing is complete and validated for the purposes of this checkpoint: the required architecture is separated, the stopword fix is confirmed, the tests pass, and the Indic NLP blocker is explicitly documented as a known limitation. Phase 3 (Model 1) remains blocked until explicit go-ahead.

Prepared by: automation scripts and validation runs in this workspace
Date: 2026-09-09
