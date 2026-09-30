# Phase 3 Handoff Summary

Status: Phase 3 (Model 1 — LID) — Complete for current checkpoint; approved after sanity-check review. Do not begin Phase 4 without explicit go-ahead.

## Model architecture

- Base model: `xlm-roberta-base`
- Task: token classification for language identification
- Labels: `HI`, `EN`, `OTHER`
- Training strategy: fine-tune the base encoder on weak-labeled token sequences using the model-input track output from Phase 2
- Training script: `models/lid_colab.py`

## Labeling and marker expansion

The weak-labeling heuristic was broadened before the final retrain so Romanized Hindi function words and common inflections were no longer forced into `EN` by default.

- Final marker count: 125
- Marker source file: `data/roman_hindi_markers.json`
- Included coverage: common Hindi verbs, pronouns, postpositions, discourse words, and frequent Romanized variants such as `hona`, `hota`, `hogi`, `hoga`, `ho`, `hue`, `karna`, `karta`, `karenge`, `kregi`, `jana`, `jayega`, `gaya`, `gayi`, `nhi`, `nahi`, `meri`, `tera`, `uska`, `hamara`, `unka`, `wala`, `wale`, `liye`, `sath`, `dono`, `ko`, `ka`, `ki`, `ke`, `se`, `mein`, `par`, and related forms.

## Training data

- Full weak-labeled corpus: `models/lid_full.jsonl`
- Train split: `models/lid_train.jsonl`
- Holdout split: `models/lid_holdout.jsonl`
- Small sample for smoke testing: `models/lid_sample_100.jsonl`
- Packaged dataset: `models/lid_dataset.zip`

The train/holdout files were regenerated after the marker expansion so they reflect the final weak-labeling rules.

## Training and evaluation results

- Training completed successfully for 3 epochs in Colab.
- Weak-holdout accuracy: 99.93%
- Confusion matrix: very low `HI`/`EN` confusion, with only 11 errors total reported.

Important interpretation:
- This 99.93% figure is an optimistic upper bound because the holdout labels were created with the same weak heuristic as the training labels.
- The number should be read as: how well the model reproduced the labeling heuristic, not as verified real-world LID accuracy.

## Qualitative sanity-check findings

A lightweight inference-only sanity check on unseen sentences from the untouched Phase 1 holdout was reviewed manually.

Observed behavior:
- Core Hindi function words such as `hai`, `ki`, `ke`, `se`, `kya`, `nhi`, `aur`, `bhi`, `kuch`, `honi`, and `chahiye` were now identified correctly as `HI`.
- Devanagari tokens remained reliably correct.
- English tokens remained reliably correct.
- Numeric tokens remained reliably correct.

Accepted residual limitation:
- Some inflected Romanized Hindi verb forms and capitalized variants such as `ldko`, `kregi`, and `Eska` can still be misclassified occasionally.
- This is documented as an accepted limitation, not a blocker, for the current checkpoint.

## Artifact persistence

- Colab checkpoints are written to `/content/lid_model` during training.
- That location is ephemeral.
- For later reuse in the full pipeline, the trained model folder must be downloaded or copied to Google Drive before the Colab session resets.
- The repository itself does not store the trained model weights.

## Relevant files

- Training / evaluation script: `models/lid_colab.py`
- Sanity-check inference script: `models/run_lid_sanity_check.py`
- Dataset preparation: `models/prepare_lid_data.py`
- Final marker list: `data/roman_hindi_markers.json`
- README / run instructions: `models/README_LID.md`

## Final status

Phase 3 is complete for the current checkpoint. The model is ready to be retrained from the regenerated weak-labeled files, and the documented sanity-check review indicates the most important Romanized Hindi function words now behave as expected. No Phase 4 work should begin until explicit approval is given.

Prepared by: automation scripts and review in this workspace
Date: 2026-09-12
