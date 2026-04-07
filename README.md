# Why Small LoRA Datasets Seem Better (But Aren’t)

## Key Finding

CLIP identity correlates with epoch count (r = +0.83), not dataset size.

Smaller datasets appear to perform better because they massively increase
per-image exposure — not because they produce better generalization.

## What This Project Does

I designed a controlled experiment across 15 LoRA training runs
(5 dataset sizes × 3 batch sizes) while holding total training
samples constant.

This isolates the effect of:
- dataset size
- batch size
- epoch count (per-image exposure)

## Why This Matters

Common LoRA advice recommends smaller datasets and specific batch sizes.

This study shows:
- These heuristics are misleading
- Practitioners are unintentionally optimizing for metrics, not quality
- Epoch count (training exposure) is the real control variable

## Full Writeup

- **See full analysis on Hugging Face**: [link]() 