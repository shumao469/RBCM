# RBCM-Edge

This branch is the code-and-configuration release for **H-RBCM (HED-lite
Retinal Boundary Context Modulation)** and the associated retinal MEA analysis.
It is synchronized with manuscript V5.

The repository intentionally excludes manuscripts, rendered figures, result
tables, raw recordings, image datasets, checkpoints, predictions, and all
historical model variants. Generated outputs remain local and are ignored by
Git.

## Final computational scope

- one trainable HED-lite edge anchor with four encoder stages and three decoder
  branches;
- four outputs derived from the same frozen anchor: `plain_identity`,
  `main_surround`, `no_surround`, and `conv_control`;
- deterministic H-RBCM calibration using normalized Sobel evidence, near/far
  square annuli, signed center-surround contrast, uncertainty gating, and
  logit-space correction;
- source-validation candidate selection followed by frozen same-domain and
  cross-domain evaluation;
- final UME/CME local-population MEA analysis;
- V5 Figure 5 statistics based on within-sample mean centering, median absolute
  residual scaling, and a symmetric one-MAD threshold.

## Source layout

- `MEA_analysis`: final MEA analysis and shared data loaders;
- `MEA_model`: reproducible MEA plotting and statistical-summary programs;
- `edge_model`: anchor training, inference, and evaluation;
- `src/rbcm_edge`: importable HED-lite model and loss implementation;
- `scripts`: calibration, protocol checks, cross-domain evaluation, Figure 5
  statistics, and release validation.

Large inputs and pretrained checkpoints are obtained separately as described
in `DOWNLOADS.md`. The full workflow is documented in `REPRODUCE.md` and
`REPRODUCE.zh-CN.md`.

## Quick validation

```bash
python -m pip install -r requirements-repro.txt
python -m pip install -e .
python scripts/release/verify_paper_release.py --code-root .
python scripts/release/smoke_paper_release.py   --checkpoint-root pretrained --dataset all
python scripts/release/reproduce_release.py   --checkpoint-root pretrained --stage all
```

The shared Python evaluator restores original image sizes, optionally applies
NMS, sweeps fixed thresholds, and uses a target-specific localization
tolerance. Its dilation matcher is a near-official common evaluator, not the
exact BSDS Matlab bipartite matcher; comparisons must use and label one backend
consistently.

