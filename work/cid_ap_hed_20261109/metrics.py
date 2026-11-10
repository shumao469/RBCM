"""Shared CID summaries for original BSDS matcher counts.

Aggregation follows Berkeley collect_eval_bdry.m. AP reuses the pinned Round 7
adapter, including its first-observed rule when pooled recall is repeated.
ODS and OIS use CID labels only as descriptive oracle scores. Deployment F
is calculated at the separately supplied source-locked threshold.
"""

from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "round7_20261031"))
from unified_bsds_evaluator import summarize_counts as summarize_round7


THRESHOLDS = np.linspace(0.01, 0.99, 99)


def _validated(counts_by_image, thresholds):
    counts = np.asarray(counts_by_image, dtype=float)
    thresholds = np.asarray(thresholds, dtype=float)
    if (counts.ndim != 3 or not len(counts) or thresholds.ndim != 1
            or not len(thresholds) or counts.shape[1:] != (len(thresholds), 4)):
        raise ValueError("Expected nonempty image × threshold × 4 count array and threshold vector")
    if not np.isfinite(thresholds).all() or np.any(np.diff(thresholds) <= 0):
        raise ValueError("Thresholds must be finite and strictly increasing")
    if not np.isfinite(counts).all() or np.any(counts < 0) or np.any(counts[..., ::2] > counts[..., 1::2]):
        raise ValueError("Match counts must be finite, nonnegative and no greater than their denominators")
    return counts, thresholds


def _f_score(precision, recall):
    return 2 * precision * recall / np.maximum(precision + recall, 1e-12)


def _prf(counts):
    matched_prediction, prediction, matched_truth, truth = np.moveaxis(counts, -1, 0)
    precision = matched_prediction / np.maximum(prediction, 1)
    recall = matched_truth / np.maximum(truth, 1)
    return precision, recall, _f_score(precision, recall)


def source_threshold(counts_by_image, thresholds=THRESHOLDS):
    """Select the first sampled maximum of pooled F on source images only."""
    counts, thresholds = _validated(counts_by_image, thresholds)
    _, _, f = _prf(counts.sum(axis=0))
    return float(thresholds[np.argmax(f)])


def summarize_counts(counts_by_image, thresholds, fixed_threshold):
    """Summarize (matched prediction, prediction, matched GT, GT) counts."""
    counts, thresholds = _validated(counts_by_image, thresholds)
    fixed_matches = np.flatnonzero(np.isclose(thresholds, fixed_threshold, rtol=0, atol=1e-12))
    if len(fixed_matches) != 1:
        raise ValueError("Expected one exact grid position for the source-locked threshold")
    fixed = fixed_matches[0]
    pooled = counts.sum(axis=0)
    curve = summarize_round7(pooled, thresholds)
    precision, recall, f = _prf(pooled)

    # MATLAB linspace(0,1) includes 100 positions on each neighboring PR segment.
    if len(thresholds) > 1:
        fraction = np.linspace(0, 1, 100)
        p = precision[:-1, None] * (1 - fraction) + precision[1:, None] * fraction
        r = recall[:-1, None] * (1 - fraction) + recall[1:, None] * fraction
        ods = _f_score(p, r)
        segment, step = np.unravel_index(np.argmax(ods), ods.shape)
        weight = fraction[step]
        ods_f, ods_p, ods_r = ods[segment, step], p[segment, step], r[segment, step]
        ods_threshold = thresholds[segment] * (1 - weight) + thresholds[segment + 1] * weight
    else:
        ods_f, ods_p, ods_r = f[0], precision[0], recall[0]
        ods_threshold = thresholds[0]

    # Pool raw counts at each image's first discrete max, as Berkeley does for OIS.
    _, _, image_f = _prf(counts)
    best_by_image = np.argmax(image_f, axis=1)
    ois_p, ois_r, ois_f = _prf(counts[np.arange(len(counts)), best_by_image].sum(axis=0))
    return {
        **curve,
        "n_images": len(counts),
        "thresholds": thresholds.tolist(),
        "fixed_development_threshold": float(fixed_threshold),
        "F_fixed": float(f[fixed]),
        "precision_fixed": float(precision[fixed]),
        "recall_fixed": float(recall[fixed]),
        "ODS": float(ods_f),
        "ODS_threshold": float(ods_threshold),
        "ODS_precision": float(ods_p),
        "ODS_recall": float(ods_r),
        "OIS": float(ois_f),
        "OIS_precision": float(ois_p),
        "OIS_recall": float(ois_r),
    }


def paired_ap_difference(first_counts, second_counts, *, seed=4517, resamples=5000):
    """Paired percentile interval for a first-minus-second BSDS-101 AP contrast."""
    first, thresholds = _validated(first_counts, THRESHOLDS)
    second, _ = _validated(second_counts, THRESHOLDS)
    if first.shape != second.shape or resamples < 1:
        raise ValueError("Use the same ordered image set for both candidates and at least one resample")

    def ap(pooled):
        return summarize_round7(pooled, thresholds)["AP_BSDS_101point"]

    n = len(first)
    weights = np.random.default_rng(seed).multinomial(n, np.full(n, 1 / n), size=resamples)
    shape = (resamples, len(thresholds), 4)
    first_draws = (weights @ first.reshape(n, -1)).reshape(shape)
    second_draws = (weights @ second.reshape(n, -1)).reshape(shape)
    differences = [ap(a) - ap(b) for a, b in zip(first_draws, second_draws)]
    low, high = np.quantile(differences, [0.025, 0.975])
    return {
        "delta_AP": float(ap(first.sum(axis=0)) - ap(second.sum(axis=0))),
        "ci95_paired_image_low": float(low),
        "ci95_paired_image_high": float(high),
        "resamples": resamples,
        "seed": seed,
    }
