"""Unit tests for the calibration metric core.

Every expected value here is computed by hand in the comments so the tests
double as a spec. No model, no network, no GPU.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from honest_quant.eval import (
    compute_auroc,
    compute_brier,
    compute_ece,
    compute_mce,
    confident_error_rate,
    evaluate,
    parse_answer_and_confidence,
    reliability_curve,
)


# ---------------------------------------------------------------------------
# ECE
# ---------------------------------------------------------------------------


def test_ece_two_bins_hand_computed():
    # conf 0.1 -> bin [0.1,0.2), conf 0.9 -> bin [0.9,1.0)
    # bin low: acc 0, conf 0.1, gap 0.1, weight 2/4
    # bin high: acc 1, conf 0.9, gap 0.1, weight 2/4
    # ECE = 0.5*0.1 + 0.5*0.1 = 0.1
    conf = [0.1, 0.1, 0.9, 0.9]
    corr = [0, 0, 1, 1]
    assert compute_ece(conf, corr, n_bins=10) == pytest.approx(0.1)


def test_ece_perfectly_calibrated_is_zero():
    # In each bin, accuracy exactly equals mean confidence.
    conf = [0.2, 0.2, 0.2, 0.2, 0.2]  # bin acc must be 0.2 -> 1 of 5 correct
    corr = [1, 0, 0, 0, 0]
    assert compute_ece(conf, corr, n_bins=10) == pytest.approx(0.0)


def test_ece_fully_confident_and_wrong_is_one():
    # confidence 1.0 everywhere, always wrong -> gap = 1.0
    conf = [1.0, 1.0, 1.0]
    corr = [0, 0, 0]
    assert compute_ece(conf, corr, n_bins=10) == pytest.approx(1.0)


def test_mce_reports_worst_bin():
    # bin @0.1: acc 0, gap 0.1 ; bin @0.9: acc 0, gap 0.9 -> MCE = 0.9
    conf = [0.1, 0.9, 0.9]
    corr = [0, 0, 0]
    assert compute_mce(conf, corr, n_bins=10) == pytest.approx(0.9)


# ---------------------------------------------------------------------------
# Brier
# ---------------------------------------------------------------------------


def test_brier_hand_computed():
    # ((0.1-0)^2 *2 + (0.9-1)^2 *2)/4 = (0.02 + 0.02)/4 = 0.01
    conf = [0.1, 0.1, 0.9, 0.9]
    corr = [0, 0, 1, 1]
    assert compute_brier(conf, corr) == pytest.approx(0.01)


def test_brier_perfect_is_zero():
    assert compute_brier([1.0, 0.0], [1, 0]) == pytest.approx(0.0)


def test_brier_worst_is_one():
    assert compute_brier([1.0, 0.0], [0, 1]) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# AUROC
# ---------------------------------------------------------------------------


def test_auroc_perfect_separation():
    # all correct answers held higher confidence than all wrong ones
    assert compute_auroc([0.1, 0.1, 0.9, 0.9], [0, 0, 1, 1]) == pytest.approx(1.0)


def test_auroc_perfectly_wrong():
    assert compute_auroc([0.9, 0.1], [0, 1]) == pytest.approx(0.0)


def test_auroc_ties_give_half():
    # single positive and single negative sharing the same score -> 0.5
    assert compute_auroc([0.5, 0.5], [1, 0]) == pytest.approx(0.5)


def test_auroc_mixed_ties():
    # scores: pos={0.8,0.5}, neg={0.8,0.2}
    # pairs (pos,neg): (0.8,0.8)->0.5 tie, (0.8,0.2)->1, (0.5,0.8)->0, (0.5,0.2)->1
    # mean = (0.5+1+0+1)/4 = 0.625
    conf = [0.8, 0.5, 0.8, 0.2]
    corr = [1, 1, 0, 0]
    assert compute_auroc(conf, corr) == pytest.approx(0.625)


def test_auroc_undefined_single_class_is_nan():
    assert math.isnan(compute_auroc([0.9, 0.8], [1, 1]))
    assert math.isnan(compute_auroc([0.9, 0.8], [0, 0]))


def test_auroc_matches_brute_force_random():
    rng = np.random.default_rng(0)
    conf = rng.random(200)
    corr = (rng.random(200) < conf).astype(int)  # better answers -> higher conf
    # brute-force Mann-Whitney with 0.5 for ties
    pos = conf[corr == 1]
    neg = conf[corr == 0]
    wins = 0.0
    for p in pos:
        wins += np.sum(p > neg) + 0.5 * np.sum(p == neg)
    expected = wins / (len(pos) * len(neg))
    assert compute_auroc(conf, corr) == pytest.approx(expected)


# ---------------------------------------------------------------------------
# Confident-error rate
# ---------------------------------------------------------------------------


def test_confident_error_rate_hand_computed():
    # threshold 0.8: confident-and-wrong at idx 0 (0.9,wrong) and idx 3 (0.85,wrong)
    conf = [0.9, 0.9, 0.1, 0.85]
    corr = [0, 1, 0, 0]
    assert confident_error_rate(conf, corr, threshold=0.8) == pytest.approx(2 / 4)


def test_confident_error_rate_threshold_is_inclusive():
    assert confident_error_rate([0.8], [0], threshold=0.8) == pytest.approx(1.0)


def test_confident_error_rate_none_when_all_correct():
    assert confident_error_rate([0.9, 0.95], [1, 1], threshold=0.8) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Reliability curve
# ---------------------------------------------------------------------------


def test_reliability_curve_bins_and_gap():
    bins = reliability_curve([0.1, 0.1, 0.9, 0.9], [0, 0, 1, 1], n_bins=10)
    assert len(bins) == 2
    low, high = bins
    assert low.count == 2 and high.count == 2
    assert low.avg_confidence == pytest.approx(0.1)
    assert low.accuracy == pytest.approx(0.0)
    assert low.gap == pytest.approx(0.1)
    assert high.accuracy == pytest.approx(1.0)
    assert high.gap == pytest.approx(-0.1)


def test_reliability_curve_confidence_one_lands_in_last_bin():
    bins = reliability_curve([1.0], [1], n_bins=10)
    assert len(bins) == 1
    assert bins[0].lo == pytest.approx(0.9)
    assert bins[0].hi == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# evaluate() bundle
# ---------------------------------------------------------------------------


def test_evaluate_bundles_everything():
    conf = [0.1, 0.1, 0.9, 0.9]
    corr = [0, 0, 1, 1]
    rep = evaluate(conf, corr, n_bins=10, confident_threshold=0.8)
    assert rep.n == 4
    assert rep.accuracy == pytest.approx(0.5)
    assert rep.ece == pytest.approx(0.1)
    assert rep.brier == pytest.approx(0.01)
    assert rep.auroc == pytest.approx(1.0)
    assert rep.mean_confidence == pytest.approx(0.5)
    d = rep.as_dict()
    assert d["overconfidence"] == pytest.approx(0.0)
    assert set(d) >= {"ece", "brier", "auroc", "confident_error_rate"}


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def test_mismatched_shapes_raise():
    with pytest.raises(ValueError):
        compute_ece([0.1, 0.2], [1])


def test_out_of_range_confidence_raises():
    with pytest.raises(ValueError):
        compute_brier([1.5], [1])


def test_non_binary_correct_raises():
    with pytest.raises(ValueError):
        compute_brier([0.5], [2])


def test_empty_input_raises():
    with pytest.raises(ValueError):
        compute_ece([], [])


# ---------------------------------------------------------------------------
# Confidence parsing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text,answer,conf",
    [
        ("Answer: B\nConfidence: 85%", "B", 0.85),
        ("Answer: Paris\nConfidence: 0.9", "Paris", 0.9),
        ("Answer: 12\nConfidence: 100", "12", 1.0),
        ("answer=  Canberra  confidence = 70", "Canberra", 0.70),
        ("The capital is Rome. Confidence: 40%", "The capital is Rome", 0.40),
    ],
)
def test_parse_answer_and_confidence(text, answer, conf):
    got_answer, got_conf = parse_answer_and_confidence(text)
    assert got_answer == answer
    assert got_conf == pytest.approx(conf)


def test_parse_missing_confidence_returns_none():
    answer, conf = parse_answer_and_confidence("Answer: B")
    assert answer == "B"
    assert conf is None


def test_parse_confidence_clamped():
    _, conf = parse_answer_and_confidence("Answer: x\nConfidence: 250")
    assert conf == pytest.approx(1.0)
