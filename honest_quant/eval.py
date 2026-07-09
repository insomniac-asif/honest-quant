"""Calibration metric core for honest-quant.

This module is deliberately *pure*: every function here operates on plain
arrays of ``(confidence, correct)`` pairs. There is no model, no network and no
GPU dependency, so the metrics can be unit-tested against hand-computed values.

Terminology
-----------
confidence : float in [0, 1]
    The model's self-reported probability that its own answer is correct.
correct : {0, 1} / bool
    Whether the answer actually matched the gold label.

The headline metrics:

* **ECE**  - Expected Calibration Error. How far the model's stated confidence
  drifts from its realised accuracy, averaged over confidence bins. 0 is
  perfect calibration.
* **Brier** - Mean squared error between confidence and correctness. A proper
  scoring rule; rewards both calibration *and* sharpness.
* **AUROC** - Area under the ROC curve using confidence as the score to
  separate correct from incorrect answers. Measures whether the model *knows
  when it knows* (discrimination), independent of absolute calibration.
* **Confident-error rate** - fraction of *all* answers that are both wrong and
  asserted with high confidence. This is the "confidently wrong" /
  hallucination proxy the whole harness exists to surface.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

__all__ = [
    "ReliabilityBin",
    "CalibrationReport",
    "compute_ece",
    "compute_mce",
    "compute_brier",
    "compute_auroc",
    "reliability_curve",
    "confident_error_rate",
    "evaluate",
    "parse_answer_and_confidence",
]


def _as_arrays(
    confidences: Sequence[float], correct: Sequence[float]
) -> tuple[np.ndarray, np.ndarray]:
    """Validate inputs and return them as float/float numpy arrays."""
    conf = np.asarray(confidences, dtype=float)
    corr = np.asarray(correct, dtype=float)
    if conf.shape != corr.shape:
        raise ValueError(
            f"confidences and correct must have the same shape, "
            f"got {conf.shape} and {corr.shape}"
        )
    if conf.ndim != 1:
        raise ValueError("confidences and correct must be 1-D")
    if conf.size == 0:
        raise ValueError("cannot compute calibration metrics on empty input")
    if np.any((conf < 0.0) | (conf > 1.0)):
        raise ValueError("confidences must lie in [0, 1]")
    uniq = set(np.unique(corr).tolist())
    if not uniq.issubset({0.0, 1.0}):
        raise ValueError("correct must be binary (0/1 or bool)")
    return conf, corr


@dataclass(frozen=True)
class ReliabilityBin:
    """One bin of a reliability diagram."""

    lo: float
    hi: float
    count: int
    avg_confidence: float
    accuracy: float

    @property
    def gap(self) -> float:
        """Signed calibration gap (confidence - accuracy) for this bin."""
        return self.avg_confidence - self.accuracy


@dataclass
class CalibrationReport:
    """Bundle of calibration metrics for a single (model, quant) run."""

    n: int
    accuracy: float
    ece: float
    mce: float
    brier: float
    auroc: float
    confident_error_rate: float
    confident_threshold: float
    mean_confidence: float
    bins: list[ReliabilityBin] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "n": self.n,
            "accuracy": self.accuracy,
            "ece": self.ece,
            "mce": self.mce,
            "brier": self.brier,
            "auroc": self.auroc,
            "confident_error_rate": self.confident_error_rate,
            "confident_threshold": self.confident_threshold,
            "mean_confidence": self.mean_confidence,
            "overconfidence": self.mean_confidence - self.accuracy,
        }


def reliability_curve(
    confidences: Sequence[float],
    correct: Sequence[float],
    n_bins: int = 10,
) -> list[ReliabilityBin]:
    """Bin predictions by confidence and return per-bin accuracy/confidence.

    Bins are equal-width over [0, 1]. Empty bins are omitted from the output
    but still contribute zero weight to ECE (i.e. they are simply absent).
    A confidence of exactly 1.0 lands in the final bin.
    """
    if n_bins < 1:
        raise ValueError("n_bins must be >= 1")
    conf, corr = _as_arrays(confidences, correct)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    # np.digitize with right=False puts x in bin i where edges[i-1] <= x < edges[i].
    idx = np.digitize(conf, edges[1:-1], right=False)  # values in [0, n_bins-1]

    bins: list[ReliabilityBin] = []
    for b in range(n_bins):
        mask = idx == b
        count = int(mask.sum())
        if count == 0:
            continue
        bins.append(
            ReliabilityBin(
                lo=float(edges[b]),
                hi=float(edges[b + 1]),
                count=count,
                avg_confidence=float(conf[mask].mean()),
                accuracy=float(corr[mask].mean()),
            )
        )
    return bins


def compute_ece(
    confidences: Sequence[float],
    correct: Sequence[float],
    n_bins: int = 10,
) -> float:
    """Expected Calibration Error (equal-width binning).

    ECE = sum_b (n_b / N) * |acc_b - conf_b|
    """
    conf, _ = _as_arrays(confidences, correct)
    n = conf.size
    bins = reliability_curve(confidences, correct, n_bins=n_bins)
    return float(sum((b.count / n) * abs(b.accuracy - b.avg_confidence) for b in bins))


def compute_mce(
    confidences: Sequence[float],
    correct: Sequence[float],
    n_bins: int = 10,
) -> float:
    """Maximum Calibration Error: the worst-case bin gap."""
    bins = reliability_curve(confidences, correct, n_bins=n_bins)
    if not bins:
        return 0.0
    return float(max(abs(b.accuracy - b.avg_confidence) for b in bins))


def compute_brier(
    confidences: Sequence[float],
    correct: Sequence[float],
) -> float:
    """Brier score: mean((confidence - correct)**2). Lower is better."""
    conf, corr = _as_arrays(confidences, correct)
    return float(np.mean((conf - corr) ** 2))


def compute_auroc(
    confidences: Sequence[float],
    correct: Sequence[float],
) -> float:
    """Area under the ROC curve, confidence as the discriminating score.

    Computed via the Mann-Whitney U statistic with average ranks so ties are
    handled correctly (a tie between a correct and incorrect example
    contributes 0.5). Returns ``nan`` if all answers are correct or all are
    incorrect (AUROC is undefined with only one class).
    """
    conf, corr = _as_arrays(confidences, correct)
    pos = conf[corr == 1.0]
    neg = conf[corr == 0.0]
    n_pos, n_neg = pos.size, neg.size
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    # Rank all scores together, using average ranks for ties.
    order = np.argsort(conf, kind="mergesort")
    ranks = np.empty(conf.size, dtype=float)
    sorted_conf = conf[order]
    i = 0
    n = conf.size
    while i < n:
        j = i
        while j + 1 < n and sorted_conf[j + 1] == sorted_conf[i]:
            j += 1
        # positions i..j (0-based) share the same value -> average rank (1-based)
        avg_rank = (i + j) / 2.0 + 1.0
        ranks[order[i : j + 1]] = avg_rank
        i = j + 1

    sum_ranks_pos = ranks[corr == 1.0].sum()
    u_pos = sum_ranks_pos - n_pos * (n_pos + 1) / 2.0
    return float(u_pos / (n_pos * n_neg))


def confident_error_rate(
    confidences: Sequence[float],
    correct: Sequence[float],
    threshold: float = 0.8,
) -> float:
    """Fraction of *all* answers that are wrong yet asserted at >= threshold.

    This is the "confidently wrong" rate - the number the harness is built to
    expose. It is computed over the whole set (not just the wrong answers), so
    it directly answers "how often does this model hand me a high-confidence
    lie?".
    """
    conf, corr = _as_arrays(confidences, correct)
    if not (0.0 <= threshold <= 1.0):
        raise ValueError("threshold must lie in [0, 1]")
    confident_wrong = (conf >= threshold) & (corr == 0.0)
    return float(confident_wrong.mean())


def evaluate(
    confidences: Sequence[float],
    correct: Sequence[float],
    n_bins: int = 10,
    confident_threshold: float = 0.8,
) -> CalibrationReport:
    """Compute the full calibration report for one run."""
    conf, corr = _as_arrays(confidences, correct)
    return CalibrationReport(
        n=int(conf.size),
        accuracy=float(corr.mean()),
        ece=compute_ece(conf, corr, n_bins=n_bins),
        mce=compute_mce(conf, corr, n_bins=n_bins),
        brier=compute_brier(conf, corr),
        auroc=compute_auroc(conf, corr),
        confident_error_rate=confident_error_rate(
            conf, corr, threshold=confident_threshold
        ),
        confident_threshold=confident_threshold,
        mean_confidence=float(conf.mean()),
        bins=reliability_curve(conf, corr, n_bins=n_bins),
    )


# ---------------------------------------------------------------------------
# Confidence elicitation parsing
# ---------------------------------------------------------------------------

import re

# Matches "Confidence: 85%", "confidence = 0.85", "CONFIDENCE 85", etc.
_CONF_RE = re.compile(
    r"confidence[^0-9]*?(?P<num>[0-9]*\.?[0-9]+)\s*(?P<pct>%)?",
    re.IGNORECASE,
)
_ANSWER_RE = re.compile(
    r"answer[^0-9A-Za-z]*[:\-]?\s*(?P<ans>.+?)(?:\r?\n|$)",
    re.IGNORECASE,
)


def _normalise_confidence(num: float, is_percent: bool) -> float:
    """Map a raw confidence number onto [0, 1]."""
    if is_percent or num > 1.0:
        num = num / 100.0
    return float(min(max(num, 0.0), 1.0))


def parse_answer_and_confidence(text: str) -> tuple[str | None, float | None]:
    """Extract ``(answer, confidence)`` from a model's free-text response.

    The elicitation prompt (see ``run.py``) asks the model to reply with two
    lines::

        Answer: B
        Confidence: 85%

    This parser is tolerant of formatting drift: it will find a ``Confidence``
    figure anywhere in the text, accept ``0.85`` or ``85`` or ``85%``, and fall
    back to the first non-empty line for the answer if no explicit
    ``Answer:`` marker is present. Returns ``(None, None)`` components that
    could not be found so the caller can decide how to score a malformed reply.
    """
    if text is None:
        return None, None

    confidence: float | None = None
    m = _CONF_RE.search(text)
    if m:
        confidence = _normalise_confidence(
            float(m.group("num")), m.group("pct") is not None
        )

    answer: str | None = None
    am = _ANSWER_RE.search(text)
    if am:
        answer = am.group("ans").strip()
    else:
        for line in text.splitlines():
            line = line.strip()
            if line:
                answer = line
                break
    if answer is not None:
        # Trim a trailing confidence clause if it leaked onto the answer line.
        answer = re.split(r"\bconfidence\b", answer, flags=re.IGNORECASE)[0].strip()
        answer = answer.strip().strip(".").strip() or None

    return answer, confidence
