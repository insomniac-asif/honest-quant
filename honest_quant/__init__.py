"""honest-quant: measure how quantization affects a local model's calibration.

Public surface:

* :mod:`honest_quant.eval`    - pure calibration metrics (ECE, Brier, AUROC, ...)
* :mod:`honest_quant.dataset` - question loading + answer grading
* :mod:`honest_quant.run`     - orchestration + CLI over an ollama model
* :mod:`honest_quant.plot`    - summary table + reliability diagrams
"""

from __future__ import annotations

from .eval import (
    CalibrationReport,
    ReliabilityBin,
    compute_auroc,
    compute_brier,
    compute_ece,
    compute_mce,
    confident_error_rate,
    evaluate,
    parse_answer_and_confidence,
    reliability_curve,
)

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "CalibrationReport",
    "ReliabilityBin",
    "compute_auroc",
    "compute_brier",
    "compute_ece",
    "compute_mce",
    "confident_error_rate",
    "evaluate",
    "parse_answer_and_confidence",
    "reliability_curve",
]
