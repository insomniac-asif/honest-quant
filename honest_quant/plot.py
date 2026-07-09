"""Rendering: a plain-text summary table and reliability-diagram images.

``render_summary_table`` is pure text (no matplotlib) so it is easy to test and
prints at the end of a CLI run. The diagram helpers import matplotlib lazily
with the non-interactive ``Agg`` backend so they never require a display.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

from .eval import CalibrationReport, ReliabilityBin

__all__ = [
    "render_summary_table",
    "save_reliability_diagram",
    "save_summary_diagram",
]

_COLUMNS = [
    ("quant", "quant"),
    ("n", "n"),
    ("accuracy", "acc"),
    ("mean_confidence", "conf"),
    ("ece", "ECE"),
    ("brier", "brier"),
    ("auroc", "AUROC"),
    ("confident_error_rate", "conf-wrong"),
]


def _fmt(value: object) -> str:
    if isinstance(value, float):
        if value != value:  # NaN
            return "nan"
        return f"{value:.3f}"
    return str(value)


def render_summary_table(summary: Mapping[str, Mapping[str, object]]) -> str:
    """Render a per-quant metrics table as fixed-width text.

    ``summary`` maps quant name -> a dict like ``CalibrationReport.as_dict()``.
    """
    rows: list[list[str]] = []
    header = [label for _, label in _COLUMNS]
    for quant, metrics in summary.items():
        row = []
        for key, _ in _COLUMNS:
            row.append(quant if key == "quant" else _fmt(metrics.get(key, "")))
        rows.append(row)

    widths = [len(h) for h in header]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def _line(cells: Sequence[str]) -> str:
        return "  ".join(c.ljust(widths[i]) for i, c in enumerate(cells))

    out = [_line(header), _line(["-" * w for w in widths])]
    out.extend(_line(r) for r in rows)
    return "\n".join(out)


def _reliability_axis(ax, bins: Sequence[ReliabilityBin], title: str) -> None:
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1, label="perfect")
    if bins:
        xs = [b.avg_confidence for b in bins]
        ys = [b.accuracy for b in bins]
        sizes = [max(20.0, 300.0 * b.count / sum(x.count for x in bins)) for b in bins]
        ax.plot(xs, ys, marker="o", linewidth=1.2, label="model")
        ax.scatter(xs, ys, s=sizes, alpha=0.4)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("confidence")
    ax.set_ylabel("accuracy")
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=8)
    ax.set_aspect("equal", adjustable="box")


def save_reliability_diagram(
    report: CalibrationReport, path: str | Path, title: str | None = None
) -> Path:
    """Write a single reliability diagram PNG for one report."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    label = title or f"reliability (ECE={report.ece:.3f})"
    _reliability_axis(ax, report.bins, label)
    fig.tight_layout()
    path = Path(path)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def save_summary_diagram(
    reports: Mapping[str, CalibrationReport], path: str | Path
) -> Path:
    """Write a grid of reliability diagrams, one per quant level."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    items = list(reports.items())
    n = len(items)
    cols = min(3, max(1, n))
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4.5 * cols, 4.5 * rows), squeeze=False)
    for idx, (quant, rep) in enumerate(items):
        ax = axes[idx // cols][idx % cols]
        _reliability_axis(
            ax, rep.bins, f"{quant}  ECE={rep.ece:.3f}  cw={rep.confident_error_rate:.3f}"
        )
    for idx in range(n, rows * cols):
        axes[idx // cols][idx % cols].axis("off")
    fig.tight_layout()
    path = Path(path)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path
