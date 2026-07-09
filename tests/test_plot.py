"""Tests for the text table and reliability diagram rendering (no display)."""

from __future__ import annotations

from honest_quant.eval import evaluate
from honest_quant.plot import (
    render_summary_table,
    save_reliability_diagram,
    save_summary_diagram,
)


def test_render_summary_table_contains_metrics():
    rep = evaluate([0.1, 0.1, 0.9, 0.9], [0, 0, 1, 1])
    table = render_summary_table({"q4_k_m": rep.as_dict()})
    assert "quant" in table
    assert "ECE" in table
    assert "q4_k_m" in table
    # header + separator + one row
    assert len(table.splitlines()) == 3


def test_render_summary_table_handles_nan():
    rep = evaluate([0.9, 0.8], [1, 1])  # single class -> AUROC nan
    table = render_summary_table({"fp16": rep.as_dict()})
    assert "nan" in table


def test_save_reliability_diagram(tmp_path):
    rep = evaluate([0.1, 0.1, 0.9, 0.9], [0, 0, 1, 1])
    out = save_reliability_diagram(rep, tmp_path / "rel.png")
    assert out.exists()
    assert out.stat().st_size > 0


def test_save_summary_diagram(tmp_path):
    reports = {
        "q4_k_m": evaluate([0.1, 0.9], [0, 1]),
        "q8_0": evaluate([0.2, 0.8], [0, 1]),
        "fp16": evaluate([0.3, 0.7], [0, 1]),
    }
    out = save_summary_diagram(reports, tmp_path / "grid.png")
    assert out.exists()
    assert out.stat().st_size > 0
