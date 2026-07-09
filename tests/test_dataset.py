"""Tests for question loading and answer grading (no model/network)."""

from __future__ import annotations

import pytest

from honest_quant.dataset import (
    Question,
    grade_answer,
    iter_bundled_questions,
    load_questions,
)


def _mcq() -> Question:
    return Question(
        id="t",
        type="mcq",
        prompt="Capital of Australia?",
        answer="C",
        choices={"A": "Sydney", "B": "Melbourne", "C": "Canberra", "D": "Perth"},
    )


@pytest.mark.parametrize(
    "model_answer,expected",
    [
        ("C", True),
        ("C.", True),
        ("(C)", True),
        ("C) Canberra", True),
        ("Canberra", True),
        ("c", True),
        ("B", False),
        ("Sydney", False),
        ("", False),
        (None, False),
    ],
)
def test_grade_mcq(model_answer, expected):
    assert grade_answer(_mcq(), model_answer) is expected


def _short() -> Question:
    return Question(
        id="s",
        type="short",
        prompt="Gas absorbed in photosynthesis?",
        answer="carbon dioxide",
        choices={},
        aliases=("CO2", "carbon-dioxide"),
    )


@pytest.mark.parametrize(
    "model_answer,expected",
    [
        ("carbon dioxide", True),
        ("Carbon Dioxide.", True),
        ("CO2", True),
        ("the carbon dioxide gas", True),  # whole-word substring + article strip
        ("oxygen", False),
        ("", False),
    ],
)
def test_grade_short(model_answer, expected):
    assert grade_answer(_short(), model_answer) is expected


def test_load_questions_skips_blanks_and_comments():
    lines = [
        "# a comment",
        "",
        '{"id":"x","type":"short","prompt":"p","answer":"a"}',
    ]
    qs = load_questions(lines)
    assert len(qs) == 1
    assert qs[0].id == "x"
    assert qs[0].aliases == ()


def test_bundled_questions_load_and_are_well_formed():
    qs = list(iter_bundled_questions())
    assert len(qs) >= 20
    ids = [q.id for q in qs]
    assert len(ids) == len(set(ids)), "question ids must be unique"
    for q in qs:
        assert q.type in ("mcq", "short")
        assert q.prompt
        assert q.answer
        if q.type == "mcq":
            assert q.answer in q.choices, f"{q.id}: gold letter not among choices"


def test_mcq_render_lists_choices():
    text = _mcq().render()
    assert "A. Sydney" in text
    assert "C. Canberra" in text
