"""Question set loading and answer grading.

The bundled question set (``data/questions.jsonl``) is small and intentionally
mixed: multiple-choice questions (single letter gold answer) and short-answer
questions (with a list of accepted aliases). It is a *smoke* set for wiring the
harness end-to-end, not a benchmark - see the README's Limitations section.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from importlib import resources
from typing import Iterable, Iterator, Literal

__all__ = ["Question", "grade_answer", "load_questions", "iter_bundled_questions"]

QuestionType = Literal["mcq", "short"]


@dataclass(frozen=True)
class Question:
    """A single labelled question."""

    id: str
    type: QuestionType
    prompt: str
    answer: str
    # For MCQ: mapping of choice letter -> choice text. Empty for short answer.
    choices: dict[str, str]
    # Additional accepted strings for grading (aliases / synonyms).
    aliases: tuple[str, ...] = ()

    def render(self) -> str:
        """Render the question as a prompt block for the model."""
        if self.type == "mcq" and self.choices:
            lines = [self.prompt, ""]
            for letter in sorted(self.choices):
                lines.append(f"{letter}. {self.choices[letter]}")
            return "\n".join(lines)
        return self.prompt


def _normalise(text: str) -> str:
    """Lowercase, strip punctuation/articles, collapse whitespace."""
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    text = re.sub(r"\b(the|a|an)\b", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def grade_answer(question: Question, model_answer: str | None) -> bool:
    """Return True if ``model_answer`` matches the gold label for ``question``.

    MCQ grading extracts a leading choice letter (``B`` / ``B.`` / ``(B)`` /
    ``B) foo`` / the full choice text). Short-answer grading normalises both
    sides and accepts any alias. Grading is deliberately lenient about
    formatting but strict about content.
    """
    if not model_answer:
        return False

    if question.type == "mcq":
        gold = question.answer.strip().upper()
        # Try to pull a standalone choice letter from the front of the answer.
        m = re.match(r"\s*[\(\[]?\s*([A-Za-z])\s*[\)\].:,-]", model_answer)
        if not m:
            m = re.match(r"\s*([A-Za-z])\s*$", model_answer.strip())
        if m and m.group(1).upper() == gold:
            return True
        # Fall back to matching the choice *text*.
        gold_text = question.choices.get(gold)
        if gold_text and _normalise(gold_text) == _normalise(model_answer):
            return True
        # Or the model wrote "<letter> <text>".
        if gold_text and _normalise(model_answer).endswith(_normalise(gold_text)):
            lead = _normalise(model_answer)[: -len(_normalise(gold_text))].strip()
            if lead in ("", gold.lower()):
                return True
        return False

    # short answer
    candidates = {_normalise(question.answer)}
    candidates.update(_normalise(a) for a in question.aliases)
    got = _normalise(model_answer)
    if got in candidates:
        return True
    # allow the gold answer to appear as a whole-word substring
    for cand in candidates:
        if cand and re.search(rf"\b{re.escape(cand)}\b", got):
            return True
    return False


def _question_from_obj(obj: dict) -> Question:
    return Question(
        id=str(obj["id"]),
        type=obj["type"],
        prompt=obj["prompt"],
        answer=str(obj["answer"]),
        choices=dict(obj.get("choices", {})),
        aliases=tuple(obj.get("aliases", [])),
    )


def load_questions(lines: Iterable[str]) -> list[Question]:
    """Parse questions from an iterable of JSONL strings."""
    out: list[Question] = []
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        out.append(_question_from_obj(json.loads(line)))
    return out


def iter_bundled_questions() -> Iterator[Question]:
    """Yield the questions packaged with honest-quant."""
    data = resources.files("honest_quant").joinpath("data/questions.jsonl")
    text = data.read_text(encoding="utf-8")
    yield from load_questions(text.splitlines())
