"""Orchestration + CLI: run a labelled question set through an ollama model at
several quantization levels and collect calibration metrics per level.

The only part of this file that touches the network is :class:`OllamaClient`.
Everything else is pure and unit-tested by mocking that one seam.

Example
-------
::

    python -m honest_quant.run --family qwen2.5:7b --quants q4_k_m,q8_0,fp16 --n 200

That requires a GPU + a running ollama with those tags pulled. With no model
available you can still exercise the whole pipeline against a fake client - see
``tests/test_run.py``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Protocol, Sequence

from .dataset import Question, grade_answer, iter_bundled_questions, load_questions
from .eval import CalibrationReport, evaluate, parse_answer_and_confidence

__all__ = [
    "OllamaClient",
    "GenerateFn",
    "AnswerRecord",
    "QuantRun",
    "build_model_tag",
    "ELICITATION_SYSTEM",
    "ELICITATION_TEMPLATE",
    "run_quant",
    "run_family",
]

# ---------------------------------------------------------------------------
# Prompting
# ---------------------------------------------------------------------------

ELICITATION_SYSTEM = (
    "You are a careful exam-taker. Answer the question, then honestly rate how "
    "likely your answer is correct. Do not inflate your confidence. Reply in "
    "exactly two lines:\n"
    "Answer: <your answer>\n"
    "Confidence: <an integer 0-100, where 100 means certain>"
)

ELICITATION_TEMPLATE = (
    "{question}\n\n"
    "Give your best answer and your honest confidence (0-100) that it is correct."
)


def build_prompt(question: Question) -> str:
    return ELICITATION_TEMPLATE.format(question=question.render())


# ---------------------------------------------------------------------------
# Quant tag construction
# ---------------------------------------------------------------------------

# Canonical ollama quant suffixes keyed by a lowercased, user-friendly name.
_QUANT_SUFFIX = {
    "q2_k": "q2_K",
    "q3_k_s": "q3_K_S",
    "q3_k_m": "q3_K_M",
    "q3_k_l": "q3_K_L",
    "q4_0": "q4_0",
    "q4_1": "q4_1",
    "q4_k_s": "q4_K_S",
    "q4_k_m": "q4_K_M",
    "q5_0": "q5_0",
    "q5_k_s": "q5_K_S",
    "q5_k_m": "q5_K_M",
    "q6_k": "q6_K",
    "q8_0": "q8_0",
    "fp16": "fp16",
    "f16": "fp16",
}


def build_model_tag(family: str, quant: str) -> str:
    """Compose an ollama model tag from a family and a quant level.

    * If ``quant`` already looks baked into ``family`` (the family string ends
      with the quant), the family is returned unchanged.
    * ``family`` may be a bare name (``qwen2.5:7b``) or already carry an
      ``-instruct`` style variant; the quant suffix is appended after a dash.

    The mapping is a convenience, not a guarantee that a given tag exists in
    the ollama registry - pull/verify tags yourself. Unknown quant names are
    passed through verbatim so you can use any tag ollama accepts.
    """
    q = _QUANT_SUFFIX.get(quant.strip().lower(), quant.strip())
    fam = family.strip()
    if fam.lower().endswith(q.lower()):
        return fam
    return f"{fam}-{q}"


# ---------------------------------------------------------------------------
# Model client (the single network seam)
# ---------------------------------------------------------------------------


class GenerateFn(Protocol):
    """A callable that maps (model, system, prompt) -> raw completion text."""

    def __call__(self, model: str, system: str, prompt: str) -> str: ...


@dataclass
class OllamaClient:
    """Minimal ollama HTTP client using only the standard library.

    Talks to the ``/api/chat`` endpoint of a locally running ollama server.
    No third-party HTTP dependency; nothing here runs unless you call it.
    """

    host: str = "http://localhost:11434"
    timeout: float = 120.0
    temperature: float = 0.0
    retries: int = 3
    backoff: float = 1.5

    def __call__(self, model: str, system: str, prompt: str) -> str:
        payload = {
            "model": model,
            "stream": False,
            "options": {"temperature": self.temperature},
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self.host}/api/chat",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        # Retry transient failures (e.g. an ollama 500 while a model cold-loads or
        # is evicted between calls) with a short backoff, so one blip does not
        # abort a whole quant level. Persistent failures still propagate.
        last_exc: Exception | None = None
        for attempt in range(self.retries):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    body = json.loads(resp.read().decode("utf-8"))
                return body.get("message", {}).get("content", "")
            except (urllib.error.HTTPError, urllib.error.URLError, ConnectionError) as exc:
                last_exc = exc
                if attempt < self.retries - 1:
                    time.sleep(self.backoff * (attempt + 1))
        raise last_exc if last_exc is not None else RuntimeError("ollama call failed")


# ---------------------------------------------------------------------------
# Collection
# ---------------------------------------------------------------------------


@dataclass
class AnswerRecord:
    """One graded model answer."""

    question_id: str
    parsed_answer: str | None
    confidence: float | None
    correct: bool
    raw: str


@dataclass
class QuantRun:
    """Everything collected for a single quant level."""

    model_tag: str
    quant: str
    records: list[AnswerRecord] = field(default_factory=list)
    report: CalibrationReport | None = None
    # Answers where confidence could not be parsed (scored at default_confidence).
    n_unparsed: int = 0

    def to_json(self) -> dict:
        return {
            "model_tag": self.model_tag,
            "quant": self.quant,
            "n_unparsed": self.n_unparsed,
            "report": self.report.as_dict() if self.report else None,
            "records": [asdict(r) for r in self.records],
        }


def run_quant(
    generate: GenerateFn,
    model_tag: str,
    quant: str,
    questions: Sequence[Question],
    default_confidence: float = 0.5,
    n_bins: int = 10,
    confident_threshold: float = 0.8,
    on_progress: Callable[[int, int], None] | None = None,
) -> QuantRun:
    """Ask every question, grade the answers, and compute a calibration report.

    ``generate`` is the model seam; pass an :class:`OllamaClient` for real runs
    or any callable for tests. Unparseable confidences fall back to
    ``default_confidence`` (and are counted) so a few malformed replies do not
    abort the run.
    """
    run = QuantRun(model_tag=model_tag, quant=quant)
    confidences: list[float] = []
    correct: list[int] = []

    for i, q in enumerate(questions):
        raw = generate(model=model_tag, system=ELICITATION_SYSTEM, prompt=build_prompt(q))
        answer, conf = parse_answer_and_confidence(raw)
        if conf is None:
            conf = default_confidence
            run.n_unparsed += 1
        is_correct = grade_answer(q, answer)
        run.records.append(
            AnswerRecord(
                question_id=q.id,
                parsed_answer=answer,
                confidence=conf,
                correct=is_correct,
                raw=raw,
            )
        )
        confidences.append(conf)
        correct.append(1 if is_correct else 0)
        if on_progress is not None:
            on_progress(i + 1, len(questions))

    run.report = evaluate(
        confidences,
        correct,
        n_bins=n_bins,
        confident_threshold=confident_threshold,
    )
    return run


def run_family(
    generate: GenerateFn,
    family: str,
    quants: Sequence[str],
    questions: Sequence[Question],
    **kwargs,
) -> dict[str, QuantRun]:
    """Run the same question set at each quant level for one model family."""
    results: dict[str, QuantRun] = {}
    for quant in quants:
        tag = build_model_tag(family, quant)
        results[quant] = run_quant(
            generate, model_tag=tag, quant=quant, questions=questions, **kwargs
        )
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _load_question_set(path: str | None, n: int | None) -> list[Question]:
    if path:
        with open(path, "r", encoding="utf-8") as fh:
            questions = load_questions(fh.readlines())
    else:
        questions = list(iter_bundled_questions())
    if n is not None and n < len(questions):
        questions = questions[:n]
    return questions


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m honest_quant.run",
        description="Measure how quantization affects a local model's calibration.",
    )
    p.add_argument(
        "--family",
        required=True,
        help="Base ollama model, e.g. 'qwen2.5:7b'. Quant suffixes are appended.",
    )
    p.add_argument(
        "--quants",
        default="q4_k_m,q8_0,fp16",
        help="Comma-separated quant levels (default: q4_k_m,q8_0,fp16).",
    )
    p.add_argument(
        "--n",
        type=int,
        default=None,
        help="Cap the number of questions (default: use the whole set).",
    )
    p.add_argument(
        "--questions",
        default=None,
        help="Path to a custom JSONL question file (default: bundled set).",
    )
    p.add_argument("--host", default="http://localhost:11434", help="ollama host URL.")
    p.add_argument(
        "--temperature", type=float, default=0.0, help="Sampling temperature."
    )
    p.add_argument(
        "--bins", type=int, default=10, help="Reliability bins for ECE (default 10)."
    )
    p.add_argument(
        "--confident-threshold",
        type=float,
        default=0.8,
        help="Confidence at/above which a wrong answer counts as confidently wrong.",
    )
    p.add_argument(
        "--out",
        default="results",
        help="Directory to write per-run JSON + summary (default: results/).",
    )
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    quants = [q.strip() for q in args.quants.split(",") if q.strip()]
    questions = _load_question_set(args.questions, args.n)

    client = OllamaClient(host=args.host, temperature=args.temperature)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"honest-quant: {args.family} over {quants} on {len(questions)} questions")
    summary: dict[str, dict] = {}

    for quant in quants:
        tag = build_model_tag(args.family, quant)
        print(f"\n>> {quant}  (tag: {tag})")
        started = time.time()

        def _progress(done: int, total: int, _q=quant) -> None:
            print(f"\r   {done}/{total}", end="", flush=True)

        try:
            run = run_quant(
                client,
                model_tag=tag,
                quant=quant,
                questions=questions,
                n_bins=args.bins,
                confident_threshold=args.confident_threshold,
                on_progress=_progress,
            )
        except (urllib.error.URLError, urllib.error.HTTPError, ConnectionError) as exc:
            print(f"\n   ! failed to reach ollama at {args.host}: {exc}")
            print("   Is ollama running and is the tag pulled? Skipping this quant.")
            continue

        elapsed = time.time() - started
        (out_dir / f"{quant}.json").write_text(
            json.dumps(run.to_json(), indent=2), encoding="utf-8"
        )
        rep = run.report
        assert rep is not None
        summary[quant] = {**rep.as_dict(), "elapsed_s": round(elapsed, 1)}
        print(
            f"\n   acc={rep.accuracy:.3f}  ece={rep.ece:.3f}  "
            f"brier={rep.brier:.3f}  auroc={rep.auroc:.3f}  "
            f"conf-wrong={rep.confident_error_rate:.3f}"
        )

    if summary:
        (out_dir / "summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        print(f"\nWrote {out_dir/'summary.json'}")
        try:
            from .plot import render_summary_table

            print("\n" + render_summary_table(summary))
        except Exception:  # pragma: no cover - plotting is optional at CLI time
            pass
        return 0

    print("\nNo results collected (no reachable model). Nothing written.")
    return 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
