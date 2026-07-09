"""Offline demo: no model, no GPU, no network.

Simulates three "quant levels" as answer generators with different
accuracy/over-confidence profiles, runs them through the *real* honest-quant
pipeline, and prints the calibration table + writes a reliability grid.

This exists so a stranger can see exactly what the harness produces before
they have ollama + a GPU set up. The numbers here are SIMULATED (drawn from a
toy over-confidence model), not measurements of any real model.

Run:
    python examples/demo_synthetic.py
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

# Allow running from a clean checkout without `pip install -e .`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from honest_quant.dataset import Question
from honest_quant.plot import render_summary_table, save_summary_diagram
from honest_quant.run import run_quant

# A tiny synthetic question set (id, correct-letter).
QUESTIONS = [
    Question(f"q{i}", "mcq", f"Question {i}?", "A",
             {"A": "right", "B": "w1", "C": "w2", "D": "w3"})
    for i in range(120)
]

# Each "quant" is (accuracy, base_confidence, overconfidence_gain).
# Lower-bit quants here are modelled as *less accurate but MORE confident* -
# the exact failure this tool is built to detect. These are illustrative.
PROFILES = {
    "q3_k_m": dict(accuracy=0.62, conf=0.86, seed=1),
    "q4_k_m": dict(accuracy=0.71, conf=0.82, seed=2),
    "q8_0":   dict(accuracy=0.78, conf=0.80, seed=3),
    "fp16":   dict(accuracy=0.80, conf=0.79, seed=4),
}


def make_generator(accuracy: float, conf: float, seed: int):
    rng = random.Random(seed)

    def generate(model, system, prompt):
        right = rng.random() < accuracy
        letter = "A" if right else rng.choice(["B", "C", "D"])
        # confidence jitters around the profile mean, clamped to [1, 99]
        c = int(max(1, min(99, rng.gauss(conf * 100, 8))))
        return f"Answer: {letter}\nConfidence: {c}"

    return generate


def main() -> None:
    summary = {}
    reports = {}
    for quant, p in PROFILES.items():
        gen = make_generator(p["accuracy"], p["conf"], p["seed"])
        run = run_quant(gen, model_tag=f"synthetic-{quant}", quant=quant,
                        questions=QUESTIONS)
        summary[quant] = run.report.as_dict()
        reports[quant] = run.report

    print("SIMULATED calibration across quant levels (not real measurements):\n")
    print(render_summary_table(summary))

    out = Path(__file__).resolve().parent.parent / "results" / "demo_reliability.png"
    out.parent.mkdir(exist_ok=True)
    save_summary_diagram(reports, out)
    print(f"\nReliability grid written to {out}")


if __name__ == "__main__":
    main()
