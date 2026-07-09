# Contributing to honest-quant

Thanks! `honest-quant` measures how quantization affects a model's calibration. The
metric core is pure and unit-tested; the model runner is the only part that touches the
network.

## Dev setup

```bash
git clone https://github.com/insomniac-asif/honest-quant
cd honest-quant
pip install -e .
pip install pytest
python -m pytest -q      # metric core is tested on synthetic data — no model/GPU needed
```

## Where to contribute

- **Metrics** — add a calibration metric to `eval.py`. It must be pure and come with a
  test that checks it against a value you computed by hand (see `tests/test_eval.py`).
- **Confidence elicitation** — verbalized confidence is one method. A token-logprob
  extractor (where the backend exposes logprobs) is a natural addition.
- **Question sets** — contribute a `--questions` JSONL (schema in `dataset.py`). ~200+
  labeled items is a reasonable floor for stable ECE/AUROC.
- **Runs** — ran it on a different model family? A results PR is welcome — include the
  exact command and `n` so it's reproducible.

## The bar

- The metric core stays pure and hand-verified. No metric ships without a test.
- **Never commit fabricated numbers.** Every reported result must be reproducible from
  the CLI — the whole point of this project is measuring instead of guessing.
