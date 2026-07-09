"""Tests for orchestration, with the ollama call mocked (no model/network)."""

from __future__ import annotations

import json

import pytest

from honest_quant.dataset import Question
from honest_quant.eval import CalibrationReport
from honest_quant.run import (
    OllamaClient,
    build_model_tag,
    run_family,
    run_quant,
)


def _questions() -> list[Question]:
    return [
        Question("q1", "mcq", "Capital of Australia?", "C",
                 {"A": "Sydney", "B": "Melbourne", "C": "Canberra", "D": "Perth"}),
        Question("q2", "short", "Red planet?", "Mars", {}),
        Question("q3", "mcq", "2+2?", "B", {"A": "3", "B": "4", "C": "5", "D": "6"}),
    ]


# ---------------------------------------------------------------------------
# Quant tag construction
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "family,quant,expected",
    [
        ("qwen2.5:7b", "q4_k_m", "qwen2.5:7b-q4_K_M"),
        ("qwen2.5:7b", "q8_0", "qwen2.5:7b-q8_0"),
        ("qwen2.5:7b", "fp16", "qwen2.5:7b-fp16"),
        ("qwen2.5:7b", "f16", "qwen2.5:7b-fp16"),  # alias
        ("qwen2.5:7b-instruct-q4_K_M", "q4_k_m", "qwen2.5:7b-instruct-q4_K_M"),  # baked in
        ("llama3.1:8b", "weirdtag", "llama3.1:8b-weirdtag"),  # passthrough
    ],
)
def test_build_model_tag(family, quant, expected):
    assert build_model_tag(family, quant) == expected


# ---------------------------------------------------------------------------
# run_quant with a mocked model
# ---------------------------------------------------------------------------


def _oracle_generate(model, system, prompt):
    """A fake model that always answers correctly with 90% confidence."""
    if "Capital of Australia" in prompt:
        return "Answer: C\nConfidence: 90"
    if "Red planet" in prompt:
        return "Answer: Mars\nConfidence: 90"
    if "2+2" in prompt:
        return "Answer: B\nConfidence: 90"
    return "Answer: ?\nConfidence: 90"


def test_run_quant_perfect_oracle():
    run = run_quant(_oracle_generate, "fake:tag", "q4_k_m", _questions())
    assert isinstance(run.report, CalibrationReport)
    assert run.report.accuracy == pytest.approx(1.0)
    assert run.n_unparsed == 0
    assert len(run.records) == 3
    assert all(r.correct for r in run.records)
    # perfectly confident and always right -> no confident errors
    assert run.report.confident_error_rate == pytest.approx(0.0)


def _confidently_wrong_generate(model, system, prompt):
    """Always wrong, always 95% sure -> the failure mode we hunt for."""
    return "Answer: definitely-not-right\nConfidence: 95"


def test_run_quant_confidently_wrong():
    run = run_quant(_confidently_wrong_generate, "fake:tag", "q3_k_m", _questions())
    assert run.report.accuracy == pytest.approx(0.0)
    # every answer is wrong and >= 0.8 confidence
    assert run.report.confident_error_rate == pytest.approx(1.0)


def test_run_quant_unparsed_confidence_uses_default():
    def _no_conf(model, system, prompt):
        return "Answer: C"  # no confidence line

    run = run_quant(_no_conf, "fake:tag", "q8_0", _questions(), default_confidence=0.5)
    assert run.n_unparsed == 3
    assert all(r.confidence == pytest.approx(0.5) for r in run.records)


def test_run_family_iterates_quants():
    results = run_family(
        _oracle_generate, "qwen2.5:7b", ["q4_k_m", "q8_0"], _questions()
    )
    assert set(results) == {"q4_k_m", "q8_0"}
    assert results["q4_k_m"].model_tag == "qwen2.5:7b-q4_K_M"
    assert results["q8_0"].report.accuracy == pytest.approx(1.0)


def test_quant_run_to_json_roundtrips():
    run = run_quant(_oracle_generate, "fake:tag", "q4_k_m", _questions())
    blob = json.dumps(run.to_json())
    loaded = json.loads(blob)
    assert loaded["model_tag"] == "fake:tag"
    assert loaded["report"]["accuracy"] == pytest.approx(1.0)
    assert len(loaded["records"]) == 3


# ---------------------------------------------------------------------------
# OllamaClient request shaping (urlopen mocked; still no network)
# ---------------------------------------------------------------------------


def test_ollama_client_builds_chat_request(monkeypatch):
    captured = {}

    class _FakeResp:
        def __init__(self, payload):
            self._payload = payload

        def read(self):
            return json.dumps(self._payload).encode("utf-8")

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def _fake_urlopen(req, timeout=None):
        captured["url"] = req.full_url
        captured["method"] = req.get_method()
        captured["body"] = json.loads(req.data.decode("utf-8"))
        captured["timeout"] = timeout
        return _FakeResp({"message": {"content": "Answer: C\nConfidence: 80"}})

    monkeypatch.setattr("urllib.request.urlopen", _fake_urlopen)

    client = OllamaClient(host="http://localhost:11434", temperature=0.0)
    out = client(model="qwen2.5:7b-q4_K_M", system="sys", prompt="hi")

    assert out == "Answer: C\nConfidence: 80"
    assert captured["url"] == "http://localhost:11434/api/chat"
    assert captured["method"] == "POST"
    body = captured["body"]
    assert body["model"] == "qwen2.5:7b-q4_K_M"
    assert body["stream"] is False
    assert body["options"]["temperature"] == 0.0
    assert body["messages"][0]["role"] == "system"
    assert body["messages"][1]["content"] == "hi"
