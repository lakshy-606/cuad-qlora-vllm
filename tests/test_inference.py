import pytest

from cuad_llm import inference
from cuad_llm.inference import Completion, ModelConfig, run_chat


def test_run_chat_keeps_order_and_resumes_from_cache(tmp_path, monkeypatch):
    calls = []

    async def fake_complete(client, cfg, messages, sem):
        calls.append(messages[0]["content"])
        return Completion(messages[0]["content"].upper(), 10, 2, 0, 0.01)

    monkeypatch.setattr(inference, "_complete", fake_complete)
    cfg = ModelConfig(model="m", base_url="http://unused")
    reqs = [[{"role": "user", "content": c}] for c in "abc"]
    cache = tmp_path / "responses.jsonl"

    assert [c.text for c in run_chat(reqs, cfg, cache)] == ["A", "B", "C"]
    assert len(calls) == 3
    more = reqs + [[{"role": "user", "content": "d"}]]
    assert [c.text for c in run_chat(more, cfg, cache)] == ["A", "B", "C", "D"]
    assert len(calls) == 4, "cached requests must not be re-sent"


def test_failed_request_is_retried_without_losing_the_rest(tmp_path, monkeypatch):
    attempts = {}

    async def flaky(client, cfg, messages, sem):
        c = messages[0]["content"]
        attempts[c] = attempts.get(c, 0) + 1
        if c == "b" and attempts[c] == 1:
            raise TimeoutError("simulated timeout")
        return Completion(c.upper(), 10, 2, 0, 0.01)

    monkeypatch.setattr(inference, "_complete", flaky)
    cfg = ModelConfig(model="m", base_url="http://unused")
    reqs = [[{"role": "user", "content": c}] for c in "abc"]
    assert [c.text for c in run_chat(reqs, cfg, tmp_path / "r.jsonl")] == ["A", "B", "C"]
    assert attempts == {"a": 1, "b": 2, "c": 1}


def test_persistent_failure_raises_but_keeps_completed_responses(tmp_path, monkeypatch):
    async def broken_b(client, cfg, messages, sem):
        c = messages[0]["content"]
        if c == "b":
            raise TimeoutError("always times out")
        return Completion(c.upper(), 10, 2, 0, 0.01)

    monkeypatch.setattr(inference, "_complete", broken_b)
    cfg = ModelConfig(model="m", base_url="http://unused")
    reqs = [[{"role": "user", "content": c}] for c in "abc"]
    cache = tmp_path / "r.jsonl"
    with pytest.raises(RuntimeError, match="1 requests still failing"):
        run_chat(reqs, cfg, cache)
    assert len(cache.read_text().splitlines()) == 2  # a and c are cached for the rerun
