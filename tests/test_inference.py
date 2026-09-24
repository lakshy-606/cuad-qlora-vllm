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
