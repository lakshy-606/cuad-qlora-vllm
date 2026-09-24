"""Chat completions against any OpenAI-compatible endpoint: the OpenAI API or a vLLM server.

Responses are cached on disk keyed by (model, messages, sampling params), so an interrupted run resumes
where it stopped and re-scoring never re-pays for generation.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from openai import AsyncOpenAI


@dataclass
class ModelConfig:
    model: str
    base_url: str | None = None  # None means the OpenAI API
    api_key_env: str = "OPENAI_API_KEY"
    temperature: float = 0.0
    max_tokens: int = 1024
    concurrency: int = 32
    # Passed through to the server, e.g. {"chat_template_kwargs": {"enable_thinking": false}} for
    # Qwen3 hybrid-thinking models on vLLM.
    extra_body: dict = field(default_factory=dict)


@dataclass
class Completion:
    text: str
    prompt_tokens: int
    completion_tokens: int
    cached_tokens: int
    latency_s: float


def _cache_key(cfg: ModelConfig, messages: list[dict]) -> str:
    payload = {
        "model": cfg.model,
        "messages": messages,
        "temperature": cfg.temperature,
        "max_tokens": cfg.max_tokens,
        "extra_body": cfg.extra_body,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _load_cache(path: Path) -> dict[str, Completion]:
    cache: dict[str, Completion] = {}
    if path.exists():
        with path.open() as f:
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    key = row.pop("key")
                    cache[key] = Completion(**row)
    return cache


async def _complete(
    client: AsyncOpenAI, cfg: ModelConfig, messages: list[dict], sem: asyncio.Semaphore
) -> Completion:
    async with sem:
        t0 = time.perf_counter()
        resp = await client.chat.completions.create(
            model=cfg.model,
            messages=messages,
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
            extra_body=cfg.extra_body or None,
        )
        latency = time.perf_counter() - t0
    usage = resp.usage
    details = getattr(usage, "prompt_tokens_details", None)
    return Completion(
        text=resp.choices[0].message.content or "",
        prompt_tokens=usage.prompt_tokens if usage else 0,
        completion_tokens=usage.completion_tokens if usage else 0,
        cached_tokens=(getattr(details, "cached_tokens", None) or 0) if details else 0,
        latency_s=latency,
    )


async def _run(requests: list[list[dict]], cfg: ModelConfig, cache_path: Path) -> list[Completion]:
    cache = _load_cache(cache_path)
    keys = [_cache_key(cfg, m) for m in requests]
    todo = sorted({k: i for i, k in enumerate(keys) if k not in cache}.values())
    print(f"{len(requests)} requests, {len(requests) - len(todo)} cached, {len(todo)} to run")

    if todo:
        # A local vLLM server accepts any key; the OpenAI client just needs one set.
        api_key = os.environ.get(cfg.api_key_env) or ("EMPTY" if cfg.base_url else None)
        client = AsyncOpenAI(base_url=cfg.base_url, api_key=api_key, max_retries=6, timeout=600)
        sem = asyncio.Semaphore(cfg.concurrency)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        t0 = time.perf_counter()
        with cache_path.open("a") as f:
            # Tasks are created in request order, and requests for one chunk are adjacent, so
            # in-flight requests tend to share a prefix the server already has cached.
            async def indexed(i: int) -> tuple[int, Completion]:
                return i, await _complete(client, cfg, requests[i], sem)

            tasks = [asyncio.ensure_future(indexed(i)) for i in todo]
            for n, fut in enumerate(asyncio.as_completed(tasks), 1):
                i, done = await fut
                cache[keys[i]] = done
                f.write(json.dumps({"key": keys[i], **asdict(done)}) + "\n")
                if n % 500 == 0 or n == len(todo):
                    f.flush()
                    rate = n / (time.perf_counter() - t0)
                    print(f"  {n}/{len(todo)} done ({rate:.1f} req/s)")
        await client.close()

    return [cache[k] for k in keys]


def run_chat(requests: list[list[dict]], cfg: ModelConfig, cache_path: Path) -> list[Completion]:
    """Complete every request (a list of chat messages), in order, reusing cached responses."""
    return asyncio.run(_run(requests, cfg, Path(cache_path)))
