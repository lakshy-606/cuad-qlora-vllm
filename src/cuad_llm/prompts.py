"""Prompt format shared by the zero-shot baselines, fine-tuning data and fine-tuned model.

The model answers with a JSON array of verbatim quotes; `[]` means the clause is absent.
"""

from __future__ import annotations

import json
import re

SYSTEM_PROMPT = (
    "You are a contract review assistant. Given an excerpt of a commercial contract and a clause "
    "category, extract every passage in the excerpt that relates to that category. Copy passages "
    "verbatim from the excerpt: do not paraphrase, summarise or combine them. Answer with a JSON "
    "array of strings and nothing else. If the excerpt has no relevant passage, answer []."
)

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$")


def build_user_prompt(category: str, description: str, excerpt: str) -> str:
    # The excerpt comes before the category so all 41 prompts for one chunk share a long prefix,
    # which vLLM's prefix cache and OpenAI's prompt caching reuse instead of re-reading the chunk.
    return (
        f"Contract excerpt:\n<<<\n{excerpt}\n>>>\n\n"
        f"Clause category: {category}\n"
        f"Definition: {description}"
    )


def format_answer(quotes: list[str]) -> str:
    return json.dumps(quotes, ensure_ascii=False)


def build_messages(
    category: str, description: str, excerpt: str, answer: list[str] | None = None
) -> list[dict]:
    """Chat messages; include `answer` to get a supervised training example."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(category, description, excerpt)},
    ]
    if answer is not None:
        messages.append({"role": "assistant", "content": format_answer(answer)})
    return messages


def parse_answer(text: str) -> list[str] | None:
    """Parse a model response into quotes. Returns None if the response isn't a JSON string array."""
    text = _FENCE_RE.sub("", _THINK_RE.sub("", text).strip()).strip()
    if text.lower().rstrip(".") in {"", "none", "no related clause"}:
        return []
    start, end = text.find("["), text.rfind("]")
    if start == -1 or end < start:
        return None
    try:
        value = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    if not isinstance(value, list) or not all(isinstance(q, str) for q in value):
        return None
    return [q for q in value if q.strip()]
