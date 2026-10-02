from __future__ import annotations

import json
import urllib.request

from graphrag.config import Settings, get_settings


def maybe_generate(prompt: str, settings: Settings | None = None) -> str | None:
    settings = settings or get_settings()
    if not settings.use_llm:
        return None
    payload = json.dumps(
        {
            "model": settings.openai_model,
            "messages": [
                {
                    "role": "system",
                    "content": "Answer using only the provided graph paths and passages.",
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.1,
        }
    ).encode()
    req = urllib.request.Request(
        f"{settings.openai_base_url.rstrip('/')}/chat/completions",
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {settings.openai_api_key}",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        body = json.loads(resp.read().decode())
    return body["choices"][0]["message"]["content"]
