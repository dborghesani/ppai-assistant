from typing import Any


def usage_log_fields(usage: Any | None, context_window: int) -> dict[str, Any]:
    def value(name: str) -> Any | None:
        return usage.get(name) if isinstance(usage, dict) else getattr(usage, name, None)

    prompt_tokens = value("prompt_tokens")
    completion_tokens = value("completion_tokens")
    total_tokens = value("total_tokens")
    if (
        not isinstance(prompt_tokens, int)
        or not isinstance(completion_tokens, int)
        or prompt_tokens < 0
        or completion_tokens < 0
        or prompt_tokens + completion_tokens == 0
    ):
        prompt_tokens = completion_tokens = total_tokens = context_tokens = context_percent = None
    else:
        context_tokens = prompt_tokens + completion_tokens
        if not isinstance(total_tokens, int) or total_tokens <= 0:
            total_tokens = context_tokens
        context_percent = round(context_tokens / context_window * 100, 1) if context_window > 0 else None
    return {
        "available": prompt_tokens is not None,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "context_tokens": context_tokens,
        "context_window_tokens": context_window,
        "context_window_percent": context_percent,
    }