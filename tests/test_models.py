from openai import OpenAI, APIStatusError
import argparse
import json
import os
import re

BASE_URL = "https://genai.atlas.intra.chrysler.com/v1"

def _extract_title(exc: APIStatusError) -> str | None:
    """Extract the `title` from the error response body (a raw string)."""
    body = exc.body
    if not isinstance(body, str):
        return None
    # Try JSON first, then fall back to HTML <title>.
    try:
        data = json.loads(body)
    except (json.JSONDecodeError, ValueError):
        match = re.search(r"<title>(.*?)</title>", body, re.IGNORECASE | re.DOTALL)
        return match.group(1).strip() if match else None
    if isinstance(data, dict):
        error = data.get("error")
        if isinstance(error, dict):
            title = error.get("title")
            return title if isinstance(title, str) else None
        title = data.get("title")
        return title if isinstance(title, str) else None
    return None


def test_size(client: OpenAI, word_count: int, model: str) -> bool:
    """
    Test whether a prompt of approximately `word_count` simple tokens is accepted.

    NOTE:
    `hello` repetitions are NOT guaranteed to map 1:1 to model tokens.
    The actual token count is determined by the server tokenizer.
    """
    prompt = "hello " * word_count

    from openai import APIStatusError

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1,
            temperature=0,
        )

        print(
            f"OK: "
            f"tokens={response.usage.prompt_tokens:,}, "
            f"bytes={len(prompt.encode('utf-8')):,}"
        )

    except APIStatusError as exc:
        title = _extract_title(exc) or exc.message
        print(
            f"FAIL: {title}, "
            f"requested tokens={word_count:,}, "
            f"bytes={len(prompt.encode('utf-8')):,}"
        )

    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test the maximum prompt size accepted by the model."
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("ATLAS_API_KEY", ""),
        help="API key (default: $ATLAS_API_KEY).",
    )
    parser.add_argument(
        "--base-url",
        default=os.environ.get("ATLAS_BASE_URL", BASE_URL),
        help="Base URL for the API (default: $ATLAS_BASE_URL or hardcoded fallback).",
    )
    parser.add_argument(
        "--model",
        default="atlas/glm-5.3-flash",
        help="Model to test (default: atlas/glm-5.3-flash).",
    )
    args = parser.parse_args()

    if not args.api_key:
        parser.error("API key required: set $ATLAS_API_KEY or pass --api-key")

    client = OpenAI(
        base_url=args.base_url,
        api_key=args.api_key,
    )

    # Start with progressively larger prompts.
    for size in [
        128_000,
        160_000,
        180_000,
        200_000,
        256_000,
    ]:
        if not test_size(client, size, args.model):
            break


if __name__ == "__main__":
    main()

