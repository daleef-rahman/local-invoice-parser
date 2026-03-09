"""Shared helpers for model backends."""

from __future__ import annotations

import json
import re
import subprocess
import time
import urllib.parse
import urllib.request
from collections.abc import Callable


def server_healthy(base_url: str) -> bool:
    parsed = urllib.parse.urlparse(base_url)
    health_url = f"{parsed.scheme}://{parsed.netloc}/health"
    try:
        with urllib.request.urlopen(health_url, timeout=2) as response:
            return response.status == 200
    except Exception:
        return False


def ensure_llama_server(
    base_url: str,
    *,
    default_port: int,
    model_args: list[str],
    timeout: int = 120,
) -> None:
    if server_healthy(base_url):
        return

    port = urllib.parse.urlparse(base_url).port or default_port
    print(f"llama-server not running at {base_url} - starting on port {port}...")
    subprocess.Popen(
        ["llama-server", *model_args, "--port", str(port)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if server_healthy(base_url):
            print("llama-server ready.")
            return
        time.sleep(2)

    raise RuntimeError(f"llama-server did not become ready within {timeout}s")


def extract_json_object(text: str) -> str:
    fence_match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if fence_match:
        text = fence_match.group(1)

    start = text.find("{")
    if start == -1:
        raise json.JSONDecodeError("No JSON object start found", text, 0)

    in_string = False
    escape = False
    depth = 0
    for idx, char in enumerate(text[start:], start=start):
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == "\"":
                in_string = False
            continue

        if char == "\"":
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start:idx + 1]

    raise json.JSONDecodeError("No complete JSON object found", text, start)


def parse_json_with_retries(
    request_fn: Callable[[str], str],
    prompts: list[str],
    *,
    error_prefix: str,
) -> dict:
    last_error: Exception | None = None

    for prompt in prompts:
        content = request_fn(prompt)
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            try:
                return json.loads(extract_json_object(content))
            except json.JSONDecodeError as exc:
                last_error = exc

    raise ValueError(f"{error_prefix}: {last_error}")
