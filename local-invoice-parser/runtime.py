"""Runtime lifecycle helpers for experiment execution."""

from __future__ import annotations

import json
import shutil
import subprocess
import time
import urllib.request
from contextlib import AbstractContextManager

from experiments.catalog import ExperimentSpec, HfAssetSpec, RuntimeSpec


def _health_url(port: int) -> str:
    return f"http://127.0.0.1:{port}/health"


def _models_url(port: int) -> str:
    return f"http://127.0.0.1:{port}/v1/models"


def _server_healthy(port: int) -> bool:
    try:
        with urllib.request.urlopen(_health_url(port), timeout=2) as response:
            return response.status == 200
    except Exception:
        return False


def _server_has_alias(port: int, alias: str) -> bool:
    try:
        with urllib.request.urlopen(_models_url(port), timeout=2) as response:
            payload = json.load(response)
    except Exception:
        return False

    for model in payload.get("data", []):
        if model.get("id") == alias:
            return True
    return False


def _wait_for_server(port: int, alias: str | None, timeout: int = 120) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if _server_healthy(port) and (alias is None or _server_has_alias(port, alias)):
            return
        time.sleep(1)
    raise RuntimeError(f"llama-server on port {port} did not become ready within {timeout}s")


def _ensure_hf_assets(asset_spec: HfAssetSpec) -> None:
    missing = [name for name in asset_spec.filenames if not (asset_spec.local_dir / name).exists()]
    if not missing:
        return

    asset_spec.local_dir.mkdir(parents=True, exist_ok=True)
    if not shutil.which("hf"):
        raise RuntimeError("Missing `hf` CLI. Install huggingface_hub or make `hf` available in PATH.")

    cmd = ["hf", "download", asset_spec.repo_id, *missing, "--local-dir", str(asset_spec.local_dir)]
    subprocess.run(cmd, check=True)


def _start_llamacpp_server(spec: RuntimeSpec) -> tuple[subprocess.Popen[bytes] | None, bool]:
    if spec.port is None or spec.model_path is None:
        raise ValueError("llamacpp_server runtime requires port and model_path")

    if spec.hf_assets is not None:
        _ensure_hf_assets(spec.hf_assets)

    alias = spec.model_alias
    if _server_healthy(spec.port):
        if alias is None or _server_has_alias(spec.port, alias):
            return None, False
        raise RuntimeError(
            f"Port {spec.port} is already serving a different model. "
            f"Stop the existing server or choose another port."
        )

    cmd = [
        "llama-server",
        "--model",
        str(spec.model_path),
        "--port",
        str(spec.port),
        "--ctx-size",
        str(spec.ctx_size),
    ]
    if spec.mmproj_path is not None:
        cmd.extend(["--mmproj", str(spec.mmproj_path)])
    if alias:
        cmd.extend(["--alias", alias])

    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    _wait_for_server(spec.port, alias)
    return proc, True


def _stop_process(proc: subprocess.Popen[bytes] | None) -> None:
    if proc is None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


class managed_experiment_runtime(AbstractContextManager[None]):
    def __init__(self, spec: ExperimentSpec):
        self.spec = spec
        self._proc: subprocess.Popen[bytes] | None = None
        self._owns_process = False

    def __enter__(self) -> None:
        runtime = self.spec.runtime
        if runtime.kind == "llamacpp_server":
            self._proc, self._owns_process = _start_llamacpp_server(runtime)
        elif runtime.kind == "hf_assets" and runtime.hf_assets is not None:
            _ensure_hf_assets(runtime.hf_assets)
        return None

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._owns_process:
            _stop_process(self._proc)
        return None
