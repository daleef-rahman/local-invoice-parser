from vlm.common import VLMBackend

BACKENDS = ["llamacpp", "minicpmv", "qwen25vl"]


def get_backend(name: str):
    if name == "llamacpp":
        from vlm.llamacpp import LlamaCppVLMBackend
        return LlamaCppVLMBackend
    if name == "minicpmv":
        from vlm.minicpmv import MiniCPMVBackend
        return MiniCPMVBackend
    if name == "qwen25vl":
        from vlm.qwen25vl import Qwen25VLBackend
        return Qwen25VLBackend
    raise ValueError(f"Unknown VLM backend: {name!r}. Choose from: {BACKENDS}")
