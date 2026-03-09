from models.vlm.common import VLMBackend

BACKENDS = ["llamacpp", "minicpmv", "qwen25vl"]


def get_backend(name: str):
    if name == "llamacpp":
        from models.vlm.llamacpp import LlamaCppVLMBackend
        return LlamaCppVLMBackend
    if name == "minicpmv":
        from models.vlm.minicpmv import MiniCPMVBackend
        return MiniCPMVBackend
    if name == "qwen25vl":
        from models.vlm.qwen25vl import Qwen25VLBackend
        return Qwen25VLBackend
    raise ValueError(f"Unknown VLM backend: {name!r}. Choose from: {BACKENDS}")
