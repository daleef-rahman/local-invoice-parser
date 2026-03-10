from models.modelbackend import ModelBackend

BACKENDS = ["llama_server", "llama_mtmd_cli"]


def get_backend(name: str):
    if name == "llama_server":
        from models.vlm.llama_server import LlamaServerVLMBackend
        return LlamaServerVLMBackend
    if name == "llama_mtmd_cli":
        from models.vlm.llama_mtmd_cli import LlamaMtmdCliVLMBackend
        return LlamaMtmdCliVLMBackend
    raise ValueError(f"Unknown VLM backend: {name!r}. Choose from: {BACKENDS}")
