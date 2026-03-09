from ner.common import NERBackend

BACKENDS = ["gliner2", "qwen3"]


def get_backend(name: str):
    if name == "gliner2":
        from ner.gliner2 import GLiNER2Backend
        return GLiNER2Backend
    if name == "qwen3":
        from ner.qwen3 import Qwen3Backend
        return Qwen3Backend
    raise ValueError(f"Unknown NER backend: {name!r}. Choose from: {BACKENDS}")
