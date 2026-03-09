from models.ner.common import NERBackend

BACKENDS = ["gliner2", "qwen3"]


def get_backend(name: str):
    if name == "gliner2":
        from models.ner.gliner2 import GLiNER2Backend
        return GLiNER2Backend
    if name == "qwen3":
        from models.ner.qwen3 import Qwen3Backend
        return Qwen3Backend
    raise ValueError(f"Unknown NER backend: {name!r}. Choose from: {BACKENDS}")
