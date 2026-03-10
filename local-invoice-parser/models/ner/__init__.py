from models.modelbackend import ModelBackend

BACKENDS = ["gliner2", "llama_server"]


def get_backend(name: str):
    if name == "gliner2":
        from models.ner.gliner2 import GLiNER2Backend
        return GLiNER2Backend
    if name == "llama_server":
        from models.ner.llama_server import LlamaServerNERBackend
        return LlamaServerNERBackend
    raise ValueError(f"Unknown NER backend: {name!r}. Choose from: {BACKENDS}")
