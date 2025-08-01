import pytest
from src import chat


def test_load_index_requires_faiss(monkeypatch, tmp_path):
    monkeypatch.setattr(chat, "faiss", None)
    with pytest.raises(RuntimeError):
        chat.load_index(tmp_path / "faiss.index")
