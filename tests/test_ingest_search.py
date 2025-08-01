import json
from pathlib import Path
import os
import sys

import numpy as np
import faiss

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.ingest import build_index
from src.chat import search


class DummyEmbed:
    def encode(self, texts, **kwargs):
        return np.array([[len(t), 0, 0] for t in texts], dtype='float32')


def test_build_and_search(tmp_path: Path):
    data = [
        {"text": "hello world"},
        {"text": "goodbye"}
    ]
    ds_path = tmp_path / "data.jsonl"
    with ds_path.open('w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')

    index_path = tmp_path / "faiss.index"
    build_index(ds_path, index_path, embed_model=DummyEmbed())

    index = faiss.read_index(str(index_path))
    texts = (index_path.parent / 'texts.jsonl').read_text(encoding='utf-8').splitlines()

    results = search("hello", index, texts, DummyEmbed(), top_k=2)
    assert "hello world" in results
