import argparse
import json
from pathlib import Path

from sentence_transformers import SentenceTransformer
import faiss


def read_dataset(path: Path):
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            text = data.get('content') or data.get('text') or ''
            yield text.strip()


def build_index(dataset_path: Path, index_path: Path, model_name: str = 'sentence-transformers/all-MiniLM-L6-v2'):
    model = SentenceTransformer(model_name)
    texts = list(read_dataset(dataset_path))
    embeddings = model.encode(texts, batch_size=64, show_progress_bar=True)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    faiss.write_index(index, str(index_path))
    (index_path.parent / 'texts.jsonl').write_text('\n'.join(texts), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description='Build FAISS index from dataset')
    parser.add_argument('--data', type=Path, required=True, help='Path to JSONL dataset')
    parser.add_argument('--out', type=Path, default=Path('faiss.index'), help='Path to store index file')
    parser.add_argument('--model', type=str, default='sentence-transformers/all-MiniLM-L6-v2', help='SentenceTransformer model')
    args = parser.parse_args()
    build_index(args.data, args.out, args.model)


if __name__ == '__main__':
    main()
