import argparse
from pathlib import Path

# ``faiss`` is an optional dependency. Import lazily so modules depending on
# ``chat`` can still be imported without the native library present.
try:  # pragma: no cover
    import faiss  # type: ignore
except Exception:  # pragma: no cover
    faiss = None


def load_index(index_path: Path):
    """Load a FAISS index and associated text corpus.

    Raises
    ------
    RuntimeError
        If the ``faiss`` library is not available.
    """

    if faiss is None:  # pragma: no cover - exercised in unit tests
        raise RuntimeError(
            "faiss is required for search; please install the optional dependency"
        )
    index = faiss.read_index(str(index_path))
    texts = (index_path.parent / 'texts.jsonl').read_text(encoding='utf-8').splitlines()
    return index, texts


def search(query: str, index, texts, embed_model, top_k: int = 5):
    vec = embed_model.encode([query])
    _, indices = index.search(vec, top_k)
    return [texts[i] for i in indices[0]]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='mistralai/Mistral-7B-v0.1')
    parser.add_argument('--index', type=Path, default=Path('faiss.index'))
    args = parser.parse_args()

    index, texts = load_index(args.index)
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
    from sentence_transformers import SentenceTransformer
    import gradio as gr

    embed_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, device_map='auto')
    gen = pipeline('text-generation', model=model, tokenizer=tokenizer, device=0)

    def respond(message, history):
        retrieved = search(message, index, texts, embed_model)
        context = '\n'.join(retrieved)
        prompt = context + '\n' + message
        out = gen(prompt, max_new_tokens=200, do_sample=True)
        return out[0]['generated_text']

    gr.ChatInterface(respond).launch()


if __name__ == '__main__':
    main()
