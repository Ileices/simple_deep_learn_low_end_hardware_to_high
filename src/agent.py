import argparse
import ast
import hashlib
import json
import subprocess
from pathlib import Path

# Heavy NLP libraries are optional. They are imported lazily so that utility
# helpers like ``record_code`` remain usable in environments where the
# dependencies are not available (e.g. during lightweight unit tests).
try:  # pragma: no cover - exercised in integration environments
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
    from peft import PeftModel
except Exception:  # pragma: no cover
    AutoModelForCausalLM = AutoTokenizer = pipeline = PeftModel = None


WORKSPACE = Path('workspace')
LOGS = WORKSPACE / 'logs'
SCRIPTS = WORKSPACE / 'scripts'
HISTORY = WORKSPACE / 'history.log'


def generate_code(gen, prompt: str) -> str:
    out = gen(prompt, max_new_tokens=200, do_sample=True)
    return out[0]['generated_text']


def run_code(path: Path) -> tuple[int, str]:
    proc = subprocess.run(['python', str(path)], capture_output=True, text=True)
    log = LOGS / f"{path.stem}.log"
    log.write_text(proc.stdout + '\n' + proc.stderr)
    return proc.returncode, proc.stderr


def hash_code(code: str) -> str:
    """Return a stable hash based on the code's AST structure."""
    tree = ast.parse(code)
    normalized = ast.dump(tree, annotate_fields=False, include_attributes=False)
    return hashlib.sha256(normalized.encode()).hexdigest()


def record_code(code: str) -> Path:
    """Store ``code`` under a deterministic hash, avoiding duplicates."""
    h = hash_code(code)[:16]
    path = SCRIPTS / f"{h}.py"
    if not path.exists():
        path.write_text(code)
    return path


def main():
    parser = argparse.ArgumentParser(description='Generate and run code tasks')
    parser.add_argument('--instructions', type=Path, required=True,
                        help='Text file with one task per line')
    parser.add_argument('--model', default='mistralai/Mistral-7B-v0.1')
    parser.add_argument('--lora', type=Path, help='Path to LoRA adapter weights')
    parser.add_argument('--dataset-out', type=Path,
                        help='Append successful tasks to this JSONL dataset')
    args = parser.parse_args()

    WORKSPACE.mkdir(exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    SCRIPTS.mkdir(parents=True, exist_ok=True)
    if not HISTORY.exists():
        HISTORY.write_text('task,attempt,returncode,script\n')

    if AutoTokenizer is None or AutoModelForCausalLM is None or pipeline is None:
        raise RuntimeError(
            "transformers and peft are required to generate code; please install"
            " the optional dependencies"
        )

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, device_map='auto')
    if args.lora:
        model = PeftModel.from_pretrained(model, args.lora)
    gen = pipeline('text-generation', model=model, tokenizer=tokenizer, device=0)

    tasks = [t.strip() for t in args.instructions.read_text().splitlines() if t.strip()]
    for idx, task in enumerate(tasks, 1):
        error = ''
        for attempt in range(3):
            prompt = f"Write a Python script to {task}. Only provide the code."\
                + (f"\nPrevious error:\n{error}" if error else '')
            code = generate_code(gen, prompt)
            script_path = record_code(code)
            rc, error = run_code(script_path)
            with HISTORY.open('a', encoding='utf-8') as log:
                log.write(f"{idx},{attempt},{rc},{script_path.name}\n")
            if rc == 0:
                print(f'Task {idx} succeeded')
                if args.dataset_out:
                    with args.dataset_out.open('a', encoding='utf-8') as ds:
                        ds.write(json.dumps({'prompt': task, 'response': code}) + '\n')
                break
            else:
                print(f'Task {idx} attempt {attempt+1} failed')
        else:
            print(f'Task {idx} failed after 3 attempts')


if __name__ == '__main__':
    main()
