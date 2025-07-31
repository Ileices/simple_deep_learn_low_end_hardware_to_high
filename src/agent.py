import argparse
import subprocess
from pathlib import Path

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


WORKSPACE = Path('workspace')
LOGS = WORKSPACE / 'logs'


def generate_code(gen, prompt: str) -> str:
    out = gen(prompt, max_new_tokens=200, do_sample=True)
    return out[0]['generated_text']


def run_code(path: Path) -> tuple[int, str]:
    proc = subprocess.run(['python', str(path)], capture_output=True, text=True)
    log = (LOGS / f'{path.stem}.log')
    log.write_text(proc.stdout + '\n' + proc.stderr)
    return proc.returncode, proc.stderr


def main():
    parser = argparse.ArgumentParser(description='Generate and run code tasks')
    parser.add_argument('--instructions', type=Path, required=True,
                        help='Text file with one task per line')
    parser.add_argument('--model', default='mistralai/Mistral-7B-v0.1')
    args = parser.parse_args()

    WORKSPACE.mkdir(exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, device_map='auto')
    gen = pipeline('text-generation', model=model, tokenizer=tokenizer, device=0)

    tasks = [t.strip() for t in args.instructions.read_text().splitlines() if t.strip()]
    for idx, task in enumerate(tasks, 1):
        error = ''
        for attempt in range(3):
            prompt = f"Write a Python script to {task}. Only provide the code."\
                + (f"\nPrevious error:\n{error}" if error else '')
            code = generate_code(gen, prompt)
            script_path = WORKSPACE / f'task_{idx}.py'
            script_path.write_text(code)
            rc, error = run_code(script_path)
            if rc == 0:
                print(f'Task {idx} succeeded')
                break
            else:
                print(f'Task {idx} attempt {attempt+1} failed')
        else:
            print(f'Task {idx} failed after 3 attempts')


if __name__ == '__main__':
    main()
