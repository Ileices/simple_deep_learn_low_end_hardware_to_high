import argparse
import subprocess
from pathlib import Path


def run(cmd: str):
    subprocess.run(cmd, shell=True, check=True)


def main():
    parser = argparse.ArgumentParser(description='Continuous train->code loop')
    parser.add_argument('--data', type=Path, required=True, help='JSONL dataset path')
    parser.add_argument('--tasks', type=Path, required=True, help='Text file with tasks')
    parser.add_argument('--model', default='mistralai/Mistral-7B-v0.1')
    parser.add_argument('--cycles', type=int, default=1, help='Number of train/code cycles')
    args = parser.parse_args()

    for _ in range(args.cycles):
        run(f'python src/train.py --data {args.data} --model {args.model}')
        run(
            'python src/agent.py --instructions {tasks} --model {model} --lora lora-output '
            '--dataset-out {data}'.format(tasks=args.tasks, model=args.model, data=args.data)
        )


if __name__ == '__main__':
    main()
