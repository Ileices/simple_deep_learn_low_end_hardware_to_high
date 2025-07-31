import argparse
import subprocess
from pathlib import Path


REQUIREMENTS = Path(__file__).resolve().parent.parent / 'requirements.txt'


def run(cmd):
    subprocess.run(cmd, shell=True, check=True)


def main():
    parser = argparse.ArgumentParser(description='Bootstrap environment')
    parser.add_argument('--data', type=Path, required=True, help='Path to JSONL dataset')
    args = parser.parse_args()

    run(f'pip install -r {REQUIREMENTS}')
    run(f'python src/ingest.py --data {args.data}')
    print('Starting chat interface...')
    run('python src/chat.py')


if __name__ == '__main__':
    main()
