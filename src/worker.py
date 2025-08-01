import argparse
import time
import subprocess
import requests


def run_task(hub: str, worker_id: str):
    resp = requests.get(f'{hub}/get_task/{worker_id}')
    if resp.status_code != 200 or not resp.json():
        return False
    task = resp.json()
    cmd = task['command']
    proc = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    result = {
        'task_id': task['id'],
        'worker_id': worker_id,
        'returncode': proc.returncode,
        'output': proc.stdout + '\n' + proc.stderr,
    }
    requests.post(f'{hub}/submit_result', json=result)
    return True


def main():
    parser = argparse.ArgumentParser(description='Worker process')
    parser.add_argument('--hub', default='http://localhost:8000')
    parser.add_argument('--interval', type=float, default=1.0)
    args = parser.parse_args()

    resp = requests.post(f'{args.hub}/register')
    worker_id = resp.json()['worker_id']

    while True:
        if not run_task(args.hub, worker_id):
            time.sleep(args.interval)
        else:
            time.sleep(0.1)


if __name__ == '__main__':
    main()
