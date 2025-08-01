import argparse
import uuid
import json
from pathlib import Path
from typing import List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn


def load_tasks(path: Path) -> List[dict]:
    tasks = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        tasks.append({'id': str(uuid.uuid4()), 'command': line, 'status': 'pending', 'result': None})
    return tasks


def create_app(tasks: List[dict]) -> tuple[FastAPI, List[dict]]:
    app = FastAPI()
    state = {'tasks': tasks, 'workers': set()}

    @app.post('/register')
    def register():
        worker_id = str(uuid.uuid4())
        state['workers'].add(worker_id)
        return {'worker_id': worker_id}

    @app.get('/get_task/{worker_id}')
    def get_task(worker_id: str):
        if worker_id not in state['workers']:
            raise HTTPException(status_code=404, detail='Unknown worker')
        for task in state['tasks']:
            if task['status'] == 'pending':
                task['status'] = 'running'
                task['worker'] = worker_id
                return task
        return {}

    class Result(BaseModel):
        task_id: str
        worker_id: str
        returncode: int
        output: str

    @app.post('/submit_result')
    def submit_result(res: Result):
        for task in state['tasks']:
            if task['id'] == res.task_id and task.get('worker') == res.worker_id:
                task['status'] = 'done'
                task['result'] = {'returncode': res.returncode, 'output': res.output}
                return {'status': 'ok'}
        raise HTTPException(status_code=404, detail='Task not found')

    @app.get('/results')
    def results():
        return state['tasks']

    return app, state['tasks']


def main():
    parser = argparse.ArgumentParser(description='Hub server')
    parser.add_argument('--tasks', type=Path, required=True, help='File with commands, one per line')
    parser.add_argument('--host', default='0.0.0.0')
    parser.add_argument('--port', type=int, default=8000)
    args = parser.parse_args()
    tasks = load_tasks(args.tasks)
    app, _ = create_app(tasks)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == '__main__':
    main()
