from pathlib import Path
from fastapi.testclient import TestClient

from src.hub import create_app, load_tasks


def test_task_flow(tmp_path):
    tasks_file = Path('tests/sample_tasks.txt')
    tasks = load_tasks(tasks_file)
    app, _ = create_app(tasks)
    client = TestClient(app)

    r = client.post('/register')
    assert r.status_code == 200
    wid = r.json()['worker_id']

    r = client.get(f'/get_task/{wid}')
    assert r.status_code == 200
    task = r.json()
    assert 'command' in task

    res = {
        'task_id': task['id'],
        'worker_id': wid,
        'returncode': 0,
        'output': 'ok',
    }
    r = client.post('/submit_result', json=res)
    assert r.status_code == 200

    r = client.get('/results')
    data = r.json()
    assert any(t['id'] == task['id'] and t['status'] == 'done' for t in data)
