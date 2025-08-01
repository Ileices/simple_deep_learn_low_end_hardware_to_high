import os
import sys
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src import agent


def test_record_code_deduplicates(tmp_path, monkeypatch):
    # Redirect script storage to temporary directory
    monkeypatch.setattr(agent, 'SCRIPTS', tmp_path)
    code = "print('hi')"
    path1 = agent.record_code(code)
    path2 = agent.record_code(code)
    assert path1 == path2
    files = list(tmp_path.glob('*.py'))
    assert len(files) == 1
    assert files[0].read_text() == code
