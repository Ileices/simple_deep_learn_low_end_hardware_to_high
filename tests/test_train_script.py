import json
import subprocess
import sys
from pathlib import Path


def test_train_script(tmp_path):
    data = tmp_path / 'data.jsonl'
    sample = {"prompt": "Hello", "response": "world"}
    data.write_text(json.dumps(sample) + "\n", encoding='utf-8')
    out = tmp_path / 'out'
    cmd = [sys.executable, 'src/train.py', '--data', str(data), '--model', 'sshleifer/tiny-gpt2', '--output', str(out), '--device', 'cpu', '--lora-target', 'c_attn']
    subprocess.run(cmd, check=True)
    assert (out / 'adapter_model.bin').exists()
