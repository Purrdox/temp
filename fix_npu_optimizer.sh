#!/usr/bin/env bash
set -euo pipefail
cd /mnt/workspace/aics-lab2-20261104/assign
export PYTHONPATH="$PWD/.python-packages${PYTHONPATH:+:$PYTHONPATH}"
python3 - <<'PY'
from pathlib import Path
from urllib.request import urlopen
import hashlib
base = "https://raw.githubusercontent.com/Purrdox/temp/5dfc62f8eacc7aa4e78526255033b9ab8b29bad7/"
files = [["nanochat/optim.py","npu-fix/optim.py","45aa9e84c44ab879a4ead518036c455b598f6c067d6125ec04b6700374ff137e"],["scripts/check_npu_optimizer.py","npu-fix/check_npu_optimizer.py","d8293ebd427d2668987da7f2fc12aaa21359d8bbdbdf8043fe0e5763868c247c"]]
downloads = []
for local, remote, expected in files:
    with urlopen(base + remote, timeout=60) as response:
        data = response.read()
    assert hashlib.sha256(data).hexdigest() == expected, "Patch SHA-256 mismatch: " + local
    downloads.append((Path(local), data))
backup = Path("nanochat/optim.py.pre-npu-scalar-fix")
if not backup.exists():
    backup.write_bytes(Path("nanochat/optim.py").read_bytes())
for target, data in downloads:
    target.write_bytes(data)
print("NPU optimizer patch downloaded and verified", flush=True)
PY
python3 -u scripts/check_npu_optimizer.py
PYTHON=python3 bash runs/train_npu.sh
