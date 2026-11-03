#!/usr/bin/env bash
set -euo pipefail
ROOT="/mnt/workspace/aics-lab2-20261104"
mkdir -p "$ROOT"
cd "$ROOT"
python3 - <<'PY'
from pathlib import Path
from urllib.request import urlopen
import hashlib
path = Path("lab2-npu.tar.gz")
url = "https://raw.githubusercontent.com/Purrdox/temp/main/lab2-npu.tar.gz"
with urlopen(url, timeout=60) as response:
    data = response.read()
expected = "accdd5b1a37eaa9398153a30f8c3f0eb17d3bfac3c6c91c822c7c728152dd16b"
assert hashlib.sha256(data).hexdigest() == expected, "Archive SHA-256 mismatch"
path.write_bytes(data)
print("Code archive downloaded; SHA-256 verified", flush=True)
PY
tar -xzf lab2-npu.tar.gz
cd assign
python3 -m pip install --target "$PWD/.python-packages" --cache-dir "$PWD/.cache/pip" --only-binary=:all: 'pyarrow>=14,<26' 'tiktoken>=0.11,<0.13'
PYTHON=python3 bash runs/train_npu.sh
