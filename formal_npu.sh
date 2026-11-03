#!/usr/bin/env bash
# Formal course-data pretraining, one visible NPU, sequential d8/d4/d6.
set -euo pipefail
cd /mnt/workspace/aics-lab2-20261104/assign
export PYTHONPATH="$PWD/.python-packages${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1 PYTHONUTF8=1 OMP_NUM_THREADS=2
export NANOCHAT_DTYPE=bfloat16 NANOCHAT_ATTN_IMPL=sdpa NANOCHAT_RMSNORM_IMPL=native
export NANOCHAT_DISABLE_VE=1 NANOCHAT_DISABLE_COMPILE=1 NANOCHAT_ENFORCE_EAGER=1
export NANOCHAT_BASE_DIR="$PWD/outputs/npu"
export NANOCHAT_DATA_DIR="$PWD/data/course/base_data_climbmix"
unset RANK LOCAL_RANK WORLD_SIZE LOCAL_WORLD_SIZE MASTER_ADDR MASTER_PORT
STEPS="${STEPS:-4096}"
DEPTHS="${DEPTHS:-8 4 6}"
if [[ ! "$STEPS" =~ ^[0-9]+$ ]] || (( STEPS <= 0 || STEPS % 512 != 0 )); then
    echo "STEPS must be a positive multiple of 512" >&2; exit 1
fi
for depth in $DEPTHS; do
    case "$depth" in 4|6|8) ;; *) echo "DEPTHS may contain only 4, 6, 8" >&2; exit 1 ;; esac
done
PREFIX="$(date +%Y%m%d-%H%M%S)"
export NPU_QUEUE_DIR="$NANOCHAT_BASE_DIR/queues/$PREFIX"
mkdir -p "$NANOCHAT_BASE_DIR/logs" "$NANOCHAT_BASE_DIR/tokenizer" "$NPU_QUEUE_DIR"
echo "Formal NPU queue: depths=$DEPTHS steps=$STEPS, started $(date -Is)"
echo "Queue directory: $NPU_QUEUE_DIR"

python3 - <<'PY'
from pathlib import Path
import hashlib, json, os, shutil, sys
import torch, torch_npu
assert torch.npu.is_available() and torch.npu.device_count() == 1, 'Need exactly one visible NPU'
torch.npu.set_device(0)
assert shutil.disk_usage('.').free >= 20 * 2**30, 'Need at least 20 GiB free for data/checkpoints'
print('NPU:', torch.npu.get_device_name(0), 'torch:', torch.__version__, 'torch_npu:', torch_npu.__version__, flush=True)
print('Formal micro-batch 8 x seq 512 x accumulation 8 = 32768 training tokens/update', flush=True)
root = Path('data/manifests')
root.mkdir(parents=True, exist_ok=True)
manifest = root / 'packed_data.sha256'
expected = 'da5d54dc89ad63f99e42a89b5bf631a63887086e82bcc8d798e9b605851f54f3'
if not manifest.exists():
    manifest.write_text(expected + '  base_data_climbmix.tar\n', encoding='utf-8')
plan = {'device': torch.npu.get_device_name(0), 'torch': torch.__version__,
        'torch_npu': torch_npu.__version__, 'python': sys.version, 'seed':42,
        'seq':512, 'device_batch':8, 'total_batch':32768, 'warmup_steps':40,
        'eval_every':512, 'eval_tokens':524288, 'save_every':512,
        'attention':'sdpa', 'rmsnorm':'native', 'value_embeddings':False, 'compile':False,
        'formal_data_sha256': expected,
        'source_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in [*Path('nanochat').glob('*.py'), Path('scripts/base_train.py'), Path('tokenizer/tokenizer.pkl')]}}
Path(os.environ['NPU_QUEUE_DIR'], 'plan.json').write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding='utf-8')
PY

# Fetch only the formal pretraining archive, verify before extraction.
python3 -u scripts/fetch_box_data.py --only base_data_climbmix.tar --dest "$PWD/data/packed" --no-extract
python3 - <<'PY'
from pathlib import Path
import json, os
from scripts.fetch_box_data import sha256_of, extract_one
from scripts.overnight_4060 import check_data
archive = Path('data/packed/base_data_climbmix.tar')
expected = 'da5d54dc89ad63f99e42a89b5bf631a63887086e82bcc8d798e9b605851f54f3'
assert sha256_of(str(archive)) == expected, 'Archive SHA-256 mismatch; no training started'
Path('data/course').mkdir(parents=True, exist_ok=True)
ok, message = extract_one(str(archive), str(Path('data/course').resolve()))
print(message, flush=True)
assert ok, message
records = check_data(Path(os.environ['NANOCHAT_DATA_DIR']))
Path(os.environ['NPU_QUEUE_DIR'], 'data.json').write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding='utf-8')
print('FORMAL DATA READY:', len(records)-1, 'train shards + 1 held-out validation shard', flush=True)
PY

if [[ ! -f "$NANOCHAT_BASE_DIR/tokenizer/tokenizer.pkl" ]]; then
    cp tokenizer/tokenizer.pkl "$NANOCHAT_BASE_DIR/tokenizer/tokenizer.pkl"
fi
if [[ ! -f "$NANOCHAT_BASE_DIR/tokenizer/token_bytes.pt" ]]; then
    python3 scripts/make_token_bytes.py
fi
for depth in $DEPTHS; do
    TAG="d${depth}-npu-course-$PREFIX"
    if [[ -e "$NANOCHAT_BASE_DIR/base_checkpoints/$TAG" ]]; then
        echo "Model tag already exists: $TAG" >&2; exit 1
    fi
    echo "START $TAG: $(date -Is)"
    python3 - "$TAG" "$depth" "$STEPS" <<'PY'
from pathlib import Path
import json, os, sys
tag, depth, steps = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
Path(os.environ['NPU_QUEUE_DIR'], tag + '.json').write_text(json.dumps({
    'tag':tag, 'depth':depth, 'steps':steps, 'training_tokens':steps*32768,
    'warmup_steps':40, 'data_dir':os.environ['NANOCHAT_DATA_DIR']}, indent=2), encoding='utf-8')
PY
    python3 -u -m scripts.base_train \
        --device-type=npu --enforce-eager --run=dummy \
        --depth="$depth" --max-seq-len=512 --window-pattern=SSSL \
        --device-batch-size=8 --total-batch-size=32768 \
        --num-iterations="$STEPS" --warmup-steps=40 \
        --eval-every=512 --eval-tokens=524288 --save-every=512 \
        --sample-every=-1 --model-tag="$TAG" \
        2>&1 | tee "$NANOCHAT_BASE_DIR/logs/$TAG.log"
    python3 - "$TAG" "$STEPS" <<'PY'
from pathlib import Path
import json, os, sys
from scripts.overnight_4060 import summarize
result = summarize(Path(os.environ['NANOCHAT_BASE_DIR']), sys.argv[1], int(sys.argv[2]))
Path(os.environ['NPU_QUEUE_DIR'], sys.argv[1] + '-result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
PY
    echo "FINISHED $TAG: $(date -Is)"
done
echo "FORMAL NPU QUEUE FINISHED: $(date -Is)"
