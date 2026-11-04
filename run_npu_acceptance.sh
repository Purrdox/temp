#!/usr/bin/env bash
set -euo pipefail
cd /mnt/workspace/aics-lab2-20261104/assign
if pgrep -u "$(id -u)" -f 'scripts[.]base_train' >/dev/null; then
    echo 'Formal training is still running. Finish it before running benchmarks.' >&2
    exit 1
fi
OLD_PYTHONPATH="$(printenv PYTHONPATH || true)"
export PYTHONPATH="$PWD/.python-packages:$OLD_PYTHONPATH"
export PYTHONUTF8=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2
export NANOCHAT_DTYPE=bfloat16 NANOCHAT_DISABLE_VE=1 NANOCHAT_DISABLE_COMPILE=1
export NANOCHAT_ATTN_IMPL=sdpa NANOCHAT_RMSNORM_IMPL=native
export NANOCHAT_BASE_DIR="$PWD/outputs/npu"
unset RANK LOCAL_RANK WORLD_SIZE LOCAL_WORLD_SIZE MASTER_ADDR MASTER_PORT
GH="$(command -v gh || true)"
if [[ -z "$GH" ]]; then GH=/mnt/workspace/.tools/github-cli/bin/gh; fi
[[ -x "$GH" ]] || { echo 'Existing GitHub CLI not found' >&2; exit 1; }
export ACCEPTANCE_GH="$GH"
python3 -u - <<'PY'
from pathlib import Path
import base64,hashlib,json,os,subprocess
commit="7ee22b2da02a2427bde424dca742d8175bbd4792"
files=[["scripts/bench_attention.py","npu-acceptance/bench_attention.py","411dcc2c918203a882f1789439ff1b41fd8a3fbadf652d71a25745a690334e48"],["scripts/check_npu_model.py","npu-acceptance/check_npu_model.py","05469a8dfd57df68724a6d1f2c11b084e70f7ad48df9de26c4595072528bbb1e"]]
pending=[]
for local,remote,expected in files:
    print("Updating:",local,flush=True)
    raw=subprocess.check_output([os.environ['ACCEPTANCE_GH'],'api',
        'repos/Purrdox/temp/contents/'+remote+'?ref='+commit])
    info=json.loads(raw)
    data=base64.b64decode(info['content'])
    assert hashlib.sha256(data).hexdigest()==expected, 'Source checksum mismatch: '+local
    pending.append((Path(local),data))
for path,data in pending:
    path.write_bytes(data)
print("Acceptance scripts downloaded and verified",flush=True)
PY
STAMP="$(TZ=Asia/Shanghai date +%Y%m%d-%H%M%S)"
OUT="$NANOCHAT_BASE_DIR/acceptance/$STAMP"
mkdir -p "$OUT"
echo "Acceptance output: $OUT"
echo 'STEP 1: independent checkpoint reload, cache logits, inference speed, continuations'
MODEL_STATUS=0
python3 -u scripts/check_npu_model.py --trust-local-checkpoints \
    --base-dir "$NANOCHAT_BASE_DIR" --output "$OUT/model-results.json" \
    2>&1 | tee "$OUT/model.log" || MODEL_STATUS=$?
echo 'STEP 2: correctness before speed; full/window and forward/backward measured separately'
OPERATOR_STATUS=0
python3 -u scripts/bench_attention.py --batch 8 --seq 512 --heads 4 --head-dim 128 \
    --window 128 --depth 8 --total-batch 32768 --output "$OUT/operator-results.json" \
    2>&1 | tee "$OUT/operators.log" || OPERATOR_STATUS=$?
export ACCEPTANCE_OUT="$OUT" ACCEPTANCE_STAMP="$STAMP" MODEL_STATUS OPERATOR_STATUS
python3 - <<'PY'
from pathlib import Path
import hashlib,json,os,zipfile
out=Path(os.environ['ACCEPTANCE_OUT'])
summary={'model_exit_code':int(os.environ['MODEL_STATUS']),
         'operator_exit_code':int(os.environ['OPERATOR_STATUS'])}
for key,name in (('model','model-results.json'),('operators','operator-results.json')):
    path=out/name
    summary[key]=json.loads(path.read_text(encoding='utf-8')) if path.exists() else {'passed':False,'error':'Report missing; inspect log'}
summary['source_sha256']={str(p):hashlib.sha256(p.read_bytes()).hexdigest()
    for p in [*Path('nanochat').glob('*.py'),Path('scripts/bench_attention.py'),Path('scripts/check_npu_model.py')]}
summary['passed']=all(summary[k].get('passed',False) for k in ('model','operators'))
summary['payload_commit']='7ee22b2da02a2427bde424dca742d8175bbd4792'
path=out/('npu-acceptance-'+os.environ['ACCEPTANCE_STAMP']+'.json')
path.write_text(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
archive=out/('npu-acceptance-'+os.environ['ACCEPTANCE_STAMP']+'.zip')
with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as z:
    for p in out.iterdir():
        if p.is_file() and p.suffix in ('.json','.log'):
            z.write(p,arcname=p.name)
    for p in [*Path('nanochat').glob('*.py'),Path('scripts/bench_attention.py'),Path('scripts/check_npu_model.py')]:
        z.write(p,arcname='source/'+p.as_posix())
print('ACCEPTANCE SUMMARY:',path,flush=True)
print('PASSED:',summary['passed'],flush=True)
PY
if "$GH" auth status --hostname github.com >/dev/null 2>&1; then
    if "$GH" release upload npu-results-20261104-132945 \
        "$OUT/npu-acceptance-$STAMP.json" "$OUT/npu-acceptance-$STAMP.zip" --repo Purrdox/temp; then
        echo 'REPORTS UPLOADED: https://github.com/Purrdox/temp/releases/tag/npu-results-20261104-132945'
    else
        echo "Report upload failed; reports are retained at $OUT" >&2
    fi
fi
if (( MODEL_STATUS != 0 || OPERATOR_STATUS != 0 )); then
    echo 'ACCEPTANCE NEEDS FIXES: inspect the saved JSON/logs; do not treat failed cases as passed.'
    exit 1
fi
echo 'NPU ACCEPTANCE FINISHED'
