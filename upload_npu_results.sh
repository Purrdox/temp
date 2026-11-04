#!/usr/bin/env bash
# Back up original results to the public repository explicitly chosen by user.
set -euo pipefail
ROOT=/mnt/workspace/aics-lab2-20261104/assign
cd "$ROOT"
if pgrep -u "$(id -u)" -f 'scripts[.]base_train' >/dev/null; then
    echo 'Training is still running. Wait for FORMAL NPU QUEUE FINISHED before whole-directory backup.' >&2
    exit 1
fi
[[ -d outputs/npu/base_checkpoints ]] || { echo 'No NPU checkpoint directory found' >&2; exit 1; }
if command -v gh >/dev/null 2>&1; then
    GH="$(command -v gh)"
else
    TOOL_DIR=/mnt/workspace/.tools/github-cli
    mkdir -p "$TOOL_DIR"
    if [[ ! -x "$TOOL_DIR/bin/gh" ]]; then
        python3 - "$TOOL_DIR" <<'PY'
from pathlib import Path
from urllib.request import Request, urlopen
import hashlib, json, platform, sys
root = Path(sys.argv[1])
arch = {'aarch64':'arm64', 'x86_64':'amd64'}[platform.machine()]
def fetch(url):
    with urlopen(Request(url, headers={'User-Agent':'aics-lab2-backup'}), timeout=90) as response:
        return response.read()
release = json.loads(fetch('https://api.github.com/repos/cli/cli/releases/latest'))
asset = next(a for a in release['assets'] if a['name'].endswith('_linux_' + arch + '.tar.gz'))
checksums = next(a for a in release['assets'] if a['name'].endswith('_checksums.txt'))
lines = fetch(checksums['browser_download_url']).decode().splitlines()
expected = next(line.split()[0] for line in lines if line.split()[-1].lstrip('*') == asset['name'])
data = fetch(asset['browser_download_url'])
assert hashlib.sha256(data).hexdigest() == expected, 'GitHub CLI package checksum mismatch'
(root / 'gh-release.tar.gz').write_bytes(data)
print('Official GitHub CLI downloaded; SHA-256 verified', flush=True)
PY
        tar -xzf "$TOOL_DIR/gh-release.tar.gz" -C "$TOOL_DIR" --strip-components=1
    fi
    GH="$TOOL_DIR/bin/gh"
fi
"$GH" --version
if ! "$GH" auth status --hostname github.com >/dev/null 2>&1; then
    echo 'Authorize GitHub CLI using the one-time code at https://github.com/login/device.'
    GH_BROWSER=echo "$GH" auth login --hostname github.com --git-protocol https --web --skip-ssh-key
fi
ACCOUNT="$("$GH" api user --jq .login)"
[[ "$ACCOUNT" == Purrdox ]] || { echo "Logged in as $ACCOUNT; log in as Purrdox before uploading." >&2; exit 1; }
CAN_PUSH="$("$GH" api repos/Purrdox/temp --jq .permissions.push)"
[[ "$CAN_PUSH" == true ]] || { echo 'No write permission for Purrdox/temp' >&2; exit 1; }

TAG="npu-results-$(TZ=Asia/Shanghai date +%Y%m%d-%H%M%S)"
EXPORT_DIR="/mnt/workspace/lab2-raw-backups/$TAG"
mkdir -p "$EXPORT_DIR"
echo "Packing original files without torch.load: $EXPORT_DIR"
# Only results and project source; no formal dataset, Python environment or credentials.
tar --exclude='__pycache__' -cf - outputs/npu nanochat scripts runs tokenizer \
    | gzip -1 \
    | split -b 1500M -d -a 3 - "$EXPORT_DIR/npu-results.tar.gz.part"
(
    cd "$EXPORT_DIR"
    sha256sum npu-results.tar.gz.part* > SHA256SUMS.txt
)
printf '%s\n' "$TAG" > "$EXPORT_DIR/RELEASE-TAG.txt"
ASSETS=("$EXPORT_DIR"/npu-results.tar.gz.part* "$EXPORT_DIR/SHA256SUMS.txt")
"$GH" release create "$TAG" "${ASSETS[@]}" --repo Purrdox/temp --target main \
    --draft --title "NPU original training results $TAG" \
    --notes 'Original NPU checkpoints, optimizer states, logs, metadata, tokenizer and source. No dataset or credentials. Concatenate numbered parts in order to reconstruct npu-results.tar.gz. Verify SHA256SUMS.txt first. Native checkpoint format retained.'
"$GH" release edit "$TAG" --repo Purrdox/temp --draft=false --latest
echo "UPLOAD FINISHED: https://github.com/Purrdox/temp/releases/tag/$TAG"
