#!/usr/bin/env python3
"""Export report evidence and completed final checkpoints; no NPU kernels run."""
import argparse
from datetime import datetime
import hashlib
import io
import json
from pathlib import Path
import re
import sys
import zipfile


def archive_add(archive, name, data):
    archive.writestr(name, data, compress_type=(zipfile.ZIP_STORED if name.endswith('.pt') else zipfile.ZIP_DEFLATED))


def completed_models(base):
    completed = {}
    for path in sorted((base / 'queues').glob('*/d*-npu-course-*-result.json')):
        try:
            record = json.loads(path.read_text(encoding='utf-8'))
        except json.JSONDecodeError:
            continue  # A just-finishing model may still be writing its summary.
        tag = record['model_tag']
        if not re.fullmatch(r'd[468]-npu-course-[A-Za-z0-9_.-]+', tag):
            raise ValueError(f'Unexpected model tag: {tag}')
        filename = Path(record['checkpoint']).name
        match = re.fullmatch(r'model_(\d+)\.pt', filename)
        if not match:
            raise ValueError(f'Unexpected checkpoint filename: {filename}')
        step = int(match[1])
        folder = base / 'base_checkpoints' / tag
        meta = json.loads((folder / f'meta_{step:06d}.json').read_text(encoding='utf-8'))
        if meta['step'] != step or meta['user_config']['num_iterations'] != step:
            raise ValueError(f'Not a final checkpoint: {tag}')
        for name in (filename, f'optim_{step:06d}_rank0.pt'):
            if not (folder / name).is_file():
                raise FileNotFoundError(folder / name)
        completed[tag] = step
    return completed


def portable_checkpoint(path):
    import torch
    # torch_npu may be needed to recognize the original serialization format.
    # All storages are loaded onto CPU; no NPU device or kernels are requested.
    try:
        import torch_npu  # noqa: F401
    except ImportError:
        pass
    torch.set_num_threads(1)
    state = torch.load(path, map_location='cpu', weights_only=True)

    def cpu(value):
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().contiguous()
        if isinstance(value, dict):
            return {k: cpu(v) for k, v in value.items()}
        if isinstance(value, list):
            return [cpu(v) for v in value]
        if isinstance(value, tuple):
            return tuple(cpu(v) for v in value)
        return value

    buffer = io.BytesIO()
    torch.save(cpu(state), buffer)
    return buffer.getvalue()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    repo = Path(__file__).resolve().parents[1]
    parser.add_argument('--base-dir', type=Path, default=repo / 'outputs' / 'npu')
    parser.add_argument('--export-dir', type=Path, default=Path('/mnt/workspace/lab2-exports'))
    parser.add_argument('--include-optimizer', action='store_true')
    parser.add_argument('--logs-only', action='store_true')
    args = parser.parse_args()
    base = args.base_dir.resolve()
    if not base.is_dir():
        parser.error(f'Results directory does not exist: {base}')
    completed = completed_models(base)
    destination = args.export_dir / datetime.now().strftime('%Y%m%d-%H%M%S-%f')
    destination.mkdir(parents=True, exist_ok=False)
    report = destination / 'npu-report.zip'
    snapshot = {'exported_at':datetime.now().isoformat(timespec='seconds'),
                'completed_models':completed,
                'note':'Logs are a snapshot; models absent from completed_models may still be running.'}
    with zipfile.ZipFile(report, 'w') as archive:
        archive_add(archive, 'export-status.json', json.dumps(snapshot, ensure_ascii=False, indent=2).encode())
        for folder in ('logs', 'queues'):
            for path in sorted((base / folder).rglob('*')):
                if path.is_file() and path.suffix in ('.json', '.log'):
                    archive_add(archive, path.relative_to(base).as_posix(), path.read_bytes())
        for tag in completed:
            for path in sorted((base / 'base_checkpoints' / tag).glob('meta_*.json')):
                archive_add(archive, path.relative_to(base).as_posix(), path.read_bytes())
    print('Report:', report, flush=True)
    if completed and not args.logs_only:
        models = destination / 'npu-models.zip'
        with zipfile.ZipFile(models, 'w') as archive:
            runtime = {'NANOCHAT_DTYPE':'bfloat16', 'NANOCHAT_DISABLE_VE':'1',
                       'NANOCHAT_ATTN_IMPL':'sdpa', 'NANOCHAT_RMSNORM_IMPL':'native',
                       'NANOCHAT_DISABLE_COMPILE':'1'}
            archive_add(archive, 'runtime-env.json', json.dumps(runtime, indent=2).encode())
            for name in ('tokenizer.pkl', 'token_bytes.pt'):
                path = base / 'tokenizer' / name
                archive_add(archive, f'tokenizer/{name}', path.read_bytes())
            for tag, step in completed.items():
                print('Exporting CPU checkpoint:', tag, flush=True)
                folder = base / 'base_checkpoints' / tag
                names = [f'model_{step:06d}.pt']
                if args.include_optimizer:
                    names.append(f'optim_{step:06d}_rank0.pt')
                for name in names:
                    archive_add(archive, f'base_checkpoints/{tag}/{name}', portable_checkpoint(folder / name))
                meta_name = f'meta_{step:06d}.json'
                archive_add(archive, f'base_checkpoints/{tag}/{meta_name}', (folder / meta_name).read_bytes())
            for path in sorted((repo / 'nanochat').glob('*.py')):
                archive_add(archive, 'source/nanochat/' + path.name, path.read_bytes())
        print('Models:', models, flush=True)
    elif not completed:
        print('No completed formal model yet: report snapshot only. Run again after FINISHED d*-npu-course-*.', flush=True)
    lines = []
    for path in sorted(destination.glob('*.zip')):
        digest = hashlib.sha256()
        with path.open('rb') as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b''):
                digest.update(chunk)
        lines.append(f'{digest.hexdigest()}  {path.name}')
    (destination / 'SHA256SUMS.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('EXPORT FINISHED:', destination, flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
