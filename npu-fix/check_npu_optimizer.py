#!/usr/bin/env python3
"""Check real NPU AdamW/Muon updates before starting the full training loop."""
import os
from pathlib import Path
import sys

os.environ["NANOCHAT_DISABLE_COMPILE"] = "1"
os.environ["NANOCHAT_DTYPE"] = "bfloat16"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    import torch
    import torch_npu
    from nanochat.optim import MuonAdamW

    assert torch.npu.is_available() and torch.npu.device_count() == 1
    torch.npu.set_device(0)
    torch.manual_seed(42)
    initial = torch.randn(8, 16, dtype=torch.float32)
    cpu_p = torch.nn.Parameter(initial.clone())
    npu_p = torch.nn.Parameter(initial.to("npu:0"))
    config = dict(lr=0.003, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1)
    reference = torch.optim.AdamW([cpu_p], **config, foreach=False)
    adam = MuonAdamW([dict(params=[npu_p], kind="adamw", **config)])
    for _ in range(3):
        grad = torch.randn_like(initial)
        cpu_p.grad = grad.clone()
        npu_p.grad = grad.to("npu:0")
        reference.step()
        adam.step()
    torch.npu.synchronize()
    torch.testing.assert_close(npu_p.detach().cpu(), cpu_p.detach(), rtol=1e-5, atol=2e-6)
    print("NPU AdamW: 3 updates agree with torch.optim.AdamW CPU reference", flush=True)

    embedding = torch.nn.Parameter(torch.randn(16, 16).to("npu:0", dtype=torch.bfloat16))
    scalar = torch.nn.Parameter(torch.ones((), device="npu:0"))
    square = torch.nn.Parameter(torch.randn(16, 16).to("npu:0"))
    tall = torch.nn.Parameter(torch.randn(32, 16).to("npu:0"))
    wide = torch.nn.Parameter(torch.randn(16, 32).to("npu:0"))
    params = [embedding, scalar, square, tall, wide]
    before = [p.detach().clone() for p in params]
    groups = [dict(params=[embedding, scalar], kind="adamw", **config)]
    groups += [dict(params=[p], kind="muon", lr=0.01, momentum=0.95,
                    ns_steps=5, beta2=0.95, weight_decay=0.1) for p in params[2:]]
    optimizer = MuonAdamW(groups)
    for _ in range(3):
        for p in params:
            p.grad = torch.randn(p.shape, dtype=torch.float32).to(p.device, dtype=p.dtype)
        optimizer.step()
    torch.npu.synchronize()
    for p, old in zip(params, before):
        assert torch.isfinite(p).all().item(), "Nonfinite optimizer output"
        assert (p != old).any().item(), "Parameter was not updated"
    print("NPU mixed AdamW/Muon: bf16 embedding, scalar, square/tall/wide matrices updated; all finite", flush=True)
    print("NPU OPTIMIZER CHECK PASSED", flush=True)


if __name__ == "__main__":
    main()
