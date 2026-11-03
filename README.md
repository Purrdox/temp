# Temporary CANNLab transfer

Public transfer authorized by the repository owner. Contains coursework code, the bundled tokenizer, and 32 public-domain documents per split for NPU pipeline checks only. Model weights, formal datasets, environments, logs, and credentials are excluded. Smoke BPB is not directly comparable with the full local smoke corpus.

In the CANNLab terminal, download bootstrap_npu.sh from this repository, then run it with bash. It downloads and verifies the code archive, installs only pyarrow/tiktoken and their dependencies in the project, then runs the single-NPU d8/50-step SDPA/native baseline. It reuses the preinstalled PyTorch/torch_npu.

The launcher checks that exactly one NPU is visible, uses logical npu:0, bf16, micro-batch 8, sequence 512, total batch 32768, and saves/evaluates at steps 25 and 50. This is a pipeline smoke test, not formal pretraining. Checkpoints and logs are written under assign/outputs/npu.
