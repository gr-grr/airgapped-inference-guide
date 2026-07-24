# Air-Gapped AI Inference Blueprint

Two identical standalone inference nodes, each with 2× NVIDIA H200 141GB GPUs (PCIe, no NVLink), in separate datacenters with no network link between them.

- **[RUNBOOK.md](./RUNBOOK.md)** — 10-step operational runbook from fresh Ubuntu install through air-gap enforcement
- **[inference-cluster-stack/](./inference-cluster-stack/)** — Docker Compose stack (vLLM PP=2, DCGM, Prometheus, Grafana)
- **[AGENTS.md](./AGENTS.md)** — Project context for AI coding agents

## Why we moved from Qwen3-VL-235B-A22B to Qwen3-VL-30B-A3B AWQ

The original 235B config hit 90% GPU utilization at just 80 users × 8K context. Target grew to 250 concurrent users at 46-64K context — KV cache alone would need ~750 GB, impossible on 2× H200 141GB.

### Benefits gained
- **17× less weight memory per GPU** (59 GB → 3.5 GB) → ~55 GB freed for KV cache
- **8× longer context** (8K → 64K) supports 46K single-request workflows
- **3× more concurrency** (80 → 250) with 84 fitting fully on-GPU
- **Native 256K context** (verified from config.json: `max_position_embeddings=262144`) — no rope scaling needed
- **FP8 KV cache** halves KV memory vs BF16

### What we lost
- **Model capability**: 235B/22B active → 30.5B/3.3B active. Weaker on long-tail knowledge and complex reasoning, but still strong on multilingual OCR, tool use, and agentic tasks
- **Video support** disabled (`--limit-mm-per-prompt.video 0`) to save memory
- **Swap-dependent throughput**: active decode limited to ~84 seqs on-GPU; remaining 166 rely on PCIe swap
- **FP8 KV cache** has <0.5% accuracy loss vs BF16

See [inference-cluster-stack/README.md](./inference-cluster-stack/README.md#configuration-rationale-july-2026) for the full rationale.
