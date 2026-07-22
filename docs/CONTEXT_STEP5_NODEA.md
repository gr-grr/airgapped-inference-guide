# Step 5 Context — Node A (node1) Model Decision

## Current State

Steps 1-4 complete on Node A (OS, RAID10, NVIDIA driver 580.159.03 + CUDA 13-3, Docker + nvidia-container-toolkit).

## Hardware

- 2× NVIDIA H200 NVL 141GB (PCIe, no NVLink) = 282 GB total VRAM
- PP=2 (Pipeline Parallelism) per vLLM docs — optimal for PCIe-connected GPUs
- 8× 7.6 TB NVMe RAID10 mounted at `/data` (28T available)
- RAID10 resync in progress (45.3%, ~22h remaining) — safe to write during resync
- Hostname: `node1`
- Internet: available (DNS flaky — `/etc/resolv.conf` set to `8.8.8.8`, `chattr +i` to prevent overwrite)
- `huggingface-cli` / `hf` installed via pipx at `~/.local/bin/hf`
- HuggingFace token `h200-token` logged in

## Model Decision

**Chosen model:** Qwen3-VL-235B-A22B-Instruct at AWQ (INT4)

### Why Qwen3-VL-235B

Research conducted July 2026 shows:

1. **Multilingual OCR in 39 languages** (>70% accuracy in 32 languages including Arabic, French, English) — per Qwen3-VL technical report (arxiv 2511.21631, Nov 2025)
2. **Handwriting recognition** — outperforms Llama 4 Scout across document/OCR benchmarks (DocVQA, OmniDocBench)
3. **Reasoning**: GPQA 77.2, MMLU-Pro 83.6 — beats Llama 4 Scout (MMLU-Pro 74.3) significantly
4. **Tool use & agentic**: Native ReAct agent loops, TAU-bench agentic score 54.1% vs Llama 4 Scout 15.5%. Tested independently — "the most capable open-weight agent model currently available"
5. **Apache 2.0 license** — cleanest for commercial use
6. **vLLM compatible** — supports PP=2, AWQ quantization, multimodal

### Quantization choice: AWQ (INT4)

| Quant | Weight Size | KV Cache Headroom | Multi-User |
|-------|-------------|-------------------|------------|
| AWQ (INT4) | ~118 GB | ~152 GB | Excellent |
| FP8 | ~235 GB | ~35 GB | Tight |

For multi-user API serving, AWQ is the right call — 152 GB headroom for KV cache.

### Model ID for download

`QuantTrio/Qwen3-VL-235B-A22B-Instruct-AWQ` — community AWQ quant supported by vLLM.

**Secondary model (later):** Llama 4 Scout 109B at FP8 for its 10M token context on long documents.

## Previous download killed

- Qwen3-VL-235B at FP8 was partially downloaded (62 GB of ~235 GB) and killed on user request
- Llama 4 Scout was not downloaded (only config files, 68K)
- Need to clean up `/data/stack/inference-cluster-stack/data/models/` before restarting

## Next Actions for new conversation

1. Read `AGENTS.md`, `RUNBOOK.md`, `NODE_A_INFO.md`, `TRACKER.md`
2. Read this file: `docs/CONTEXT_STEP5_NODEA.md`
3. Clean up old partial downloads: `rm -rf /data/stack/inference-cluster-stack/data/models/qwen3-vl-235b`
4. Download AWQ model: `hf download QuantTrio/Qwen3-VL-235B-A22B-Instruct-AWQ --local-dir /data/stack/inference-cluster-stack/data/models/qwen3-vl-235b`
5. Once done, checksum the model files
6. Update TRACKER.md — Step 5 complete, proceed to Step 6
