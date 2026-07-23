# Step 8 Context — Node A (node1) Verify the Deployment

## Current State

Steps 1-8 complete on Node A.

## System Identity

- **Hostname:** node1
- **IP:** 192.168.100.119
- **GPU:** 2× NVIDIA H200 NVL 141GB (PCIe, no NVLink), driver 580.159.03
- **RAID10:** 8× 7.6 TB NVMe at `/dev/md0` → `/data` (28T available)
- **Docker:** Engine 29.6.2, Compose v5.3.1
- **Internet:** Available (not yet air-gapped)

## vLLM

- **Image:** `vllm/vllm-openai:patched-qwen3vl-pp` (v0.25.1)
- **Model:** Qwen3-VL-235B-A22B-AWQ
- **Served model name:** `qwen3-vl`
- **Flags:** PP=2, FP8, enable-auto-tool-choice, tool-call-parser=hermes

## What Was Done in This Session

### Tool calling fix
- vLLM was serving with `--served-model-name qwen3-vl` but missing `--enable-auto-tool-choice` and `--tool-call-parser`
- Open WebUI sends `tool_choice: "auto"` which requires these flags
- **Fix:** Added `--enable-auto-tool-choice` and `--tool-call-parser hermes` to the vLLM command in `docker-compose.yml`
- Applied to both deployed and repo copies
- Verified: tool calls return properly parsed (`finish_reason: tool_calls`, `tool_calls` array populated)

### Step 8 verification — all 13 services
| Service | Status |
|---|---|
| postgres | healthy |
| redis | healthy |
| qdrant | healthy |
| vllm | healthy |
| dcgm-exporter | healthy |
| n8n | healthy |
| open-webui | healthy |
| prometheus | healthy |
| grafana | healthy |
| loki | disabled (healthcheck) |
| promtail | healthy |
| node-exporter | healthy |
| alertmanager | healthy |

### Repo state
- 4 commits ahead of origin/main
- 2 pending changes committed in this session:
  1. Fix served-model-name, remove API key env vars, update RAG_EMBEDDING_MODEL
  2. Enable tool calling with `--enable-auto-tool-choice --tool-call-parser hermes`

## Next Steps (for next session)

### Steps 9-11
- Step 9: N/A (not adding new models now)
- Step 10: Pre-cutoff hardening
  - Disable Ubuntu auto-updates (`systemctl disable --now apt-daily.timer`)
  - Remove unattended-upgrades
  - Snapshot package state (`dpkg --get-selections`, `docker images > ~/docker-images.txt`)
  - Save Docker images as `.tar`
- Step 11: Air-gap enforcement
  - WAN disconnect
  - Reboot and verify full offline recovery

### Node B
- Not started yet — repeat Steps 1-8 identically

### Minor known diff
- Deployed compose file has `grafana/promtail:3.6.11`, repo has `grafana/promtail:3.7.3`
- `.env` in repo is a template (gitignored); actual `.env` on Node A has proper secrets
