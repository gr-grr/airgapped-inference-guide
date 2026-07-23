# Step 6 Context — Node A (node1) Stack Setup and Per-Node Config

## Current State

Steps 1-5 complete on Node A. Step 6 now completed.

## System Identity

- **Hostname:** node1
- **IP:** 192.168.100.119
- **GPU:** 2× NVIDIA H200 NVL 141GB (PCIe, no NVLink), driver 580.159.03
- **RAID10:** 8× 7.6 TB NVMe at `/dev/md0` → `/data` (28T available)
- **Docker:** Engine 29.6.2, Compose v5.3.1
- **Internet:** Available

## Step 6 — What Was Done

### 6.1 Stack directory
Stack already at `/data/stack/inference-cluster-stack/` — no copy needed.

### 6.2 Data directories
All 10 runtime data subdirs already exist under `data/`.

### 6.3 Pull container images
`docker compose pull` completed successfully. All 13 images pulled:

| Image | Tag |
|---|---|
| postgres | 18.4-alpine |
| redis | 8.8.0-alpine |
| qdrant/qdrant | v1.18.3 |
| vllm/vllm-openai | latest (v0.25.1) |
| nvidia/dcgm-exporter | 4.8.3 |
| n8nio/n8n | 2.31.5 |
| ghcr.io/open-webui/open-webui | v0.10.2 |
| prom/prometheus | v3.13.1 |
| grafana/grafana | 13.0.3 |
| grafana/loki | 3.7.3 |
| grafana/promtail | 3.6.11 |
| prom/node-exporter | v1.8.2 |
| prom/alertmanager | v0.33.1 |

### 6.4 Configure environment
`.env` configured with Node A values:
- `NODE_NAME=node1`
- `NODE_IP=192.168.100.119`
- `MODEL_DIR=/data/stack/inference-cluster-stack/data/models`
- `MODEL_NAME=/models/qwen3-vl-235b-awq` (primary)
- PP=2, dtype=auto, GPU_MEMORY_UTILIZATION=0.90

### 6.5 Generate secrets
`make generate-secrets` already run in prior session. All secrets populated (no CHANGE_ME remaining):
- POSTGRES_PASSWORD, REDIS_PASSWORD, N8N_ENCRYPTION_KEY, N8N_JWT_SECRET, WEBUI_SECRET_KEY, GRAFANA_ADMIN_PASSWORD

### Image version upgrades applied
Images bumped to latest stable versions per web research (Jul 2026):
- n8n: 2.30.8 → 2.31.5
- Prometheus: v2.54.1 → v3.13.1
- Grafana: 11.3.0 → 13.0.3
- DCGM Exporter: `:latest` (stale) → pinned `4.8.3`

Note: Promtail 3.6.11 is EOL (Mar 2026). Migrate to Grafana Alloy during Step 10 hardening.

## Model Status

- **Primary:** Qwen3-VL 235B-A22B AWQ — 42/42 safetensors files, checksums verified ✅
- **Secondary:** Llama 4 Scout — config only, weights deferred (post-Step-5)

## Next step

Proceed to **Step 7** — Deploy the stack: `docker compose up -d`
