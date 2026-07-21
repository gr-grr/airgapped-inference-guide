# Project context for AI coding agents

## What this is

Blueprint/runbook for two identical standalone LLM inference nodes, each with 2× NVIDIA H200 141GB GPUs (PCIe, no NVLink), running in separate datacenters with **no network link between them**.

## Key architectural decisions (hard constraints)

- **Parallelism**: Pipeline Parallelism PP=2 (not TP) because H200s connect via PCIe, not NVLink
- **No role split**: Both nodes run the identical full stack — no control-plane/worker
- **No cross-node networking**: Each node fully standalone, no shared services
- **Deployment**: Step-by-step runbook followed identically at each DC; no deployment bundle
- **Storage**: 2× 480 GB NVMe → RAID1 (OS), 8× 7.6 TB NVMe → RAID10 (models + data under stack)

## Models

- **Primary**: Llama 4 Scout 109B (17B active, MoE 16 experts) — FP8 by default
- **Secondary**: Qwen3-VL 235B-A22B (22B active, MoE) — FP8 or Q4
- Switchable via `MODEL_NAME` in `.env` + `docker compose up -d vllm --force-recreate`
- New models: download on internet machine, USB transfer to node, update `.env`

## Stack layout (`inference-cluster-stack/`)

```
inference-cluster-stack/
├── config/
│   ├── prometheus/          # prometheus.yml, rules/, alertmanager.yml
│   ├── promtail/            # promtail.yml
│   └── postgres/init/       # Init SQL scripts
├── data/                    # Runtime data (per-service subdirs)
│   ├── models/              # Model weights (MODEL_DIR)
│   ├── prometheus/          # Prometheus TSDB
│   ├── grafana/             # Grafana data
│   ├── postgres/            # PostgreSQL database files
│   ├── redis/               # Redis AOF + RDB snapshots
│   ├── qdrant/              # Qdrant vector storage
│   ├── n8n/                 # n8n workflow data + credentials
│   ├── open-webui/          # Open WebUI sessions, chats, Whisper models
│   ├── loki/                # Loki log index + chunks
│   └── alertmanager/        # Alertmanager notification state
├── backups/                 # Backup artifacts
├── docker-compose.yml       # 13 services: vllm, dcgm-exporter, postgres, redis,
│                           #   qdrant, n8n, open-webui, prometheus, grafana,
│                           #   loki, promtail, alertmanager
├── .env.template            # All config vars — copy to .env, never commit .env
├── .env                     # Gitignored, per-node config
├── Makefile                 # Operational helpers: health, backup, secrets
└── README.md
```

First service in compose is vllm (PP=2 via `--pipeline-parallel-size ${PIPELINE_PARALLEL_SIZE}`). All services start with `docker compose up -d`.

Data volumes use relative paths (`./data/{service}`) resolved from the compose file. Config volumes use relative paths (`./config/{service}`). The stack is designed to live at `/data/stack/inference-cluster-stack/` on the RAID10 bulk array.

## RUNBOOK.md — 11-step runbook

| Step | Title | Notes |
|---|---|---|
| 1 | OS baseline and system prep | Hostname, apt, base packages, disable nouveau |
| 2 | Storage setup (RAID10 + XFS) | mdadm RAID10, mkfs.xfs, fstab, directory tree |
| 3 | NVIDIA driver and CUDA | nvidia-driver-580-server, cuda-toolkit-13-3 |
| 4 | Docker and NVIDIA Container Toolkit | docker-ce + compose-plugin + nvidia-container-toolkit |
| 5 | Model selection and download | Download on internet machine, USB transfer to server, checksum |
| 6 | Stack setup and per-node config | Copy stack dir, create data dirs, `docker compose pull`, configure `.env`, `make generate-secrets` |
| 7 | Deploy the stack | `docker compose up -d` — starts all 13 services |
| 8 | Verify the deployment | curl checks for vLLM, Open WebUI, n8n, Qdrant, DCGM, Prometheus, Grafana, Loki, Alertmanager |
| 9 | Adding a new model later | Post-cutoff: internet machine → USB → server → `.env` → restart |
| 10 | Pre-cutoff hardening | Disable auto-updates, save images as tar, snapshot state |
| 11 | Air-gap enforcement | WAN disconnect, verify offline recovery |

## Deployment workflow

- **Follow `RUNBOOK.md` sequentially step by step** — do not skip ahead
- **Keep `TRACKER.md` updated** as you go: mark steps ✅ done, ◐ partial, ⬜ not started
- Wait for user confirmation after each step before proceeding to the next
- When completing a step, update TRACKER.md and summarize what was done + any issues
- If a step fails or needs deviation, mark it ◐, note the issue in TRACKER.md, and ask the user

## Before editing

- Keep `.env` in `.gitignore` — never commit secrets
- When editing RUNBOOK.md, preserve the 11-step numbered structure and checklist format
- All env vars referenced in docker-compose.yml must exist in `.env.template`
- Grafana dashboard ID 25261 (DCGM) is the standard reference
- Telemetry is disabled via `VLLM_NO_USAGE_STATS=1` and `DO_NOT_TRACK=1`


## Golden rules

- **DON'T EVER ASSUME** — always rely on up-to-date web resources (hardware specs, CUDA versions, driver versions, model info, Docker image tags)
- **Plan first, then wait for user review and approval** before executing any multi-step changes