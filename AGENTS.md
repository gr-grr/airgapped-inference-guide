# Project context for AI coding agents

## What this is

Blueprint/runbook for two identical standalone LLM inference nodes, each with 2× NVIDIA H200 141GB GPUs (PCIe, no NVLink), running in separate datacenters with **no network link between them**.

## Key architectural decisions (hard constraints)

- **Parallelism**: Pipeline Parallelism PP=2 (not TP) because H200s connect via PCIe, not NVLink
- **No role split**: Both nodes run the identical full stack — no control-plane/worker
- **No cross-node networking**: Each node fully standalone, no shared services
- **Deployment**: Step-by-step runbook followed identically at each DC; no deployment bundle

## Models

- **Primary**: Llama 4 Scout 109B (17B active, MoE 16 experts) — FP8 by default
- **Secondary**: Qwen3-VL 235B-A22B (22B active, MoE) — FP8 or Q4
- Switchable via `MODEL_NAME` in `.env` + `docker compose up -d vllm --force-recreate`
- New models: download on internet machine, USB transfer to node, update `.env`

## Stack layout (`inference-cluster-stack/`)

```
inference-cluster-stack/
├── docker-compose.yml      # Services: vllm, dcgm-exporter, prometheus, grafana
├── .env.template            # All config vars — copy to .env, never commit .env
├── .env                     # Gitignored, per-node config
├── prometheus/
│   └── prometheus.yml       # Local-only scraping (localhost targets)
└── grafana/
    └── data/                # Persistent storage (runtime-created)
```

First service in compose is vllm (PP=2 via `--pipeline-parallel-size ${PIPELINE_PARALLEL_SIZE}`). All services start with `docker compose up -d`.

## README — 10-step runbook

| Step | Title | Notes |
|---|---|---|
| 1 | OS baseline and system prep | Hostname, apt, base packages, disable nouveau |
| 2 | NVIDIA driver and CUDA | nvidia-driver-580-server, cuda-toolkit-12-8 |
| 3 | Docker and NVIDIA Container Toolkit | docker-ce + compose-plugin + nvidia-container-toolkit |
| 4 | Model selection and download | Download on internet machine, USB transfer to server, checksum |
| 5 | Stack setup and per-node config | Copy stack dir, `docker compose pull`, configure `.env` |
| 6 | Deploy the stack | `docker compose up -d` — starts all 4 services |
| 7 | Verify the deployment | curl checks for vLLM, DCGM, Prometheus, Grafana |
| 8 | Adding a new model later | Post-cutoff: internet machine → USB → server → `.env` → restart |
| 9 | Pre-cutoff hardening | Disable auto-updates, save images as tar, snapshot state |
| 10 | Air-gap enforcement | WAN disconnect, verify offline recovery |

## Before editing

- Keep `.env` in `.gitignore` — never commit secrets
- When editing README, preserve the 10-step numbered structure and checklist format
- All env vars referenced in docker-compose.yml must exist in `.env.template`
- Grafana dashboard ID 25261 (DCGM) is the standard reference
- Telemetry is disabled via `VLLM_NO_USAGE_STATS=1` and `DO_NOT_TRACK=1`


## Golden rules

- **DON'T EVER ASSUME** — always rely on up-to-date web resources (hardware specs, CUDA versions, driver versions, model info, Docker image tags)
- **Plan first, then wait for user review and approval** before executing any multi-step changes