# Deployment Tracker — Air-Gapped Inference Nodes

## Legend

| Symbol | Meaning |
|--------|---------|
| ✅ Done | Verified complete |
| ◐ Partial | In progress or partially done |
| ⬜ Not started | Not yet touched |
| 🔲 N/A | Not applicable yet |

---

## Runbook Progress (11-step runbook)

All steps are **not started** on both nodes. This is a deployment blueprint — no hardware has been provisioned yet.

| Step | Title | Node A (node1) | Node B (node2) |
|------|-------|----------------|----------------|
| 1 | OS baseline and system prep | ⬜ | ⬜ |
| 2 | Storage setup (RAID10 + XFS) | ⬜ | ⬜ |
| 3 | NVIDIA driver and CUDA | ⬜ | ⬜ |
| 4 | Docker and NVIDIA Container Toolkit | ⬜ | ⬜ |
| 5 | Model selection and download | ⬜ | ⬜ |
| 6 | Stack setup and per-node config | ⬜ | ⬜ |
| 7 | Deploy the stack | ⬜ | ⬜ |
| 8 | Verify the deployment | ⬜ | ⬜ |
| 9 | Adding a new model later | 🔲 | 🔲 |
| 10 | Pre-cutoff hardening | ⬜ | ⬜ |
| 11 | Air-gap enforcement | ⬜ | ⬜ |

---

## What exists so far (blueprint/runbook only)

These are the planning artifacts — no hardware has been provisioned:

| Artifact | Purpose |
|----------|---------|
| `RUNBOOK.md` | Full 11-step runbook for both nodes |
| `AGENTS.md` | AI agent context — architecture decisions, constraints |
| `NODE_A_INFO.md` | Hardware inventory template for Node A (reference spec) |
| `NODE_SETUP_USER.md` | Recipe for nvm/Node.js/OpenCode install |
| `README.md` | Project overview |
| `inference-cluster-stack/docker-compose.yml` | 13-service Compose definition |
| `inference-cluster-stack/.env.template` | Config template with all env vars |
| `inference-cluster-stack/.env` | Placeholder (needs per-node values at deploy time) |
| `inference-cluster-stack/Makefile` | Operational helpers (health, backup, secrets) |
| `inference-cluster-stack/config/` | Prometheus, promtail, postgres configs |
| `inference-cluster-stack/data/` | Runtime data directories (empty, ready for deployment) |

---

## What needs to happen (both nodes)

### Per node — Steps 1-4 (OS, storage, drivers, Docker)
1. Install Ubuntu 26.04, set hostname (`node1` / `node2`), disable nouveau
2. Create RAID10 from 8× NVMe, format XFS, mount at `/data`
3. Install NVIDIA driver 580-server + CUDA 13-3
4. Install Docker CE + nvidia-container-toolkit

### Per node — Steps 5-6 (models + stack config)
5. Download model weights (Llama 4 Scout + Qwen3-VL) → `data/models/`
6. Copy stack, create data dirs, configure `.env`, run `make generate-secrets`

### Per node — Steps 7-8 (deploy + verify)
7. `docker compose up -d` → start all 13 services
8. Health checks: vLLM, Open WebUI, n8n, Qdrant, Prometheus, Grafana, Loki, Alertmanager

### Per node — Steps 10-11 (harden + air-gap)
9. Disable auto-updates, telemetry, snapshot state
10. WAN disconnect, reboot, verify full offline recovery

---

## Key files reference

| File | Purpose |
|------|---------|
| `AGENTS.md` | AI agent context — architecture decisions, constraints, golden rules |
| `RUNBOOK.md` | Complete 11-step runbook for both nodes |
| `NODE_A_INFO.md` | Node A hardware/software inventory |
| `NODE_SETUP_USER.md` | nvm/Node.js/OpenCode install recipe for any node |
| `TRACKER.md` | **This file** — what's done, where, and what's left |
| `inference-cluster-stack/README.md` | Stack-specific documentation |
| `inference-cluster-stack/docker-compose.yml` | 13-service Compose file |
| `inference-cluster-stack/.env` | Per-node config (gitignored — never commit) |
| `inference-cluster-stack/.env.template` | Config template with all env vars |
| `inference-cluster-stack/Makefile` | Operational commands |
