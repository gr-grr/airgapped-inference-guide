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
| 1 | OS baseline and system prep | ✅ | ⬜ |
| 2 | Storage setup (RAID10 + XFS) | ✅ | ⬜ |
| 3 | NVIDIA driver and CUDA | ✅ | ⬜ |
| 4 | Docker and NVIDIA Container Toolkit | ✅ | ⬜ |
| 5 | Model selection and download | ✅ | ⬜ |
| 6 | Stack setup and per-node config | ✅ | ⬜ |
| 7 | Deploy the stack | ✅ | ⬜ |
| 8 | Verify the deployment | ⬜ | ⬜ |
| 9 | Adding a new model later | 🔲 | 🔲 |
| 10 | Pre-cutoff hardening | ⬜ | ⬜ |
| 11 | Air-gap enforcement | ⬜ | ⬜ |

---

## What exists so far

These artifacts exist, and Node A hardware is now provisioned through Step 6 (Step 6 complete):

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
| `docs/CONTEXT_STEP5_NODEA.md` | Session context: model research, CLI install, next actions for Node A |

---

## What needs to happen (both nodes)

### Per node — Steps 1-4 (OS, storage, drivers, Docker)
1. Install Ubuntu 26.04, set hostname (`node1` / `node2`), disable nouveau
2. Create RAID10 from 8× NVMe, format XFS, mount at `/data`
   ⚠️ **After RAID creation:** overwrite `/etc/mdadm/mdadm.conf` (don't append),
      rebuild initramfs. See `docs/nvme-device-shuffle-raid-boot-failure.md`.
3. Install NVIDIA driver 580-server + CUDA 13-3
4. Install Docker CE + nvidia-container-toolkit

### Per node — Steps 5-6 (models + stack config)
5. Download model weights → `data/models/`
   - **Decision executed:** Qwen3-VL-235B-A22B at AWQ (INT4) as primary, Llama 4 Scout FP8 as secondary
   - ✅ Qwen3-VL 235B-A22B AWQ downloaded (121 GB, 42/42 files) and checksums verified
   - Llama 4 Scout download deferred (secondary, post-Step-5)
6. ✅ Stack dir in place, data dirs created, `.env` configured for node1, secrets generated, `docker compose pull` completed (all 13 images latest stable)

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
| `docs/nvme-device-shuffle-raid-boot-failure.md` | Incident report: NVMe device name shuffle after reboot causing RAID boot failure |
