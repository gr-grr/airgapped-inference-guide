# Air-Gapped AI Inference Blueprint

Two identical standalone inference nodes, each with 2× NVIDIA H200 141GB GPUs (PCIe, no NVLink), in separate datacenters with no network link between them.

- **[RUNBOOK.md](./RUNBOOK.md)** — 10-step operational runbook from fresh Ubuntu install through air-gap enforcement
- **[inference-cluster-stack/](./inference-cluster-stack/)** — Docker Compose stack (vLLM PP=2, DCGM, Prometheus, Grafana)
- **[AGENTS.md](./AGENTS.md)** — Project context for AI coding agents
