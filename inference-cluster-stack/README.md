# Air-Gapped Inference Stack (Identical Node Blueprint)

Docker Compose stack for **identical standalone inference nodes** in separate datacenters.
Each node: 2× NVIDIA H200 141GB (PCIe), Pipeline Parallelism PP=2, local monitoring.

## Folder Structure

```
inference-cluster-stack/
├── docker-compose.yml     # Full stack — identical on every node
├── .env.template           # Template — copy to .env and customize
├── prometheus/
│   └── prometheus.yml     # Local-only scrape config (localhost targets)
├── grafana/
│   └── data/              # Grafana persistent storage
└── models/                # Placeholder — point .env MODEL_DIR to bulk SSD mount
```

## Usage (identical on every node)

```bash
cd inference-cluster-stack
# Edit .env — set NODE_NAME and MODEL_NAME
docker compose up -d
```

## Before First Run

1. Edit `.env` — set `NODE_NAME` to your server's hostname.
2. Point `MODEL_DIR` to your model weights path (bulk SSDs).
3. Choose your model in `MODEL_NAME` (`/models/llama-4-scout` or `/models/qwen3-vl-235b`).
4. Pre-pull all images while still online:
   ```bash
   docker compose pull
   ```

## Notes

- vLLM runs with **Pipeline Parallelism (PP=2)** for PCIe-connected GPUs. See `.env` TP=1 / PP=2.
- The stack is identical on all nodes — no role split, no profiles.
- Switch models by changing `MODEL_NAME` in `.env` and `docker compose up -d vllm --force-recreate`.
- Grafana default login: admin/admin (change immediately).
- Import Grafana dashboard ID 25261 for DCGM GPU metrics.
- `VLLM_NO_USAGE_STATS=1` and `DO_NOT_TRACK=1` disable telemetry.
