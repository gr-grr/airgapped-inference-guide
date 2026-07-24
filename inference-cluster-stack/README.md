# Air-Gapped Inference Stack (Identical Node Blueprint)

Docker Compose stack for **identical standalone inference nodes** in separate datacenters.
Each node: 2× NVIDIA H200 141GB (PCIe), Pipeline Parallelism PP=2, local monitoring.

## Folder Structure

```
inference-cluster-stack/
├── config/
│   ├── prometheus/          # prometheus.yml, rules/, alertmanager.yml
│   ├── promtail/            # promtail.yml
│   └── postgres/init/       # Init SQL scripts
├── data/
│   ├── models/              # Model weights (MODEL_DIR)
│   ├── postgres/            # PostgreSQL database files
│   ├── redis/               # Redis AOF + RDB snapshots
│   ├── qdrant/              # Qdrant vector storage
│   ├── n8n/                 # n8n workflow data + credentials
│   ├── open-webui/          # Sessions, chats, Whisper models
│   ├── prometheus/          # Prometheus TSDB
│   ├── grafana/             # Grafana data
│   ├── loki/                # Loki log index + chunks
│   └── alertmanager/        # Alertmanager notification state
├── backups/                 # Backup artifacts
├── docker-compose.yml       # Full stack — identical on every node
├── .env.template            # Template — copy to .env and customize
├── Makefile                 # Operational helpers
└── README.md
```

## Usage (identical on every node)

```bash
cd /data/stack/inference-cluster-stack
# Edit .env — set NODE_NAME and MODEL_NAME
docker compose up -d
```

## Before First Run

1. Edit `.env` — set `NODE_NAME` to your server's hostname.
2. Point `MODEL_DIR` to your model weights path (`/data/stack/inference-cluster-stack/data/models`).
3. Choose your model in `MODEL_NAME` (`/models/qwen3-vl-30b-a3b-awq`).
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

## Configuration Rationale (July 2026)

### Why

The original config (Qwen3-VL-235B-A22B AWQ, max-num-seqs=80, 8K context) hit 90% GPU
utilization under load. Target requirements grew to 250 concurrent users with 46-64K
context windows per request, which the 235B model cannot support on 2× H200 141GB —
KV cache alone would need ~750 GB.

### What we changed

| Change | From | To |
|---|---|---|
| Model | Qwen3-VL-235B-A22B AWQ (118 GB) | Qwen3-VL-30B-A3B AWQ (17 GB) |
| Context length | 8,192 | 65,536 |
| Max concurrent sequences | 80 | 250 |
| GPU memory utilization | 0.90 | 0.95 |
| KV cache dtype | BF16 (2B/elem) | FP8 (1B/elem) |
| CPU swap space | none | 350 GB |
| Scheduler policy | default | guaranteed |

### Benefits gained

- **17× less weight memory per GPU** (~59 GB → ~3.5 GB) → frees ~55 GB for KV cache
- **8× longer context** (8K → 64K) supports the 46K single-request workflow
- **3× more concurrent slots** (80 → 250) with 84 fitting entirely on-GPU
- ~126 GB per GPU available for KV cache at FP8 precision
- Remaining 166 sequences backed by CPU swap over PCIe (16-token block granularity)
- Native 256K context — no rope scaling needed for 64K

### What we lost

- **Model capability**: 235B → 30B is a significant reduction in total parameters
  (235B total / 22B active → 30.5B total / 3.3B active). Benchmarks show this is
  still strong for multilingual OCR, tool use, and reasoning, but quality on the
  hardest tasks (long-tail knowledge, complex multi-step reasoning) will be lower.
- **No video support** (`--limit-mm-per-prompt.video 0`) to save memory
- **Swap-dependent throughput**: active decode batch limited to ~84 sequences;
  remaining 166 incur PCIe block migration when scheduled
- **FP8 KV cache** has slightly lower accuracy than BF16 (typically <0.5% on perplexity)
