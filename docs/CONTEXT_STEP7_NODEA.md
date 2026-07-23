# Step 7 Context — Node A (node1) Deploy the Stack

## Current State

Steps 1-6 complete on Node A. Step 7 now completed.

## System Identity

- **Hostname:** node1
- **IP:** 192.168.100.119
- **GPU:** 2× NVIDIA H200 NVL 141GB (PCIe, no NVLink), driver 580.159.03
- **RAID10:** 8× 7.6 TB NVMe at `/dev/md0` → `/data` (28T available)
- **Docker:** Engine 29.6.2, Compose v5.3.1
- **Internet:** Available

## Step 7 — What Was Done

### 7.1 First deploy attempt

```bash
cd /data/stack/inference-cluster-stack
docker compose up -d
```

Several services failed health checks on first attempt. Diagnosed each issue and applied fixes to `docker-compose.yml`.

### 7.2 Issues encountered and fixes applied

#### Issue 1: Postgres failed to start — wrong data directory

PostgreSQL 18 Alpine changed the default data directory layout. The container expected data at `/var/lib/postgresql` but the volume mount pointed to `/var/lib/postgresql/data`, which confused the init sequence.

**Fix:** Changed volume mount from `./data/postgres:/var/lib/postgresql/data` to `./data/postgres:/var/lib/postgresql`.

**Gotcha:** Postgres 18+ Alpine images place the `data` subdirectory under `/var/lib/postgresql` automatically. Mounting to `/var/lib/postgresql/data` creates a nested `/var/lib/postgresql/data/data` on init. This may affect future Postgres version upgrades.

#### Issue 2: Qdrant health check failed — no wget in image

The Qdrant `v1.18.3` image does not include `wget`, causing the `CMD-SHELL wget --spider` health check to fail immediately.

**Fix:** Replaced health check with a bash TCP connection test:
```yaml
test: ["CMD", "/bin/bash", "-c", "exec 3<>/dev/tcp/localhost/6333"]
```

**Gotcha:** This relies on `/bin/bash` being present in the image (it is in Qdrant's Ubuntu-based image). Avoid `wget`-based health checks for images that may be distroless or Alpine-based without wget installed.

#### Issue 3: vLLM health check failed — no wget in image

The `vllm/vllm-openai` image does not include `wget` but does include `curl`.

**Fix:** Changed health check from `CMD-SHELL wget --spider` to `CMD curl -sf`:
```yaml
test: ["CMD", "curl", "-sf", "http://localhost:8000/health"]
```

#### Issue 4: vLLM Qwen3-VL crash with PP>1

The stock `vllm/vllm-openai:latest` (v0.25.1) crashes when loading Qwen3-VL MoE models with `--pipeline-parallel-size 2`. This is tracked in vLLM PR #43272.

**Fix:** Switched to a patched image built specifically for this setup:
```yaml
image: vllm/vllm-openai:patched-qwen3vl-pp
```

This custom image incorporates the fix from vLLM PR #43272 and is built from the same v0.25.1 base. The Dockerfile and patch script are in the repo at `inference-cluster-stack/patches/` for rebuilding on Node B.

#### Issue 5: Loki health check failed — distroless image

The `grafana/loki:3.7.3` image is distroless — no shell, no wget, no curl. The `CMD-SHELL` health check cannot execute.

**Fix:** Disabled Loki's health check entirely:
```yaml
healthcheck:
  disable: true
```

Since Promtail no longer depends on Loki's health, the `depends_on` condition for Promtail was also changed.

#### Issue 6: Open WebUI missing QDRANT_URI env var

Open WebUI v0.10.2 requires both `QDRANT_URL` and `QDRANT_URI` environment variables to connect to Qdrant. The repo template only had `QDRANT_URL`.

**Fix:** Added `QDRANT_URI: http://qdrant:6333` to the open-webui environment.

#### Issue 8: Open WebUI health check — no wget

Same as vLLM — the open-webui image has `curl` but not `wget`.

**Fix:** Changed health check from `CMD-SHELL wget --spider` to `CMD curl -sf`.

#### Issue 7: vLLM model volume mount simplified

Changed from binding the model directory at its host path (`${MODEL_DIR}:${MODEL_DIR}`) to mounting it at a fixed `/models` path inside the container. This simplifies config and makes `MODEL_NAME` paths predictable (`/models/<model-name>`).

### 7.3 Fixes applied to docker-compose.yml

| Service | Change | Reason |
|---|---|---|
| postgres | Volume: `/var/lib/postgresql/data` → `/var/lib/postgresql` | Postgres 18 Alpine layout change |
| qdrant | Health check: wget → bash TCP | No wget in image |
| vllm | Image: `latest` → `patched-qwen3vl-pp` | Qwen3-VL MoE PP>1 crash (PR #43272) |
| vllm | Volume: `${MODEL_DIR}:${MODEL_DIR}` → `${MODEL_DIR}:/models` | Simpler fixed mount point |
| vllm | Health check: wget → curl | No wget in image, curl present |
| loki | Health check: disabled (distroless image) | No shell/wget/curl |
| promtail | depends_on: `service_healthy` → `service_started` | Loki health disabled |
| open-webui | Added `QDRANT_URI: http://qdrant:6333` | Required by open-webui v0.10.2 |
| open-webui | Health check: wget → curl | No wget in image |

### 7.4 Build the patched image on Node B

For Node B, the patched image must be built before `docker compose up -d`:

```bash
cd /data/stack/inference-cluster-stack
docker build -t vllm/vllm-openai:patched-qwen3vl-pp -f patches/vllm-qwen3-vl-pp-fix.dockerfile .
```

The Dockerfile and Python patch script live in `patches/` in the stack directory — they are copied over with the rest of the stack in Step 6 and will be present on both nodes.

### 7.5 Final deploy

After applying all fixes:

```bash
docker compose up -d
```

All 13 services started and passed health checks (Loki excluded — disabled intentionally).

### 7.6 Verify

```bash
make health
```

Output:
```
postgres              healthy
redis                 healthy
qdrant                healthy
vllm                  healthy
dcgm-exporter         healthy
n8n                   healthy
open-webui            healthy
prometheus            healthy
grafana               healthy
loki                  disabled (healthcheck)
promtail              healthy
node-exporter         healthy
alertmanager          healthy
```

## Model Status

- **Primary:** Qwen3-VL 235B-A22B AWQ — loaded and serving at `http://node1:8000/v1`
- **Secondary:** Llama 4 Scout — weights on disk, ready to switch via `.env` `MODEL_NAME`

## Next step

Proceed to **Step 8** — Verify the deployment.
