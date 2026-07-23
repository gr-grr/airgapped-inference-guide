# Air-Gapped AI Inference Blueprint — Identical Nodes, Separate Datacenters

## Goal and scope

This guide packages our deployment plan for **two identical standalone inference nodes**, each with 2× NVIDIA H200 141GB GPUs connected by **PCIe** (no NVLink), running in **separate datacenters** with no network link between them.

The architecture changed from the original plan:
- **Intra-node**: 2× H200 on PCIe → use **Pipeline Parallelism (PP=2)** instead of Tensor Parallelism (TP), because PP has lower inter-GPU communication overhead on PCIe
- **Inter-node**: No 10GbE link between datacenters. Each node is fully independent.
- **Goal**: Both nodes run the **exact same stack** — same configs, same images, same model weights — via the same reproducible procedure.

Use this as the operational runbook from fresh Ubuntu install → internet-connected setup → identical deployment at each DC → final hardening and air-gap.

## Infrastructure summary

### Per server (both nodes identical)
- 2× NVIDIA H200, 141GB HBM3e each
- PCIe interconnect (no NVLink between the two GPUs)
- 2× Intel Xeon 6737P, 64 cores / 128 threads
- Ubuntu Server 26.04 LTS
- 8× 7.6TB NVMe SSDs → RAID10 → `/data` (XFS) for model weights + persistent data
- 2× 480GB NVMe → RAID1 → `/` (ext4) for OS + Docker

### Deployment model
- 2 physical servers, one per datacenter
- No network link between them
- Each node is air-gapped independently
- Identical open-source model serving stack on both

## Architecture decision

### Intra-node: PCIe → Pipeline Parallelism (PP)
The two H200s inside each node connect via **PCIe Gen5 ×16** (~128 GB/s bidirectional), not NVLink (900 GB/s). Tensor Parallelism (TP) requires an all-reduce after every transformer layer, which becomes a bottleneck on PCIe. Scaling efficiency for TP on PCIe is roughly 70-78% per additional GPU.

**Pipeline Parallelism (PP=2)** is the recommended strategy for PCIe-connected GPUs. It splits the model's layers sequentially across the two GPUs:
- GPU 0 processes layers 0-N, GPU 1 processes layers N+1 to end
- Inter-GPU communication is limited to activation tensors between stages (~67 MB per pass vs. 160+ all-reduce ops per token for TP)
- For **multi-user API workloads** (high concurrency), the idle pipeline bubbles get filled with queued requests, making PP throughput-efficient
- vLLM's own documentation recommends PP over TP for non-NVLink multi-GPU setups

### Per-node stack (both nodes identical)
Both nodes run the **full stack** — there is no control-plane/worker split:
- vLLM (PP=2 across both GPUs)
- DCGM exporter
- Prometheus + Grafana (local monitoring only)

### Why no cross-node networking
The two nodes are in separate datacenters with no direct link:
- No cross-node Prometheus scraping
- No rsync of model weights
- Each node is independently provisioned and air-gapped

---

## Step 1 — OS baseline and system prep

### 1.1 Set hostname
On the node at DC A:
```bash
sudo hostnamectl set-hostname node1
```

On the node at DC B:
```bash
sudo hostnamectl set-hostname node2
```

### 1.2 Update the OS while internet is still available
```bash
sudo apt update && sudo apt full-upgrade -y
```

### 1.3 Install base packages
```bash
sudo apt install -y build-essential dkms curl wget gnupg lsb-release ca-certificates software-properties-common ethtool iperf3 rsync git python3-pip
```

### 1.4 Disable nouveau
```bash
sudo bash -c "echo 'blacklist nouveau' > /etc/modprobe.d/blacklist-nouveau.conf"
sudo bash -c "echo 'options nouveau modeset=0' >> /etc/modprobe.d/blacklist-nouveau.conf"
sudo update-initramfs -u
sudo reboot
```

### 1.5 Verify nouveau is gone
```bash
lsmod | grep nouveau
```
Expected result: no output.

---

## Step 2 — Storage setup (RAID10 + XFS)

### 2.1 Identify the drives

The server has 10 NVMe drives. Two are small (480 GB each, for OS — already configured in Step 1 as RAID1). The other 8 are large (7.6 TB each) for models and persistent data.

```bash
lsblk
```

Expected output (8 drives ~7.6T):
```
nvme0n1    480G  ─┐
nvme2n1    480G  ─┤  OS drives (already in RAID1 from Step 1)
                  │
nvme1n1    7.6T  ─┐
nvme3n1    7.6T   │
nvme4n1    7.6T   │
nvme5n1    7.6T   ├── Bulk storage → RAID10
nvme6n1    7.6T   │
nvme7n1    7.6T   │
nvme8n1    7.6T   │
nvme9n1    7.6T  ─┘
```

### 2.2 Create RAID10 array

Combine all 8 drives into a single RAID10 array. RAID10 mirrors pairs of drives then stripes across pairs — it survives one drive failure per pair without data loss.

```bash
sudo mdadm --create /dev/md0 --level=10 --raid-devices=8 \
  /dev/nvme1n1 /dev/nvme3n1 /dev/nvme4n1 /dev/nvme5n1 \
  /dev/nvme6n1 /dev/nvme7n1 /dev/nvme8n1 /dev/nvme9n1
```

Monitor the initial sync:

```bash
watch cat /proc/mdstat
```

Wait until the array shows `UUUUUUUU` (all 8 drives active) and resync is complete.

### 2.3 Create and mount filesystem (XFS)

XFS is chosen for the bulk array — it handles large sequential reads (model weights) and multi-TB volumes better than ext4.

```bash
sudo mkfs.xfs /dev/md0
sudo mkdir -p /data
sudo mount /dev/md0 /data
```

### 2.4 Persistent mount via fstab

```bash
echo "UUID=$(sudo blkid -s UUID -o value /dev/md0) /data xfs defaults,noatime 0 0" | sudo tee -a /etc/fstab
```

### 2.5 Save RAID configuration

**Important: overwrite, don't append.** Using `tee -a` can accumulate stale
entries from previous arrays (including IMSM containers from vendor RAID),
which can cause boot failures if NVMe device names change across reboots.
See [`docs/nvme-device-shuffle-raid-boot-failure.md`](docs/nvme-device-shuffle-raid-boot-failure.md)
for the full incident report.

```bash
sudo bash -c 'echo "# mdadm.conf – overwritten by setup" > /etc/mdadm/mdadm.conf && mdadm --detail --scan >> /etc/mdadm/mdadm.conf'
sudo update-initramfs -u
```

### 2.6 Create directory tree

```bash
sudo mkdir -p /data/stack
```

| Directory | Purpose |
|---|---|
| `/data/stack` | Root for `inference-cluster-stack/` and all its data |

Docker itself stays on the OS RAID1 at `/var/lib/docker` — the 480 GB OS drives have ample room for ~10 GB of images, and keeping Docker on the OS avoids adding another dependency on the bulk array.

### 2.7 Verify

```bash
df -h /data
```

Expected:
```
Filesystem      Size  Used Avail Use% Mounted on
/dev/md0        30T   XX   30T    X% /data
```

**Before the next reboot:** confirm `/etc/mdadm/mdadm.conf` contains only
current arrays (no stale entries from old or vendor RAID configs). NVMe device
names can change across reboots — the UUID-based references in
`mdadm --detail --scan` handle this correctly, but stale entries can confuse
assembly. See [`docs/nvme-device-shuffle-raid-boot-failure.md`](docs/nvme-device-shuffle-raid-boot-failure.md).

---

## Step 3 — NVIDIA driver and CUDA

### 3.1 Add NVIDIA CUDA repository
```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2604/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
```

### 3.2 Install the datacenter driver
```bash
sudo apt search nvidia-driver
sudo apt install -y nvidia-driver-580-server
```

### 3.3 Install CUDA toolkit
```bash
sudo apt install -y cuda-toolkit-13-3
```

### 3.4 Reboot and verify
```bash
sudo reboot
nvidia-smi
```
Expected result: both H200 GPUs visible.

### 3.5 Verify local GPU topology
```bash
nvidia-smi topo -m
```
Expected: the two H200s show a PCIe connection, not NVLink. The `nvlink` flag will confirm no NVLink links exist — this is expected and confirms our PP strategy is correct.

---

## Step 4 — Docker and NVIDIA Container Toolkit

### 4.1 Remove conflicting packages
```bash
for pkg in docker.io docker-doc docker-compose docker-compose-v2 podman-docker containerd runc; do sudo apt-get remove -y $pkg; done
```

### 4.2 Add Docker repository
```bash
sudo apt-get update
sudo apt-get install -y ca-certificates curl
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc
sudo tee /etc/apt/sources.list.d/docker.sources <<EOF
Types: deb
URIs: https://download.docker.com/linux/ubuntu
Suites: $(. /etc/os-release && echo "${UBUNTU_CODENAME:-$VERSION_CODENAME}")
Components: stable
Architectures: $(dpkg --print-architecture)
Signed-By: /etc/apt/keyrings/docker.asc
EOF
sudo apt-get update
```

### 4.3 Install Docker
```bash
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
```

### 4.4 Add your user to docker group
```bash
sudo groupadd docker 2>/dev/null
sudo usermod -aG docker $USER
newgrp docker
```

### 4.5 Install NVIDIA Container Toolkit
```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list |   sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' |   sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

### 4.6 Verify GPU access in containers
```bash
docker run --rm --gpus all nvidia/cuda:13.3.0-base-ubuntu24.04 nvidia-smi
```

---

## Step 5 — Model selection and download

### 5.1 Serving engine
Use **vLLM** with **Pipeline Parallelism (PP=2)** to split the model across the two PCIe-connected H200s.

### 5.2 Model candidates
We deploy **both** models for comparison:

| Model | Total params | Active params | Architecture | Context | Multimodal | Tool use |
|---|---|---|---|---|---|---|
| **Llama 4 Scout** | 109B | 17B | MoE (16 experts) | 10M tokens | Text + Image | Yes |
| **Qwen3-VL 235B-A22B** | 235B | 22B | MoE | 256K native, 1M extended | Text + Image + Video | Yes |

### 5.3 Memory budget per model

**Qwen3-VL 235B-A22B at AWQ (INT4)** — recommended primary for multi-user:
```
Total GPU memory:     282 GB (2 × 141 GB)
Model weights (AWQ):  ~118 GB
KV cache + overhead:  ~152 GB headroom ← excellent for multi-user
```
Best suited if you need multilingual OCR (39 languages incl. Arabic, French, English), reasoning, tool use, and agentic capabilities under Apache 2.0 license.

**Qwen3-VL 235B-A22B at FP8** (tighter, still workable):
```
Total GPU memory:     282 GB
Model weights (FP8):  ~235 GB
KV cache + overhead:  ~35 GB headroom ← tight, fewer concurrent users
```

**Llama 4 Scout at FP8** — secondary for long-context tasks:
```
Total GPU memory:     282 GB
Model weights (FP8):  ~109 GB
KV cache + overhead:  ~161 GB headroom ← excellent for multi-user + 10M context
```
Use for tasks needing extreme context length (10M tokens). Weaker on Arabic OCR and agentic tasks compared to Qwen3-VL.

**FP8** is the default precision on H200 (native Hopper FP8 tensor cores, near-lossless quality). **AWQ (INT4)** compresses weights ~2× further with minimal quality loss, freeing more memory for KV cache and multi-user concurrency — the recommended choice for multi-user API serving.

### 5.4 Model storage layout

Model weights live at `/data/stack/inference-cluster-stack/data/models` on the RAID10 bulk array, separate from the OS RAID1. The data directory will be created in Step 6 after the stack is copied.

### 5.5 Install HuggingFace CLI

Ubuntu 26.04 enforces PEP 668 (externally managed Python), so `pip install` system-wide is blocked. Use `pipx` instead:

```bash
sudo apt install -y pipx
pipx install huggingface-hub
```

The `hf` binary is at `~/.local/bin/hf`. Add it to PATH or use the full path.

**Note:** `huggingface-cli` is deprecated. Use the new `hf` CLI for all operations.

### 5.6 Authenticate

Llama 4 Scout requires gated access — accept terms at https://huggingface.co/meta-llama/Llama-4-Scout-17B-16E-Instruct first.

```bash
hf auth login --token <your-hf-token>
```

### 5.7 Download model weights

Download on a machine with internet access, then transfer to the server via USB:

```bash
# On the workstation:
# Llama 4 Scout
hf download meta-llama/Llama-4-Scout-17B-16E-Instruct \
  --local-dir ./llama-4-scout

# Qwen3-VL (AWQ quantized — recommended for multi-user)
hf download QuantTrio/Qwen3-VL-235B-A22B-Instruct-AWQ \
  --local-dir ./qwen3-vl-235b

# Package for transfer
tar czf llama-4-scout.tar.gz ./llama-4-scout
tar czf qwen3-vl-235b-awq.tar.gz ./qwen3-vl-235b
# Copy tar files to USB drive
```

If the server has temporary internet access during setup, you can download directly:

```bash
model_dir=/data/stack/inference-cluster-stack/data/models
sudo mkdir -p "$model_dir"
hf download meta-llama/Llama-4-Scout-17B-16E-Instruct \
  --local-dir "$model_dir/llama-4-scout"
hf download QuantTrio/Qwen3-VL-235B-A22B-Instruct-AWQ \
  --local-dir "$model_dir/qwen3-vl-235b"
```

### 5.8 Transfer to the server

```bash
sudo mkdir -p /data/stack/inference-cluster-stack/data/models
sudo tar xzf /path/to/usb/llama-4-scout.tar.gz -C /data/stack/inference-cluster-stack/data/models/
sudo tar xzf /path/to/usb/qwen3-vl-235b-awq.tar.gz -C /data/stack/inference-cluster-stack/data/models/
```

### 5.9 Checksum the model files

```bash
cd /data/stack/inference-cluster-stack/data/models
find llama-4-scout -type f -exec sha256sum {} \; > llama-4-scout.sha256
find qwen3-vl-235b -type f -exec sha256sum {} \; > qwen3-vl-235b.sha256
```

---

## Step 6 — Stack setup and per-node configuration

### 6.1 Copy the stack directory
The `inference-cluster-stack/` directory contains the full Docker Compose stack. Copy it to the server (via USB or however the server was provisioned):

```bash
cp -r /path/to/usb/inference-cluster-stack /data/stack
```

### 6.2 Create data directories
Create runtime data directories under the stack. These hold persistent data for each service:

```bash
mkdir -p /data/stack/inference-cluster-stack/data/{models,prometheus,grafana,postgres,redis,qdrant,n8n,open-webui,loki,alertmanager}
```

| Directory | Purpose |
|---|---|
| `data/models` | Model weight files (Llama 4 Scout, Qwen3-VL) |
| `data/prometheus` | Prometheus TSDB (metrics history) |
| `data/grafana` | Grafana data (dashboards, users, SQLite) |
| `data/postgres` | PostgreSQL database files |
| `data/redis` | Redis append-only log and snapshots |
| `data/qdrant` | Qdrant vector database storage |
| `data/n8n` | n8n workflow data and credentials |
| `data/open-webui` | Open WebUI sessions, chats, and Whisper models |
| `data/loki` | Loki log index and chunks |
| `data/alertmanager` | Alertmanager silences and notification state |

### 6.3 Pull container images
Pull all container images now while internet is available:

```bash
cd /data/stack/inference-cluster-stack
docker compose pull
```

This pulls vLLM, DCGM exporter, Postgres, Redis, Qdrant, n8n, Open WebUI, Prometheus, Grafana, Loki, Promtail, and Alertmanager images.

### 6.4 Configure environment

```bash
cp .env.template .env
```

Edit `.env` and set:
- `NODE_NAME` — e.g. `node1` (must match the hostname from Step 1.1)
- `NODE_IP` — the node's IP address on the local network (e.g. `192.168.1.100`)
- `MODEL_DIR` — path to model weights (e.g. `/data/stack/inference-cluster-stack/data/models`)
- `MODEL_NAME` — e.g. `/models/llama-4-scout`

### 6.5 Generate secrets

```bash
make generate-secrets
```

Paste the output values into `.env` — replace all `CHANGE_ME` placeholders.

### 6.6 Both nodes are identical
The two nodes run the **same stack** with the same config files and the same procedure. Each node gets its own copy of `inference-cluster-stack/` and is configured independently. The only per-node differences are:
- **Hostname** (set in Step 1.1)
- **Node IP** (`NODE_IP` in `.env`)
- **`.env` values** (node name, model path, secrets)

From this point forward, every step is executed **identically on both nodes**.

---

## Step 7 — Deploy the stack

### 7.1 Build the patched vLLM image (Qwen3-VL MoE PP>1 fix)

The stock `vllm/vllm-openai:latest` crashes with Qwen3-VL MoE models at `pipeline-parallel-size > 1`
([vLLM PR #43272](https://github.com/vllm-project/vllm/pull/43272)).
Build the patched image before starting the stack:

```bash
cd /data/stack/inference-cluster-stack
docker build -t vllm/vllm-openai:patched-qwen3vl-pp -f patches/vllm-qwen3-vl-pp-fix.dockerfile .
```

### 7.2 Start all services

```bash
cd /data/stack/inference-cluster-stack
docker compose up -d
```

> **Note:** `docker compose up -d` may fail for some services on first run due to health checks or permission issues. Common fixes documented in `docs/CONTEXT_STEP7_NODEA.md`.

### Services started

| Service | Container name | Purpose |
|---|---|---|
| Postgres | `postgres` | Database for n8n (and future services) |
| Redis | `redis` | Cache and queue backend for n8n |
| Qdrant | `qdrant` | Vector database for RAG (Open WebUI) |
| vLLM | `vllm` | LLM inference engine (PP=2 across both GPUs) |
| DCGM exporter | `dcgm-exporter` | GPU metrics (power, temperature, memory) |
| n8n | `n8n` | Workflow automation (scheduled tasks, AI pipelines) |
| Open WebUI | `open-webui` | Chat UI with RAG, document upload, Whisper STT |
| Prometheus | `prometheus` | Metrics collection (scrapes all services by name) |
| Grafana | `grafana` | Monitoring dashboards |
| Loki | `loki` | Centralized log aggregation |
| Promtail | `promtail` | Ships Docker container logs to Loki |
| Node Exporter | `node-exporter` | Host-level metrics (disk, CPU, memory) |
| Alertmanager | `alertmanager` | Alert routing (Prometheus alert evaluation) |

### Startup order
The data layer starts first (Postgres, Redis, Qdrant), then inference (vLLM, DCGM), then applications (n8n, Open WebUI), then observability (Prometheus, Grafana, Loki, Promtail, Alertmanager). Docker Compose handles this automatically via `depends_on` with health checks.

### Known gotchas

These issues were discovered during initial deployment and fixed in the repo's `docker-compose.yml`. Be aware if customizing:

- **Postgres 18+ volume path**: Mount to `/var/lib/postgresql`, not `/var/lib/postgresql/data`. The Alpine image places the `data` subdirectory automatically under the mount point.
- **Qdrant health check**: The `qurant/qdrant` image has no `wget`. Use `CMD /bin/bash -c "exec 3<>/dev/tcp/localhost/6333"` instead of `CMD-SHELL wget`.
- **vLLM health check**: The `vllm/vllm-openai` image has `curl` but not `wget`. Use `CMD curl -sf` instead of `CMD-SHELL wget`.
- **Loki health check**: The `grafana/loki` image is distroless (no shell/wget/curl). Set `healthcheck: disable: true`.
- **vLLM Qwen3-VL PP>1**: Stock vLLM crashes loading Qwen3-VL MoE with PP>1. Use patched image `vllm/vllm-openai:patched-qwen3vl-pp` (PR #43272 fix).
- **Open WebUI Qdrant**: v0.10.2 needs both `QDRANT_URL` and `QDRANT_URI` set to `http://qdrant:6333`.
- **Loki health → Promtail dependency**: If Loki health is disabled, change Promtail's `depends_on` to `condition: service_started`.

### Switching models
To switch between Llama 4 Scout and Qwen3-VL, edit `MODEL_NAME` in `.env` and force-recreate vLLM:

```bash
docker compose up -d vllm --force-recreate
```

### Checking logs for a specific service

```bash
docker compose logs -f vllm
docker compose logs -f open-webui
docker compose logs -f n8n
```

### Checking all service health

```bash
make health
```

Expected output (all should show `healthy` after startup completes):

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

---

## Step 8 — Verify the deployment

### 8.1 Check all services are running
```bash
make health
```
Expected: all 13 services show `healthy`.

### 8.2 Check vLLM is serving
```bash
curl http://localhost:8000/v1/models
```
Expected: JSON response listing the loaded model.

### 8.3 Check Open WebUI
```bash
curl http://localhost:8080/health
```
Expected: JSON health response.

Then open `http://<node-ip>:8080` in a browser and create the first admin account.

### 8.4 Check n8n
```bash
curl http://localhost:5678/healthz
```
Expected: `ok`.

Open `http://<node-ip>:5678` in a browser and create the admin account.

### 8.5 Check Qdrant
```bash
curl http://localhost:6333/healthz
```
Expected: `{"title":"qdrant - vector search engine","version":"...","commit":...}`

### 8.6 Check DCGM metrics
```bash
curl http://localhost:9400/metrics
```
Expected: GPU metrics output (power, temperature, memory usage, etc.).

### 8.7 Check Prometheus
```bash
curl http://localhost:9090/-/ready
```
Expected: `Prometheus is Ready`.

### 8.8 Check Loki
```bash
curl http://localhost:3100/ready
```
Expected: empty response with status 200.

### 8.9 Check Alertmanager
```bash
curl http://localhost:9093/-/healthy
```
Expected: `ok`.

### 8.10 Check Grafana
- URL: `http://<node-ip>:3000`
- Default credentials: the `GRAFANA_ADMIN_USER` and `GRAFANA_ADMIN_PASSWORD` from `.env`
- Add Prometheus as a data source (`http://prometheus:9090`)
- Add Loki as a data source (`http://loki:3100`)
- Import DCGM dashboard ID **25261**
- Optionally import a Loki log dashboard for centralized log browsing

---

## Step 9 — Adding a new model later

Once the server is air-gapped, new models must be prepared on an internet-connected workstation and transferred physically.

### 9.1 On a workstation with internet access
```bash
pip install -U huggingface_hub[cli]
huggingface-cli login

# Download the new model
huggingface-cli download <org>/<model-name> \
  --local-dir ./<model-folder> \
  --local-dir-use-symlinks False

# Create checksums
find ./<model-folder> -type f -exec sha256sum {} \; > ./<model-folder>.sha256

# Package and copy to USB drive
tar czf <model-folder>.tar.gz ./<model-folder>
```

### 9.2 Copy to the server
```bash
model_dir=/data/stack/inference-cluster-stack/data/models
sudo tar xzf /path/to/usb/<model-folder>.tar.gz -C "$model_dir"

# Verify checksums
cd "$model_dir/<model-folder>"
sha256sum -c "../<model-folder>.sha256"
```

### 9.3 Update .env and restart
```bash
cd /data/stack/inference-cluster-stack
# Edit .env — change MODEL_NAME to /models/<model-folder>
docker compose up -d vllm --force-recreate
```

### 9.4 Verify
```bash
curl http://localhost:8000/v1/models
```

---

## Step 10 — Pre-cutoff hardening

### 10.1 Disable Ubuntu automatic updates
```bash
sudo systemctl disable --now apt-daily.timer apt-daily-upgrade.timer
sudo systemctl mask apt-daily.service apt-daily-upgrade.service
sudo apt remove -y unattended-upgrades
```

### 10.2 Remove or neutralize snap if unused
If you do not need snap-based software, consider removing `snapd` before cutoff.

### 10.3 Disable vLLM usage stats
Set these environment variables in the runtime:
```text
VLLM_NO_USAGE_STATS=1
DO_NOT_TRACK=1
```

### 10.4 Snapshot package and image state
```bash
apt-mark showmanual > ~/manual-packages.txt
dpkg --get-selections > ~/dpkg-selections.txt
docker images > ~/docker-images.txt
```

### 10.5 Confirm local-only operation
- All required Docker images present locally (`docker images`)
- Models present and checksums verified
- Prometheus scrapes local targets successfully
- Grafana dashboard renders data
- vLLM answers a test request without internet
- Time sync is either local (NTP to internal source) or intentionally disabled
- Compose stack can restart fully without network access

---

## Step 11 — Air-gap enforcement (per node)

### 11.1 Physical disconnect
Disconnect the WAN uplink. There is no inter-node link to preserve — each node is fully standalone.

### 11.2 Fallback: egress-deny firewall
If a staged cutoff is required (e.g. during a burn-in period), enforce an outbound-deny firewall policy allowing only local subnet traffic.

### 11.3 Verify offline behavior
- Confirm no package or image pulls are possible (apt, pip, docker pull all fail)
- Confirm vLLM answers requests at `localhost:8000`
- Confirm Prometheus and Grafana render dashboards from local data
- Confirm model loads successfully after `docker compose restart vllm`
- Reboot the node and confirm full recovery without internet

---

## Model comparison strategy

Both models are pre-downloaded and available. The `.env` file determines which one vLLM loads at runtime.

**Recommended evaluation order:**
1. Start with **Llama 4 Scout at FP8** — more headroom (~150 GB for KV cache), easier operational margin, 10M context
2. Evaluate **Qwen3-VL 235B-A22B at FP8** — stronger reasoning, Apache 2.0 license, but tighter memory (~30-40 GB headroom)
3. If Qwen3-VL at FP8 causes OOM under your workload, switch to **Qwen3-VL at Q4** (~140 GB headroom)

Switching between them requires only changing `MODEL_NAME` in `.env` and restarting the vLLM container.

---

## Disk layout

```text
OS RAID1 (ext4) — 480 GB
  / (root)
  └── /var/lib/docker        ← Docker images, containers, volumes

Bulk RAID10 (XFS) — 30 TB → /data
  /data/
  └── stack/                  ← inference-cluster-stack/
      ├── config/
      │   ├── prometheus/
      │   │   ├── prometheus.yml
      │   │   ├── rules/
      │   │   │   └── ai-node.yml
      │   │   └── alertmanager.yml
      │   ├── promtail/
      │   │   └── promtail.yml
      │   └── postgres/
      │       └── init/
      │           └── 01-create-n8n-db.sql
      ├── data/
      │   ├── models/         ← Model weights (MODEL_DIR)
      │   │   ├── llama-4-scout/
      │   │   └── qwen3-vl-235b/
      │   ├── prometheus/     ← Prometheus TSDB
      │   ├── grafana/        ← Grafana data
      │   ├── postgres/       ← PostgreSQL database files
      │   ├── redis/          ← Redis AOF + RDB snapshots
      │   ├── qdrant/         ← Qdrant vector storage
      │   ├── n8n/            ← n8n workflow data + credentials
      │   ├── open-webui/     ← Open WebUI sessions, chats, Whisper models
      │   ├── loki/           ← Loki log index + chunks
      │   └── alertmanager/   ← Alertmanager notification state
      ├── backups/            ← Backup artifacts
      ├── docker-compose.yml
      ├── .env
      ├── .env.template
      ├── Makefile
      └── README.md
```

---

## Final checklist

### Before cutting internet (do on each node independently)
- [ ] Ubuntu installed, hostname set
- [ ] RAID10 array created, formatted XFS, mounted at `/data`, in fstab
- [ ] `/data/stack` directory created and `inference-cluster-stack/` copied in
- [ ] Data subdirectories created under `inference-cluster-stack/data/`
- [ ] Stack config directory tree verified (`config/` has prometheus, promtail, postgres)
- [ ] NVIDIA driver installed, `nvidia-smi` shows both H200s
- [ ] Docker + NVIDIA runtime verified (`docker run --gpus all nvidia/cuda:13.3.0-base-ubuntu24.04 nvidia-smi`)
- [ ] All Docker images pulled (`docker compose pull`)
- [ ] Scout and Qwen3-VL model weights downloaded and checksummed
- [ ] `.env` configured with `NODE_NAME`, `NODE_IP`, `MODEL_DIR`, `MODEL_NAME`
- [ ] Secrets generated (`make generate-secrets`) and written to `.env`
- [ ] All 13 services start cleanly (`docker compose up -d`)
- [ ] `make health` shows all services healthy
- [ ] vLLM answers a test request (`curl http://localhost:8000/v1/models`)
- [ ] Open WebUI is accessible at `http://<node-ip>:8080`
- [ ] n8n is accessible at `http://<node-ip>:5678`
- [ ] Grafana login works, DCGM dashboard renders data
- [ ] Loki + Promtail ship logs (check Grafana Explore)
- [ ] Auto-updates and telemetry disabled
- [ ] Package state snapshotted, docker images saved as `.tar`

### After cutoff (do on each node independently)
- [ ] WAN disconnected, no egress possible
- [ ] Reboot node and verify full recovery without internet
- [ ] `docker compose up -d` starts all 13 services
- [ ] `make health` shows all services healthy
- [ ] vLLM loads model and answers inference requests
- [ ] Open WebUI chat works, RAG document upload works
- [ ] Prometheus and Grafana render dashboards from local data
- [ ] Loki logs visible in Grafana Explore
- [ ] Model survives container restart (`docker compose restart vllm`)
