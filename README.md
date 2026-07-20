# Air-Gapped AI Inference Blueprint — Identical Nodes, Separate Datacenters

## Goal and scope

This guide packages our deployment plan for **two identical standalone inference nodes**, each with 2× NVIDIA H200 141GB GPUs connected by **PCIe** (no NVLink), running in **separate datacenters** with no network link between them.

The architecture changed from the original plan:
- **Intra-node**: 2× H200 on PCIe → use **Pipeline Parallelism (PP=2)** instead of Tensor Parallelism (TP), because PP has lower inter-GPU communication overhead on PCIe
- **Inter-node**: No 10GbE link between datacenters. Each node is fully independent.
- **Goal**: Both nodes run the **exact same stack** — same configs, same images, same model weights — delivered via a reproducible blueprint.

Use this as the operational runbook from fresh Ubuntu install → internet-connected setup → build the deployment bundle → identical deployment at each DC → final hardening and air-gap.

## Infrastructure summary

### Per server (both nodes identical)
- 2× NVIDIA H200, 141GB HBM3e each
- PCIe interconnect (no NVLink between the two GPUs)
- Intel Xeon 6700P-series, 32 cores
- Ubuntu Server 26.04 LTS
- 8× 7.6TB SSDs for bulk storage (model weights, registry, monitoring data)
- 2× NVMe drives for OS and hot data

### Deployment model
- 2 physical servers, one per datacenter
- No network link between them
- Each node is air-gapped independently
- Identical open-source model serving stack on both

## Architecture decision

### Intra-node: PCIe → Pipeline Parallelism (PP)
The two H200s inside each node connect via **PCIe** (~64 GB/s on Gen4 ×16 or ~128 GB/s on Gen5 ×16), not NVLink (900 GB/s). Tensor Parallelism (TP) requires an all-reduce after every transformer layer, which becomes a bottleneck on PCIe. Scaling efficiency for TP on PCIe is roughly 70-78% per additional GPU.

**Pipeline Parallelism (PP=2)** is the recommended strategy for PCIe-connected GPUs. It splits the model's layers sequentially across the two GPUs:
- GPU 0 processes layers 0-N, GPU 1 processes layers N+1 to end
- Inter-GPU communication is limited to activation tensors between stages (~67 MB per pass vs. 160+ all-reduce ops per token for TP)
- For **multi-user API workloads** (high concurrency), the idle pipeline bubbles get filled with queued requests, making PP throughput-efficient
- vLLM's own documentation recommends PP over TP for non-NVLink multi-GPU setups

### Per-node stack (both nodes identical)
Both nodes run the **full stack** — there is no control-plane/worker split:
- vLLM (PP=2 across both GPUs)
- DCGM exporter
- zot OCI registry (local image cache)
- HAProxy (optional, for local TLS termination / rate limiting)
- Prometheus + Grafana (local monitoring only)

### Why no cross-node networking
The two nodes are in separate datacenters with no direct link:
- No HAProxy load balancing across nodes
- No cross-node Prometheus scraping
- No zot registry syncing
- No rsync of model weights
- Each node is independently provisioned and air-gapped

---

## Step 1 — OS baseline and system prep

### 1.1 Set hostname
On the node at DC A:
```bash
sudo hostnamectl set-hostname inference-dc-a
```

On the node at DC B:
```bash
sudo hostnamectl set-hostname inference-dc-b
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

## Step 2 — NVIDIA driver and CUDA

### 2.1 Add NVIDIA CUDA repository
```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2604/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
```

### 2.2 Install the datacenter driver
```bash
sudo apt search nvidia-driver
sudo apt install -y nvidia-driver-580-server
```

### 2.3 Install CUDA toolkit
```bash
sudo apt install -y cuda-toolkit-12-8
```

### 2.4 Reboot and verify
```bash
sudo reboot
nvidia-smi
```
Expected result: both H200 GPUs visible.

### 2.5 Verify local GPU topology
```bash
nvidia-smi topo -m
```
Expected: the two H200s show a PCIe connection, not NVLink. The `nvlink` flag will confirm no NVLink links exist — this is expected and confirms our PP strategy is correct.

---

## Step 3 — Docker and NVIDIA Container Toolkit

### 3.1 Remove conflicting packages
```bash
for pkg in docker.io docker-doc docker-compose docker-compose-v2 podman-docker containerd runc; do sudo apt-get remove -y $pkg; done
```

### 3.2 Add Docker repository
```bash
sudo apt-get update
sudo apt-get install -y ca-certificates curl
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc
echo   "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu   $(. /etc/os-release && echo "$VERSION_CODENAME") stable" |   sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update
```

### 3.3 Install Docker
```bash
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
```

### 3.4 Add your user to docker group
```bash
sudo groupadd docker 2>/dev/null
sudo usermod -aG docker $USER
newgrp docker
```

### 3.5 Install NVIDIA Container Toolkit
```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list |   sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' |   sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

### 3.6 Verify GPU access in containers
```bash
docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu24.04 nvidia-smi
```

---

## Step 4 — Model serving stack and model selection

### 4.1 Serving engine
Use **vLLM** as the primary serving layer, with **Pipeline Parallelism (PP=2)** to split the model across the two PCIe-connected H200s.

Pull the image now while internet is available:
```bash
docker pull vllm/vllm-openai:latest
```

### 4.2 Parallelism configuration
Override the old `--tensor-parallel-size` approach. Run vLLM with:

```
--pipeline-parallel-size 2
--tensor-parallel-size 1
```

PP=2 splits the model by layers. GPU 0 handles the first half of layers, GPU 1 handles the second half. For a multi-user API with high concurrency, PP fills idle pipeline bubbles with queued requests, making it efficient on PCIe.

### 4.3 Model selection — two candidates
We deploy **both** models for comparison:

| Model | Total params | Active params | Architecture | Context | Multimodal | Tool use |
|---|---|---|---|---|---|---|
| **Llama 4 Scout** | 109B | 17B | MoE (16 experts) | 10M tokens | Text + Image | Yes |
| **Qwen3-VL 235B-A22B** | 235B | 22B | MoE | 256K native, 1M extended | Text + Image + Video | Yes |

### 4.4 Memory budget per model

**Llama 4 Scout at FP8** (recommended starting point):
```
Total GPU memory:     282 GB (2 × 141 GB)
Model weights (FP8):  ~109 GB
KV cache + overhead:  ~150 GB headroom ← excellent for multi-user + long context
```

**Qwen3-VL 235B-A22B at FP8** (tighter, still workable):
```
Total GPU memory:     282 GB
Model weights (FP8):  ~235 GB
KV cache + overhead:  ~30-40 GB headroom ← tight, fewer concurrent users
```

**Qwen3-VL 235B-A22B at Q4** (more headroom):
```
Total GPU memory:     282 GB
Model weights (Q4):   ~118 GB
KV cache + overhead:  ~140 GB headroom ← comfortable
```

**FP8** is the default precision on H200 (native Hopper FP8 tensor cores, near-lossless quality). **Q4** (AWQ/GPTQ) compresses weights further at a small quality cost but frees more memory for KV cache and concurrency.

### 4.5 Model storage layout
- Keep model weights on bulk SSD storage, for example `/data/models`
- Keep OS and hot container/runtime data on NVMe

Example:
```bash
sudo mkdir -p /data/models
```

### 4.6 Hugging Face CLI
```bash
pip install -U huggingface_hub[cli]
huggingface-cli login
```

### 4.7 Download model weights
Download **both** model checkpoints while internet is available:

```bash
# Llama 4 Scout
huggingface-cli download meta-llama/Llama-4-Scout-17B-16E-Instruct \
  --local-dir /data/models/llama-4-scout \
  --local-dir-use-symlinks False

# Qwen3-VL
huggingface-cli download Qwen/Qwen3-VL-235B-A22B-Instruct \
  --local-dir /data/models/qwen3-vl-235b \
  --local-dir-use-symlinks False
```

### 4.8 Checksum the model files
```bash
find /data/models/llama-4-scout -type f -exec sha256sum {} \; > /data/models/llama-4-scout.sha256
find /data/models/qwen3-vl-235b -type f -exec sha256sum {} \; > /data/models/qwen3-vl-235b.sha256
```

---

## Step 5 — Deployment bundle and identical provisioning

Since the two nodes are in separate datacenters with no network link, we use a **deployment bundle** approach: build the complete stack on one reference node, package it into a transportable bundle, then deploy identically at each DC.

### 5.1 The blueprint concept
Both DC operators follow the **same step-by-step guide**. The "sameness" comes from using the same config files, same image versions, same model checkpoints, and the same procedure. There is no cross-node syncing.

### 5.2 Reference deployment directory
The entire stack lives under `inference-cluster-stack/`. Copy this directory to a USB drive to carry between sites:
```bash
cp -r inference-cluster-stack /path/to/usb/
```

### 5.3 Per-node customization
Each node only differs in:
- **Hostname** (set in Step 1.1)
- **`.env` values** — hostname, node name (all other config is identical)

Everything else — docker-compose.yml, config files, model paths, port numbers — is the same on both nodes.

### 5.4 Set hostname on each node
On the node at DC A:
```bash
sudo hostnamectl set-hostname inference-dc-a
```

On the node at DC B:
```bash
sudo hostnamectl set-hostname inference-dc-b
```

### 5.5 Everything else is identical
From this point forward, every step in this guide is executed **identically on both nodes** unless explicitly noted.

---

## Step 6 — Internal registry with zot (per node)

Each node runs its **own local** zot registry. The registry serves as a local image cache for air-gapped operation — no images are pulled from another node.

### 6.1 Create registry storage
```bash
sudo mkdir -p /data/registry
sudo mkdir -p /etc/zot
```

### 6.2 Pull zot
```bash
docker pull ghcr.io/project-zot/zot:latest
```

### 6.3 Create zot config
Create `/etc/zot/config.json`:
```json
{
  "storage": {
    "rootDirectory": "/var/lib/registry"
  },
  "http": {
    "address": "0.0.0.0",
    "port": "5000"
  },
  "log": {
    "level": "info"
  }
}
```

### 6.4 Run zot
```bash
docker run -d   --name zot-registry   --restart unless-stopped   -p 5000:5000   -v /data/registry:/var/lib/registry   -v /etc/zot/config.json:/etc/zot/config.json   ghcr.io/project-zot/zot:latest   serve /etc/zot/config.json
```

### 6.5 Verify
```bash
curl http://localhost:5000/v2/_catalog
```

### 6.6 Configure Docker insecure registry (local only)
Edit `/etc/docker/daemon.json` to use the local registry:
```json
{
  "insecure-registries": ["localhost:5000"]
}
```
Restart Docker:
```bash
sudo systemctl restart docker
```

### 6.7 Mirror images into local zot
```bash
docker tag vllm/vllm-openai:latest localhost:5000/vllm-openai:latest
docker push localhost:5000/vllm-openai:latest

docker tag nvidia/dcgm-exporter:latest localhost:5000/dcgm-exporter:latest
docker push localhost:5000/dcgm-exporter:latest

docker tag prom/prometheus:latest localhost:5000/prometheus:latest
docker push localhost:5000/prometheus:latest

docker tag grafana/grafana:latest localhost:5000/grafana:latest
docker push localhost:5000/grafana:latest

docker tag haproxy:latest localhost:5000/haproxy:latest
docker push localhost:5000/haproxy:latest
```

---

## Step 7 — HAProxy (optional, per node)

With a single vLLM instance per node (PP=2 across both GPUs), HAProxy is **optional**. You can skip it entirely and point clients directly at `localhost:8000`.

Use HAProxy if you need:
- TLS termination in front of vLLM
- Rate limiting per client
- Request logging / access control
- A health-check endpoint decoupled from vLLM

### 7.1 Install HAProxy
```bash
sudo apt update
sudo apt install -y haproxy
```

### 7.2 Configure HAProxy for local-only balancing
Edit `/etc/haproxy/haproxy.cfg`:
```cfg
global
    log stdout format raw local0
    maxconn 4096

defaults
    mode http
    log global
    option httplog
    timeout connect 10s
    timeout client 300s
    timeout server 300s
    retries 2

frontend llm_front
    bind *:8080
    default_backend llm_backends

backend llm_backends
    balance roundrobin
    option httpchk GET /v1/models
    http-check expect status 200
    server local-vllm localhost:8000 check inter 10s

listen stats
    bind *:8404
    stats enable
    stats uri /stats
    stats refresh 10s
```

### 7.3 Restart and enable
```bash
sudo systemctl restart haproxy
sudo systemctl enable haproxy
```

### 7.4 Verify
```bash
curl http://localhost:8080/v1/models
```

---

## Step 8 — Monitoring with DCGM, Prometheus, and Grafana (per node)

Each node runs its **own** monitoring stack. Prometheus only scrapes local targets. There is no cross-node observability.

### 8.1 Pull images
```bash
docker pull nvidia/dcgm-exporter:latest
docker pull prom/prometheus:latest
docker pull grafana/grafana:latest
```

### 8.2 Run DCGM exporter
```bash
docker run -d   --name dcgm-exporter   --restart unless-stopped   --gpus all   --cap-add SYS_ADMIN   -p 9400:9400   nvidia/dcgm-exporter:latest
```

### 8.3 Prometheus config (local-only scraping)
Create `/data/prometheus/prometheus.yml`:
```yaml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'dcgm'
    static_configs:
      - targets: ['localhost:9400']
  - job_name: 'vllm'
    static_configs:
      - targets: ['localhost:8000']
```

### 8.4 Run Prometheus
```bash
docker run -d   --name prometheus   --restart unless-stopped   -p 9090:9090   -v /data/prometheus/prometheus.yml:/etc/prometheus/prometheus.yml   prom/prometheus:latest
```

### 8.5 Run Grafana
```bash
docker run -d   --name grafana   --restart unless-stopped   -p 3000:3000   -v /data/grafana:/var/lib/grafana   grafana/grafana:latest
```

### 8.6 Grafana first login
- URL: `http://localhost:3000`
- Default credentials: `admin/admin`
- Change password immediately
- Add Prometheus as a data source (http://localhost:9090)
- Import DCGM dashboard ID 25261

---

## Step 9 — Compose-based stack management (identical on both nodes)

Both nodes run the **full Compose stack** — there are no profiles or role splits.

### Included services
- vLLM (PP=2 across both GPUs)
- DCGM exporter
- zot registry (local image cache)
- HAProxy (optional, enable if you need TLS/rate-limiting)
- Prometheus (local-only scraping)
- Grafana

### Typical commands (identical on both nodes)
```bash
cd inference-cluster-stack
docker compose up -d
```

### Running a different model
To switch between Scout and Qwen3-VL, edit `.env` and restart:
```bash
# Set MODEL_NAME to the active model, then:
docker compose up -d vllm --force-recreate
```

### Checking logs
```bash
docker compose logs -f vllm
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

### 10.5 Save critical images as tar archives
```bash
docker save -o ~/vllm-openai.tar vllm/vllm-openai:latest
docker save -o ~/dcgm-exporter.tar nvidia/dcgm-exporter:latest
docker save -o ~/prometheus.tar prom/prometheus:latest
docker save -o ~/grafana.tar grafana/grafana:latest
docker save -o ~/zot.tar ghcr.io/project-zot/zot:latest
```

### 10.6 Confirm local-only operation
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

## Folder structure for the stack

```text
inference-cluster-stack/
├── docker-compose.yml
├── .env
├── haproxy/
│   └── haproxy.cfg
├── registry/
│   ├── config.json
│   └── data/
├── prometheus/
│   └── prometheus.yml
├── grafana/
│   └── data/
└── models/
```

---

## Final checklist

### Before cutting internet (do on each node independently)
- [ ] Ubuntu installed, hostname set
- [ ] NVIDIA driver installed, `nvidia-smi` shows both H200s
- [ ] Docker + NVIDIA runtime verified (`docker run --gpus all nvidia/cuda:12.8.0-base-ubuntu24.04 nvidia-smi`)
- [ ] All Docker images pulled and mirrored to local zot
- [ ] Scout and Qwen3-VL model weights downloaded and checksummed
- [ ] zot registry verified (`curl http://localhost:5000/v2/_catalog`)
- [ ] Monitoring stack verified (Prometheus scrapes, Grafana login)
- [ ] Compose stack starts cleanly (`docker compose up -d`)
- [ ] vLLM answers a test request (`curl http://localhost:8000/v1/models`)
- [ ] Auto-updates and telemetry disabled
- [ ] Package state snapshotted, docker images saved as `.tar`
- [ ] `inference-cluster-stack/` directory copied to USB drive

### After cutoff (do on each node independently)
- [ ] WAN disconnected, no egress possible
- [ ] Reboot node and verify full recovery without internet
- [ ] `docker compose up -d` starts all services
- [ ] vLLM loads model and answers inference requests
- [ ] Prometheus and Grafana render dashboards from local data
- [ ] Model survives container restart (`docker compose restart vllm`)

