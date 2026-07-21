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
- 8× 7.6TB SSDs for bulk storage (model weights, monitoring data)
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
sudo apt install -y cuda-toolkit-13-3
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
docker run --rm --gpus all nvidia/cuda:13.3.0-base-ubuntu24.04 nvidia-smi
```

---

## Step 4 — Model selection and download

### 4.1 Serving engine
Use **vLLM** with **Pipeline Parallelism (PP=2)** to split the model across the two PCIe-connected H200s.

### 4.2 Model candidates
We deploy **both** models for comparison:

| Model | Total params | Active params | Architecture | Context | Multimodal | Tool use |
|---|---|---|---|---|---|---|
| **Llama 4 Scout** | 109B | 17B | MoE (16 experts) | 10M tokens | Text + Image | Yes |
| **Qwen3-VL 235B-A22B** | 235B | 22B | MoE | 256K native, 1M extended | Text + Image + Video | Yes |

### 4.3 Memory budget per model

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

### 4.4 Model storage layout
- Keep model weights on bulk SSD storage, for example `/data/models`
- Keep OS and hot container/runtime data on NVMe

```bash
sudo mkdir -p /data/models
```

### 4.5 Download model weights

Download on a machine with internet access, then transfer to the server via USB:

```bash
# On the workstation:
pip install -U huggingface_hub[cli]
huggingface-cli login

# Llama 4 Scout
huggingface-cli download meta-llama/Llama-4-Scout-17B-16E-Instruct \
  --local-dir ./llama-4-scout \
  --local-dir-use-symlinks False

# Qwen3-VL
huggingface-cli download Qwen/Qwen3-VL-235B-A22B-Instruct \
  --local-dir ./qwen3-vl-235b \
  --local-dir-use-symlinks False

# Package for transfer
tar czf llama-4-scout.tar.gz ./llama-4-scout
tar czf qwen3-vl-235b.tar.gz ./qwen3-vl-235b
# Copy tar files to USB drive
```

If the server has temporary internet access during setup, you can download directly:

```bash
huggingface-cli download meta-llama/Llama-4-Scout-17B-16E-Instruct \
  --local-dir /data/models/llama-4-scout \
  --local-dir-use-symlinks False
huggingface-cli download Qwen/Qwen3-VL-235B-A22B-Instruct \
  --local-dir /data/models/qwen3-vl-235b \
  --local-dir-use-symlinks False
```

### 4.6 Transfer to the server

```bash
sudo tar xzf /path/to/usb/llama-4-scout.tar.gz -C /data/models/
sudo tar xzf /path/to/usb/qwen3-vl-235b.tar.gz -C /data/models/
```

### 4.7 Checksum the model files

```bash
find /data/models/llama-4-scout -type f -exec sha256sum {} \; > /data/models/llama-4-scout.sha256
find /data/models/qwen3-vl-235b -type f -exec sha256sum {} \; > /data/models/qwen3-vl-235b.sha256
```

---

## Step 5 — Stack setup and per-node configuration

### 5.1 Copy the stack directory
The `inference-cluster-stack/` directory contains the full Docker Compose stack. Copy it to the server (via USB or however the server was provisioned):

```bash
cp -r /path/to/usb/inference-cluster-stack ~/
```

### 5.2 Pull container images
Pull all container images now while internet is available:

```bash
cd ~/inference-cluster-stack
docker compose pull
```

This pulls vLLM, DCGM exporter, Prometheus, and Grafana images.

### 5.3 Configure environment

```bash
cp .env.template .env
```

Edit `.env` and set:
- `NODE_NAME` — e.g. `node1` (must match the hostname from Step 1.1)
- `MODEL_DIR` — path to model weights on bulk SSD (e.g. `/data/models`)
- `MODEL_NAME` — e.g. `/models/llama-4-scout`

### 5.4 Both nodes are identical
The two nodes run the **same stack** with the same config files and the same procedure. Each node gets its own copy of `inference-cluster-stack/` and is configured independently. The only per-node differences are:
- **Hostname** (set in Step 1.1)
- **`.env` values** (node name, model path)

From this point forward, every step is executed **identically on both nodes**.

---

## Step 6 — Deploy the stack

All services are managed through Docker Compose. Start everything with a single command:

```bash
cd ~/inference-cluster-stack
docker compose up -d
```

### Services started

| Service | Container name | Purpose |
|---|---|---|
| vLLM | `vllm-server` | LLM inference engine (PP=2 across both GPUs) |
| DCGM exporter | `dcgm-exporter` | GPU metrics (power, temperature, memory) |
| Prometheus | `prometheus` | Metrics collection (scrapes localhost targets only) |
| Grafana | `grafana` | GPU monitoring dashboards |

### Switching models
To switch between Llama 4 Scout and Qwen3-VL, edit `MODEL_NAME` in `.env` and force-recreate vLLM:

```bash
docker compose up -d vllm --force-recreate
```

### Checking logs

```bash
docker compose logs -f vllm
```

---

## Step 7 — Verify the deployment

### 7.1 Check vLLM is serving
```bash
curl http://localhost:8000/v1/models
```
Expected: JSON response listing the loaded model.

### 7.2 Check DCGM metrics
```bash
curl http://localhost:9400/metrics
```
Expected: GPU metrics output (power, temperature, memory usage, etc.).

### 7.3 Check Prometheus
```bash
curl http://localhost:9090/-/ready
```
Expected: `Prometheus is Ready`.

### 7.4 Check Grafana
- URL: `http://localhost:3000`
- Default credentials: `admin/admin`
- Change password immediately
- Add Prometheus as a data source (`http://localhost:9090`)
- Import DCGM dashboard ID **25261**

---

## Step 8 — Adding a new model later

Once the server is air-gapped, new models must be prepared on an internet-connected workstation and transferred physically.

### 8.1 On a workstation with internet access
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

### 8.2 Copy to the server
```bash
sudo tar xzf /path/to/usb/<model-folder>.tar.gz -C /data/models/

# Verify checksums
cd /data/models/<model-folder>
sha256sum -c ../<model-folder>.sha256
```

### 8.3 Update .env and restart
```bash
cd ~/inference-cluster-stack
# Edit .env — change MODEL_NAME to /models/<model-folder>
docker compose up -d vllm --force-recreate
```

### 8.4 Verify
```bash
curl http://localhost:8000/v1/models
```

---

## Step 9 — Pre-cutoff hardening

### 9.1 Disable Ubuntu automatic updates
```bash
sudo systemctl disable --now apt-daily.timer apt-daily-upgrade.timer
sudo systemctl mask apt-daily.service apt-daily-upgrade.service
sudo apt remove -y unattended-upgrades
```

### 9.2 Remove or neutralize snap if unused
If you do not need snap-based software, consider removing `snapd` before cutoff.

### 9.3 Disable vLLM usage stats
Set these environment variables in the runtime:
```text
VLLM_NO_USAGE_STATS=1
DO_NOT_TRACK=1
```

### 9.4 Snapshot package and image state
```bash
apt-mark showmanual > ~/manual-packages.txt
dpkg --get-selections > ~/dpkg-selections.txt
docker images > ~/docker-images.txt
```

### 9.5 Save container images as tar archives
```bash
docker save -o ~/vllm-openai.tar vllm/vllm-openai:latest
docker save -o ~/dcgm-exporter.tar nvidia/dcgm-exporter:latest
docker save -o ~/prometheus.tar prom/prometheus:latest
docker save -o ~/grafana.tar grafana/grafana:latest
```

### 9.6 Confirm local-only operation
- All required Docker images present locally (`docker images`)
- Models present and checksums verified
- Prometheus scrapes local targets successfully
- Grafana dashboard renders data
- vLLM answers a test request without internet
- Time sync is either local (NTP to internal source) or intentionally disabled
- Compose stack can restart fully without network access

---

## Step 10 — Air-gap enforcement (per node)

### 10.1 Physical disconnect
Disconnect the WAN uplink. There is no inter-node link to preserve — each node is fully standalone.

### 10.2 Fallback: egress-deny firewall
If a staged cutoff is required (e.g. during a burn-in period), enforce an outbound-deny firewall policy allowing only local subnet traffic.

### 10.3 Verify offline behavior
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
├── .env.template
├── README.md
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
- [ ] Docker + NVIDIA runtime verified (`docker run --gpus all nvidia/cuda:13.3.0-base-ubuntu24.04 nvidia-smi`)
- [ ] All Docker images pulled (`docker compose pull`)
- [ ] Scout and Qwen3-VL model weights downloaded and checksummed
- [ ] Prometheus scrapes, Grafana login works
- [ ] Compose stack starts cleanly (`docker compose up -d`)
- [ ] vLLM answers a test request (`curl http://localhost:8000/v1/models`)
- [ ] Auto-updates and telemetry disabled
- [ ] Package state snapshotted, docker images saved as `.tar`

### After cutoff (do on each node independently)
- [ ] WAN disconnected, no egress possible
- [ ] Reboot node and verify full recovery without internet
- [ ] `docker compose up -d` starts all services
- [ ] vLLM loads model and answers inference requests
- [ ] Prometheus and Grafana render dashboards from local data
- [ ] Model survives container restart (`docker compose restart vllm`)
