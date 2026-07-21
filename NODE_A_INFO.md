# Node A — Inference DC A

## System Identity
- **Hostname:** inference-dc-a
- **Location:** Datacenter A
- **Vendor:** Lenovo
- **Model:** ThinkSystem SR650a V4
- **Serial:** J9033PMT

## Hardware
- **CPU:** 2× Intel Xeon 6737P, 64 cores / 128 threads
- **RAM:** 1.0 TiB
- **GPUs:**
  - GPU 0: NVIDIA H200 NVL 141GB (S/N: 1791626098591)
  - GPU 1: NVIDIA H200 NVL 141GB (S/N: 1791626098735)
- **Storage:**
  - 2× 480 GB NVMe (OS): nvme0n1, nvme2n1
  - 8× 7.6 TB SSD (model weights, monitoring data): nvme1n1, nvme3n1–nvme9n1
- **Network:** 192.168.100.119 / MAC: 36:07:0d:85:b6:98

## Software Stack
| Component | Version |
|---|---|
| OS | Ubuntu 26.04 LTS (Resolute Raccoon) |
| Kernel | 7.0.0-28-generic |
| NVIDIA Driver | 580.159.03 |
| CUDA Toolkit | 13-3 |

## Inference Stack
- **Default model:** Llama 4 Scout 109B (17B active, MoE 16 experts, FP8)
- **Secondary model:** Qwen3-VL 235B-A22B (22B active, MoE, FP8/Q4)
- **Storage path:** `/data/models`
- **Parallelism:** Pipeline Parallelism PP=2 (PCIe-connected GPUs)
