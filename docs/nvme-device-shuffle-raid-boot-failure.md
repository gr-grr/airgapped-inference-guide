# NVMe Device Shuffle: RAID Boot Failure After Reboot

## Symptom

After rebooting (e.g. after CUDA install), the server entered **dracut emergency
mode**:

```
/dev/mapper/ubuntu--vg-ubuntu--lv does not exist
```

## Root cause

**NVMe device names (`nvme0n1`, `nvme1n1`, etc.) are NOT stable across
reboots.** PCIe enumeration order can vary between boots and kernel versions.
After a fresh RAID10 creation + reboot, the kernel assigned different `nvmeXn1`
names to the same physical drives:

- The two Sandisk 480 GB OS drives got new device names
- Eight KIOXIA 7.68 TB bulk drives also got shuffled

The initramfs tried to assemble RAID arrays using stale device name references
in `/etc/mdadm/mdadm.conf`:

- Old IMSM entries from a previous RAID5 array (already torn down) confused
  `mdadm --incremental` assembly
- OS RAID1 grabbed two bulk drives instead of the OS drives
- RAID10 grabbed two OS drives instead of bulk drives
- LVM couldn't find the root PV because `md126` had wrong members

## Diagnosis

1. Booted the **previous kernel** from GRUB (Advanced Options) — older kernel
   enumerated drives differently and assembled arrays correctly.
2. Verified drive identity using serial numbers:
   ```bash
   cat /sys/block/nvmeXn1/device/serial
   cat /sys/block/nvmeXn1/device/model
   ls -la /dev/disk/by-id/ | grep -E "nvme-Sandisk|nvme-KIOXIA"
   ```
3. Confirmed arrays were **physically correct** (same physical drives, just
   different `nvmeXn1` names).

## Fix

1. Cleaned up `/etc/mdadm/mdadm.conf` — removed stale entries from old RAID5,
   kept only current arrays:
   ```bash
   sudo bash -c 'echo "# mdadm.conf - regenerated" > /etc/mdadm/mdadm.conf && mdadm --detail --scan >> /etc/mdadm/mdadm.conf'
   ```
2. Rebuilt initramfs for all kernels:
   ```bash
   sudo update-initramfs -u -k 7.0.0-28-generic
   sudo update-grub
   ```
3. Rebooted — success.

## Prevention for Node B (and future setups)

- **Always clean up mdadm.conf** after creating/changing RAID arrays — run
  `mdadm --detail --scan` to regenerate the file.
- **Rebuild initramfs** after any mdadm.conf change:
  ```bash
  sudo update-initramfs -u
  sudo update-grub
  ```
- **Use UUID-based references** (not device paths) — `mdadm --detail --scan`
  outputs UUIDs which are stable across device name changes.
- **Remove stale IMSM metadata** completely after tearing down old arrays:
  ```bash
  sudo mdadm --zero-superblock /dev/nvmeXn1
  ```
  Repeat for each drive that was in the old array.
- When in doubt, **identify drives by serial number** via
  `/dev/disk/by-id/` — not by `nvmeXn1` name.
