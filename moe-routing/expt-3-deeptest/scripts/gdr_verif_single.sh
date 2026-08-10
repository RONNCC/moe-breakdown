#!/bin/bash
#SBATCH --partition=ice-gpu
#SBATCH --qos=coc-ice
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:05:00
#SBATCH --job-name=gdr-verif-single
#SBATCH --output=/home/hice1/sghose7/scratch/moe-breakdown-runs/expt-3-deeptest/gdr-verif-single-%j.out
#SBATCH --error=/home/hice1/sghose7/scratch/moe-breakdown-runs/expt-3-deeptest/gdr-verif-single-%j.err

CURRENT_HOST=$(hostname -s)

echo "============================================"
echo "GPUDirect RDMA Diagnostics on Host: $CURRENT_HOST"
echo "Date: $(date)"
echo "============================================"

echo ""
echo "=== 1. Check IOMMU / DMAR Mode ==="
echo "--- Kernel Boot Arguments ---"
cat /proc/cmdline
echo "--- dmesg IOMMU Logs (Last 15) ---"
dmesg 2>/dev/null | grep -iE 'iommu|DMAR|AMD-Vi' | head -n 15 || echo "dmesg access denied or no entries found"

echo ""
echo "=== 2. Check ACS on PCIe Bridges ==="
found_acs=0
for bridge in $(lspci -d ::0604 | awk '{print $1}'); do
    acs=$(setpci -s "$bridge" ECAP_ACS+6.w 2>/dev/null)
    if [ -n "$acs" ] && [ "$acs" != "0000" ] && [ "$acs" != "ffff" ]; then
        echo "⚠️  ACS ACTIVE on bridge $bridge: control=$acs"
        lspci -v -s "$bridge" | head -n 10
        found_acs=1
    fi
done
if [ "$found_acs" -eq 0 ]; then
    echo "✅ No active ACS-blocking bridges found on this node."
fi

echo ""
echo "=== 3. Check GPU-to-NIC Topology ==="
nvidia-smi topo -m
echo ""
echo "--- PCIe Tree Layout ---"
lspci -t 2>/dev/null | head -n 15 || echo "lspci -t not available"

echo ""
echo "=== 4. Check nvidia_peermem Status ==="
echo "--- nvidia_peermem Refcount ---"
cat /sys/module/nvidia_peermem/refcnt 2>/dev/null || echo "nvidia_peermem refcnt not accessible"
echo "--- dmesg peermem logs ---"
dmesg 2>/dev/null | grep -iE 'peermem|nv_peer_mem|nvidia_p2p' | tail -n 15 || echo "dmesg access denied or no entries found"

echo ""
echo "============================================"
echo "End of Diagnostics on $CURRENT_HOST"
echo "============================================"
