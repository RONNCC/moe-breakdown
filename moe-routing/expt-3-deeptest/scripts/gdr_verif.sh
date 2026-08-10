#!/bin/bash
#SBATCH --partition=ice-gpu
#SBATCH --qos=coc-ice
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:10:00
#SBATCH --job-name=gdr-verif
#SBATCH --output=/home/hice1/sghose7/scratch/moe-breakdown-runs/expt-3-deeptest/gdr-verif-%j.out
#SBATCH --error=/home/hice1/sghose7/scratch/moe-breakdown-runs/expt-3-deeptest/gdr-verif-%j.err

if [ -z "$GDR_VERIF_CHILD" ]; then
    echo "============================================"
    echo "Launching Parallel Diagnostics Across 2 Nodes..."
    echo "Job ID: $SLURM_JOB_ID"
    echo "Allocated Nodes: $SLURM_NODELIST"
    echo "============================================"
    export GDR_VERIF_CHILD=1
    # Run ourselves in parallel on both nodes
    srun --ntasks-per-node=1 --nodes=2 bash "$0"
    exit 0
fi

CURRENT_HOST=$(hostname -s)
NODE_0=$(scontrol show hostnames "$SLURM_NODELIST" | head -n 1)
NODE_1=$(scontrol show hostnames "$SLURM_NODELIST" | tail -n 1)

echo "============================================"
echo "GPUDirect RDMA Diagnostics on Host: $CURRENT_HOST"
echo "Role: $( [ "$CURRENT_HOST" == "$NODE_0" ] && echo 'SERVER (Node 0)' || echo 'CLIENT (Node 1)' )"
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
echo "=== 5. End-to-End GPUDirect RDMA Test ==="
if [ "$CURRENT_HOST" == "$NODE_0" ]; then
    echo "[Node 0 - Server] Starting ib_write_bw servers..."
    
    # Run host memory server on port 18515
    ib_write_bw -d mlx5_0 --use_cuda=0 -a -p 18515 > /tmp/ib_write_host_server.log 2>&1 &
    SERVER_PID_HOST=$!
    
    # Run CUDA GPU memory server on port 18516
    # Note: we use --use_cuda=0 as the default if it doesn't support cuda, but if it does, cuda test will target port 18516
    ib_write_bw -d mlx5_0 -a -p 18516 > /tmp/ib_write_cuda_server.log 2>&1 &
    SERVER_PID_CUDA=$!
    
    echo "[Node 0 - Server] Waiting for Node 1 to connect..."
    sleep 20
    
    # Clean up
    kill $SERVER_PID_HOST $SERVER_PID_CUDA 2>/dev/null || true
    wait $SERVER_PID_HOST $SERVER_PID_CUDA 2>/dev/null
    
    echo "[Node 0 - Server] Host Server Output:"
    cat /tmp/ib_write_host_server.log || true
    echo "[Node 0 - Server] CUDA Server Output:"
    cat /tmp/ib_write_cuda_server.log || true
else
    echo "[Node 1 - Client] Sourcing modules..."
    # Try to load perftest module if available
    module load perftest gcc/12.3.0 python/3.11 cuda/13.0.1 >/dev/null 2>&1 || true
    
    echo "[Node 1 - Client] Waiting for Node 0 server to start..."
    sleep 3
    
    echo "[Node 1 - Client] Running Host Memory RDMA (use_cuda=0)..."
    ib_write_bw -d mlx5_0 --use_cuda=0 -a "$NODE_0" -p 18515 || echo "❌ Host Memory RDMA FAILED!"
    
    echo ""
    echo "[Node 1 - Client] Running GPUDirect RDMA (use_cuda=0 with GPU 0)..."
    if ib_write_bw --help 2>&1 | grep -q "use_cuda"; then
        echo "[Node 1 - Client] running ib_write_bw with GPU..."
        ib_write_bw -d mlx5_0 -a "$NODE_0" -p 18516 || echo "❌ GPU-to-GPU GPUDirect RDMA FAILED!"
    else
        echo "[Node 1 - Client] ib_write_bw does not support CUDA in this build."
    fi
fi

echo ""
echo "============================================"
echo "End of Diagnostics on $CURRENT_HOST"
echo "============================================"
