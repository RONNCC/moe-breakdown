#!/bin/bash
#SBATCH --partition=ice-gpu
#SBATCH --qos=coc-ice
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:05:00
#SBATCH --job-name=gdr-active-test
#SBATCH --output=/home/hice1/sghose7/scratch/moe-breakdown-runs/expt-3-deeptest/gdr-active-%j.out
#SBATCH --error=/home/hice1/sghose7/scratch/moe-breakdown-runs/expt-3-deeptest/gdr-active-%j.err

echo "============================================"
echo "Active GPUDirect RDMA Loopback Diagnostics"
echo "Host: $(hostname -s)"
echo "Date: $(date)"
echo "============================================"

# Load modules
echo "[1/5] Loading Environment Modules..."
module purge
module load gcc/12.3.0 python/3.11 cuda/13.0.1 >/dev/null 2>&1 || true
# Load perftest if available
module load perftest >/dev/null 2>&1 || true

echo ""
echo "=== 2. Check Active IOMMU Runtime State ==="
echo "--- /sys/class/iommu Group Types ---"
# Check if IOMMU is translating or bypass/identity
if ls /sys/class/iommu/*/iommu_group/type >/dev/null 2>&1; then
    for type_file in /sys/class/iommu/*/iommu_group/type; do
        echo "Group $(basename $(dirname "$type_file")): $(cat "$type_file")"
    done | sort | uniq -c
else
    echo "No active IOMMU groups found in sysfs (IOMMU likely disabled)."
fi

echo "--- dmesg IOMMU Logs ---"
dmesg 2>/dev/null | grep -iE 'DMAR|iommu|AMD-Vi' | head -n 15 || echo "dmesg access denied or no logs"

echo ""
echo "=== 3. Check Kernel Symbol Resolution for nvidia_peermem ==="
# Check if nvidia core symbols are present in kallsyms
echo "Searching kallsyms for nvidia_p2p symbols:"
cat /proc/kallsyms | grep -E 'nvidia_p2p_dma_map_pages|nvidia_p2p_dma_unmap_pages' || echo "⚠️ Symbols NOT found in kallsyms (or access restricted)"

echo ""
echo "=== 4. Active GPUDirect RDMA Loopback Test (ib_write_bw) ==="
# Check if ib_write_bw is available in path
if which ib_write_bw >/dev/null 2>&1; then
    echo "ib_write_bw found!"
    if ib_write_bw --help 2>&1 | grep -q "use_cuda"; then
        echo "✅ ib_write_bw supports CUDA memory mapping! Starting local loopback test..."
        
        # Start server in background on port 18530
        ib_write_bw -d mlx5_0 --use_cuda=0 -a -p 18530 > /tmp/gdr_loopback_server.log 2>&1 &
        SERVER_PID=$!
        sleep 2
        
        # Run client connecting to localhost
        echo "Running client..."
        ib_write_bw -d mlx5_0 --use_cuda=0 -a localhost -p 18530 || echo "❌ GPUDirect RDMA loopback test failed!"
        
        # Clean up
        kill $SERVER_PID 2>/dev/null || true
        wait $SERVER_PID 2>/dev/null
        
        echo "--- Server Output Log ---"
        cat /tmp/gdr_loopback_server.log
    else
        echo "⚠️ ib_write_bw is installed but does NOT support CUDA (--use_cuda option missing)."
    fi
else
    echo "⚠️ ib_write_bw not found in PATH."
fi

echo ""
echo "=== 5. Active NCCL GPUDirect RDMA Diagnostics ==="
# Sourcing our virtual environment to run python torch tests
VENV_DIR="/storage/ice1/0/2/sghose7/moe-breakdown-deepep-venv"
if [ -d "$VENV_DIR" ]; then
    echo "Sourcing Python Virtual Environment at $VENV_DIR"
    source "$VENV_DIR/bin/activate"
    
    export NCCL_DEBUG=INFO
    export NCCL_DEBUG_SUBSYS=NET,INIT
    export NCCL_NET_GDR_LEVEL=5
    export MASTER_ADDR=localhost
    export MASTER_PORT=29505
    
    echo "Executing Python NCCL initialization test..."
    python3 -c "
import os
import torch
import torch.distributed as dist

print('CUDA device count:', torch.cuda.device_count())
print('Active Device:', torch.cuda.get_device_name(0))

try:
    dist.init_process_group('nccl', rank=0, world_size=1)
    t = torch.randn(1024, 1024, device='cuda')
    dist.all_reduce(t)
    print('✅ NCCL local collective test passed!')
except Exception as e:
    print('❌ NCCL test crashed:', e)
"
else
    echo "⚠️ Python virtual environment not found. Skipping NCCL diagnostics."
fi

echo ""
echo "============================================"
echo "End of Active Diagnostics"
echo "============================================"
