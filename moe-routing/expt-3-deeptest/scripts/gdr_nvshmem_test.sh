#!/bin/bash
#SBATCH --partition=ice-gpu
#SBATCH --qos=coc-ice
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:05:00
#SBATCH --job-name=gdr-nvshmem-test
#SBATCH --output=/home/hice1/sghose7/scratch/moe-breakdown-runs/expt-3-deeptest/gdr-nvshmem-%j.out
#SBATCH --error=/home/hice1/sghose7/scratch/moe-breakdown-runs/expt-3-deeptest/gdr-nvshmem-%j.err

echo "============================================"
echo "NVSHMEM Native Active Initialization Test"
echo "Host: $(hostname -s)"
echo "Date: $(date)"
echo "============================================"

# Load modules
echo "[1/4] Sourcing Environment Modules..."
module purge
module load gcc/12.3.0 python/3.11 cuda/13.0.1 >/dev/null 2>&1 || true

# Source virtual environment
VENV_DIR="/storage/ice1/0/2/sghose7/moe-breakdown-deepep-venv"
if [ -d "$VENV_DIR" ]; then
    echo "Sourcing Virtual Environment at $VENV_DIR"
    source "$VENV_DIR/bin/activate"
else
    echo "❌ Virtual environment not found at $VENV_DIR"
    exit 1
fi

# Set NVSHMEM TRACE environment variables
export NVSHMEM_DEBUG=TRACE
export NVSHMEM_DEBUG_SUBSYS=ALL
export NVSHMEM_DEBUG_FILE="/tmp/nvshmem_debug_active.log"
rm -f "$NVSHMEM_DEBUG_FILE"

# Make sure we are NOT using the stub intercept!
# We want to test the original reinstated ibrc and ibdevx libraries.
export LD_LIBRARY_PATH="/storage/ice1/0/2/sghose7/moe-breakdown-deepep-venv/lib/python3.11/site-packages/nvidia/nccl/lib:$LD_LIBRARY_PATH"

export MASTER_ADDR=localhost
export MASTER_PORT=29510

echo ""
echo "=== 2. Running Active NVSHMEM/DeepEP Initialization in Python ==="
python3 -c "
import sys
import torch
import torch.distributed as dist

print('Initializing PyTorch distributed NCCL...')
dist.init_process_group('nccl', rank=0, world_size=1)

print('Importing nvidia.nvshmem...')
import nvidia.nvshmem as nvshmem
print('Package attributes:', dir(nvshmem))

try:
    if hasattr(nvshmem, 'init'):
        print('Triggering nvshmem.init()...')
        nvshmem.init()
        print('✅ nvshmem.init() completed successfully!')
    elif hasattr(nvshmem, 'init_process_group'):
        print('Triggering nvshmem.init_process_group()...')
        nvshmem.init_process_group()
        print('✅ nvshmem.init_process_group() completed successfully!')
    else:
        print('nvidia.nvshmem lacks a default python init method. Attempting to trigger NVSHMEM init via DeepEP...')
        import deep_ep
        print('Imported deep_ep successfully! Creating Buffer with correct kwargs (HT mode)...')
        buf = deep_ep.Buffer(
            group=dist.group.WORLD,
            num_nvl_bytes=10 * 1024 * 1024,
            num_rdma_bytes=10 * 1024 * 1024,
            low_latency_mode=True,
            num_qps_per_rank=1,
            explicitly_destroy=True
        )
        print('✅ DeepEP Buffer initialized successfully!')
        buf.destroy()
except Exception as e:
    print('❌ NVSHMEM/DeepEP initialization failed with exception:', e)
"

echo ""
echo "=== 3. Inspecting Verbose NVSHMEM Trace Log ==="
if [ -f "$NVSHMEM_DEBUG_FILE" ]; then
    echo "Trace file found! Printing last 100 lines:"
    echo "----------------------------------------"
    cat "$NVSHMEM_DEBUG_FILE" | tail -n 100
    echo "----------------------------------------"
else
    echo "⚠️  NVSHMEM trace file was NOT created! This means the C++ initialization was not triggered, or the process crashed before writing."
fi

echo ""
echo "=== 4. Checking Core Dumps / Stack Traces ==="
if [ -f "core" ]; then
    echo "Core dump found! Querying stack trace via GDB:"
    gdb -ex 'bt' -ex 'quit' $(which python3) core 2>/dev/null || echo "GDB trace failed."
else
    echo "No local core dump file found."
fi

echo "============================================"
echo "NVSHMEM Native Test Completed"
echo "============================================"
