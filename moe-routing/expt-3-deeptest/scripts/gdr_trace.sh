#!/bin/bash
#SBATCH --partition=ice-gpu
#SBATCH --qos=coc-ice
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:05:00
#SBATCH --job-name=gdr-trace
#SBATCH --output=/home/hice1/sghose7/scratch/moe-breakdown-runs/expt-3-deeptest/gdr-trace-%j.out
#SBATCH --error=/home/hice1/sghose7/scratch/moe-breakdown-runs/expt-3-deeptest/gdr-trace-%j.err

echo "============================================"
echo "NVSHMEM & DeepEP Low-Level Software Tracing"
echo "Host: $(hostname -s)"
echo "Date: $(date)"
echo "============================================"

# Load modules
echo "[1/6] Loading Environment Modules..."
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

NVSHMEM_LIB_DIR="/storage/ice1/0/2/sghose7/moe-breakdown-deepep-venv/lib/python3.11/site-packages/nvidia/nvshmem/lib"
echo "NVSHMEM library dir: $NVSHMEM_LIB_DIR"

echo ""
echo "=== 2. Check NVSHMEM Version & Build Metadata ==="
# Check if nvshmem-info is available in the environment path or bin
if [ -d "/storage/ice1/0/2/sghose7/moe-breakdown-deepep-venv/bin" ]; then
    export PATH="/storage/ice1/0/2/sghose7/moe-breakdown-deepep-venv/bin:$PATH"
fi

if which nvshmem-info >/dev/null 2>&1; then
    echo "nvshmem-info found! Version information:"
    nvshmem-info -n || true
    echo ""
    echo "Detailed Build Configuration:"
    nvshmem-info -a || true
else
    echo "nvshmem-info not found in PATH. Extracting build metadata from libnvshmem_host.so via strings..."
    strings "$NVSHMEM_LIB_DIR/libnvshmem_host.so.3" | grep -iE 'cuda|version|build|nvshmem' | head -n 30 || echo "No metadata found via strings."
fi

echo ""
echo "=== 3. Run ldd on NVSHMEM Dynamic Transport Libraries ==="
for lib in "$NVSHMEM_LIB_DIR"/*.so*; do
    echo "--- ldd on $(basename "$lib") ---"
    ldd "$lib" || echo "ldd failed on $lib"
done

echo ""
echo "=== 4. Check for Library Shadowing and Symbol Conflicts ==="
# Check where Python resolves libibverbs and libmlx5
echo "--- Dynamic Linker Search Paths with LD_DEBUG=libs ---"
# We first run WITH the NCCL prepend if nvidia.nccl is installed
if python -c "import nvidia.nccl" 2>/dev/null; then
    NCCL_LIB_DIR=$(python -c "import nvidia.nccl, os; base = getattr(nvidia.nccl, '__file__', None) or list(nvidia.nccl.__path__)[0]; print(os.path.join(os.path.dirname(base), 'lib'))")
    echo "NCCL_LIB_DIR found: $NCCL_LIB_DIR"
    echo "Running with LD_LIBRARY_PATH=NCCL_LIB_DIR..."
    LD_LIBRARY_PATH="$NCCL_LIB_DIR:$LD_LIBRARY_PATH" LD_DEBUG=files python3 -c "import torch" 2>&1 | grep -E 'libnccl|libibverbs|libmlx5' | head -n 30
else
    echo "nvidia.nccl not installed. Scanning standard dynamic load..."
    LD_DEBUG=files python3 -c "import torch" 2>&1 | grep -E 'libnccl|libibverbs|libmlx5' | head -n 30
fi

echo ""
echo "=== 5. Trace NVSHMEM Initialization with VERBOSE Logging ==="
export NVSHMEM_DEBUG=TRACE
export NVSHMEM_DEBUG_SUBSYS=ALL
export NVSHMEM_DEBUG_FILE="/tmp/nvshmem_trace_run.log"
rm -f "$NVSHMEM_DEBUG_FILE"

echo "Initializing Python NVSHMEM & DeepEP test..."
# Run a quick python import and distributed load to trace the exact initialization/crash boundary
python3 -c "
try:
    import torch
    import torch.distributed as dist
    print('Initializing PyTorch distributed (NCCL)...')
    dist.init_process_group('nccl', rank=0, world_size=1)
    
    print('Importing nvidia.nvshmem...')
    import nvidia.nvshmem
    print('✅ nvidia.nvshmem imported successfully!')
    
    print('Importing deep_ep...')
    import deep_ep
    print('✅ deep_ep imported successfully!')
except Exception as e:
    print('❌ Diagnostic import or initialization crashed:', e)
"

echo ""
echo "--- NVSHMEM Trace Log File (Last 60 Lines) ---"
if [ -f "$NVSHMEM_DEBUG_FILE" ]; then
    cat "$NVSHMEM_DEBUG_FILE" | tail -n 60
else
    echo "NVSHMEM debug file was not created (no trace written)."
fi

echo ""
echo "=== 6. Check libtorch_nvshmem Symbols & Links ==="
TORCH_NVSHMEM="/storage/ice1/0/2/sghose7/moe-breakdown-deepep-venv/lib/python3.11/site-packages/torch/lib/libtorch_nvshmem.so"
if [ -f "$TORCH_NVSHMEM" ]; then
    echo "Found libtorch_nvshmem.so. Dynamic dependencies:"
    ldd "$TORCH_NVSHMEM" | grep -E 'nvshmem|nccl|cuda' || true
else
    echo "libtorch_nvshmem.so not found."
fi

echo "============================================"
echo "Diagnostics and Tracing Complete"
echo "============================================"
