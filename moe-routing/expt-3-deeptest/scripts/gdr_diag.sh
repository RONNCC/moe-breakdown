#!/bin/bash
#SBATCH --partition=ice-gpu
#SBATCH --qos=coc-ice
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:05:00
#SBATCH --job-name=gdr-diag
#SBATCH --output=/home/hice1/sghose7/scratch/moe-breakdown-runs/expt-3-deeptest/gdr-diag-%j.out
#SBATCH --error=/home/hice1/sghose7/scratch/moe-breakdown-runs/expt-3-deeptest/gdr-diag-%j.err

echo "============================================"
echo "GPUDirect RDMA Diagnostic Report"
echo "Node: \$(hostname)"
echo "Date: \$(date)"
echo "============================================"

echo ""
echo "=== 1. Kernel Modules ==="
echo "--- nvidia_peermem / nv_peer_mem ---"
lsmod | grep -E 'nvidia_peermem|nv_peer_mem' || echo "PEERMEM NOT LOADED"
echo "--- gdrdrv ---"
lsmod | grep gdrdrv || echo "GDRDRV NOT LOADED"
echo "--- ib_core / mlx5 ---"
lsmod | grep -E 'ib_core|mlx5_ib|mlx5_core' || echo "IB MODULES NOT LOADED"

echo ""
echo "=== 2. InfiniBand Device Access ==="
echo "--- uverbs device nodes ---"
ls -la /dev/infiniband/uverbs* 2>/dev/null || echo "uverbs DEVICES NOT FOUND"
echo "--- nvidia-caps ---"
ls -la /dev/nvidia-caps/ 2>/dev/null || echo "nvidia-caps NOT ACCESSIBLE"
echo "--- memory lock limit ---"
ulimit -l

echo ""
echo "=== 3. IB Device & Port Status ==="
ibv_devices 2>/dev/null || echo "ibv_devices FAILED"
echo "---"
ibv_devinfo 2>/dev/null || echo "ibv_devinfo FAILED"
echo "---"
ibstat 2>/dev/null || echo "ibstat FAILED"

echo ""
echo "=== 4. GPU & Driver Info ==="
nvidia-smi --query-gpu=driver_version,name --format=csv,noheader
echo "--- GPU Topology ---"
nvidia-smi topo -m

echo ""
echo "=== 5. dmesg peermem/gdr ==="
dmesg 2>/dev/null | grep -iE 'peermem|gdr|nv_peer' || echo "No relevant dmesg entries (or no dmesg access)"

echo ""
echo "============================================"
echo "Diagnostic complete."
echo "============================================"
