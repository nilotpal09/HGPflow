#!/usr/bin/env bash
# COCOA smoke: Mean Shift segmentation -> public checkpoint inference -> output check.
#
# Run from anywhere; paths are resolved from this repository.
# Required environment variables:
#   HGPFLOW_PYTHON        python with torch/flash_attn/uproot (see smoke/ENVIRONMENT.md)
#   COCOA_DATA            directory holding the COCOA Zenodo ROOT files (read only)
#   HGPFLOW_RUN_DIR       new run directory (must not exist)
#   CUDA_VISIBLE_DEVICES  exactly one GPU (index or UUID); must be idle at start
# Optional (defaults are the smoke limits):
#   SMOKE_ENTRY_START=0 SMOKE_ENTRY_STOP=10 SMOKE_CPUS=0-15 SMOKE_CPU_QUOTA=1600%
#   SMOKE_MEMORY_MAX=32G SMOKE_STEP_TIMEOUT=30m SMOKE_DISK_MAX_MIB=1024
set -euo pipefail

: "${HGPFLOW_PYTHON:?}" "${COCOA_DATA:?}" "${HGPFLOW_RUN_DIR:?}" "${CUDA_VISIBLE_DEVICES:?}"
ENTRY_START=${SMOKE_ENTRY_START:-0}
ENTRY_STOP=${SMOKE_ENTRY_STOP:-10}
CPUS=${SMOKE_CPUS:-0-15}
CPU_QUOTA=${SMOKE_CPU_QUOTA:-1600%}
MEMORY_MAX=${SMOKE_MEMORY_MAX:-32G}
STEP_TIMEOUT=${SMOKE_STEP_TIMEOUT:-30m}
DISK_MAX_MIB=${SMOKE_DISK_MAX_MIB:-1024}

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
SOURCE="$COCOA_DATA/dijet_test.root"
CONFIG=configs/cocoa/smoke/inference_dijet_test_first10.yml
CKPT_DIR=saved_checkpoints/cocoaxminixv1x250kxs2xxxtgxrwsik
TAG="dijet_test_e${ENTRY_START}-${ENTRY_STOP}_seg_bw0.4"

if [[ "$CUDA_VISIBLE_DEVICES" == *,* ]]; then
    echo "CUDA_VISIBLE_DEVICES must name exactly one GPU" >&2; exit 2
fi
if [[ -e "$HGPFLOW_RUN_DIR" ]]; then
    echo "run directory already exists, refusing to overwrite: $HGPFLOW_RUN_DIR" >&2; exit 2
fi
if [[ -n "$(git -C "$REPO" status --porcelain)" ]]; then
    echo "implementation checkout is dirty; commit before a recorded run" >&2; exit 2
fi
if [[ "$ENTRY_START" != 0 || "$ENTRY_STOP" != 10 ]]; then
    echo "the committed inference config is fixed to entries 0-10" >&2; exit 2
fi
gpu_used=$(nvidia-smi --id="$CUDA_VISIBLE_DEVICES" --query-gpu=memory.used --format=csv,noheader,nounits)
if [[ "$gpu_used" -ne 0 ]]; then
    echo "selected GPU is not idle (memory.used=${gpu_used} MiB)" >&2; exit 2
fi

mkdir -p "$HGPFLOW_RUN_DIR"/{seg,pred,logs,env}
export HGPFLOW_RUN_DIR
LOG="$HGPFLOW_RUN_DIR/logs"
ENV="$HGPFLOW_RUN_DIR/env"

# each step: cgroup scope (memory, CPU quota) + CPU affinity + wall-time limit
step() {
    local name=$1; shift
    echo "[$(date -Is)] start $name" | tee -a "$LOG/steps.log"
    local rc=0
    systemd-run --user --scope --quiet -p MemoryMax="$MEMORY_MAX" -p MemorySwapMax=0 \
        -p CPUQuota="$CPU_QUOTA" \
        /usr/bin/time -v -o "$LOG/$name.time" \
        taskset -c "$CPUS" timeout --signal=TERM --kill-after=30s "$STEP_TIMEOUT" "$@" \
        > "$LOG/$name.log" 2>&1 || rc=$?
    echo "[$(date -Is)] end $name exit=$rc" | tee -a "$LOG/steps.log"
    return $rc
}

{
    echo "started_at=$(date -Is)"
    echo "host=$(hostname)"
    echo "repo_commit=$(git -C "$REPO" rev-parse HEAD)"
    echo "cuda_visible_devices=$CUDA_VISIBLE_DEVICES"
    echo "entry_range=[$ENTRY_START,$ENTRY_STOP)"
    echo "limits: cpus=$CPUS cpu_quota=$CPU_QUOTA memory_max=$MEMORY_MAX step_timeout=$STEP_TIMEOUT disk_max_mib=$DISK_MAX_MIB"
} > "$ENV/run_info.txt"
uname -a > "$ENV/uname.txt"
cat /etc/os-release > "$ENV/os-release.txt"
"$HGPFLOW_PYTHON" -m pip freeze > "$ENV/pip_freeze.txt" 2>/dev/null
"$HGPFLOW_PYTHON" -c "import sys, torch, flash_attn, uproot, awkward, numpy, sklearn, pytorch_lightning as pl
print('python', sys.version.replace(chr(10), ' '))
print('torch', torch.__version__, 'cuda', torch.version.cuda, 'cudnn', torch.backends.cudnn.version())
print('flash_attn', flash_attn.__version__, 'pytorch_lightning', pl.__version__)
print('uproot', uproot.__version__, 'awkward', awkward.__version__, 'numpy', numpy.__version__, 'sklearn', sklearn.__version__)" > "$ENV/versions.txt"
nvidia-smi --id="$CUDA_VISIBLE_DEVICES" --query-gpu=index,uuid,name,driver_version,memory.total,memory.used --format=csv > "$ENV/gpu.csv"
(cd "$REPO" && sha256sum "$CONFIG" "$CKPT_DIR"/config_*.yml "$CKPT_DIR"/checkpoints/*.ckpt) > "$ENV/inputs.sha256"
stat -c '%n %s %Y' "$SOURCE" > "$ENV/source_stat.txt"

cd "$REPO/event_segmentation"
step segmentation "$HGPFLOW_PYTHON" run_cocoa_event_splitter.py \
    -i "$SOURCE" -t Out_Tree -estart "$ENTRY_START" -estop "$ENTRY_STOP" -bw 0.4 -is_inf \
    -o "$HGPFLOW_RUN_DIR/seg/$TAG.root"

cd "$REPO"
nvidia-smi --id="$CUDA_VISIBLE_DEVICES" --query-gpu=memory.used --format=csv,noheader,nounits > "$ENV/gpu_used_before_inference.txt"
nvidia-smi --id="$CUDA_VISIBLE_DEVICES" --query-gpu=timestamp,memory.used,utilization.gpu --format=csv -l 1 > "$LOG/gpu_monitor.csv" &
MON=$!
trap 'kill "$MON" 2>/dev/null || true' EXIT
export HGPFLOW_BATCH_ORDER_PATH="$HGPFLOW_RUN_DIR/pred/batch_order.npz"
step inference "$HGPFLOW_PYTHON" -m hgpflow_v2.eval -i "$CONFIG"
kill "$MON" 2>/dev/null || true

step check "$HGPFLOW_PYTHON" smoke/check_cocoa_smoke.py \
    --source "$SOURCE" --entry-start "$ENTRY_START" --entry-stop "$ENTRY_STOP" \
    --seg "$HGPFLOW_RUN_DIR/seg/$TAG.root" \
    --pred "$HGPFLOW_RUN_DIR/pred/pred_${TAG}_merged.root" \
    --config-v "$CKPT_DIR/config_v.yml" \
    --output "$HGPFLOW_RUN_DIR/check_report.json"

used_mib=$(du -sm "$HGPFLOW_RUN_DIR" | cut -f1)
echo "run_dir_mib=$used_mib" >> "$ENV/run_info.txt"
echo "ended_at=$(date -Is)" >> "$ENV/run_info.txt"
if (( used_mib > DISK_MAX_MIB )); then
    echo "run directory exceeds disk limit: ${used_mib} MiB > ${DISK_MAX_MIB} MiB" >&2; exit 3
fi
(cd "$HGPFLOW_RUN_DIR" && find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS)
echo "smoke finished: $HGPFLOW_RUN_DIR"
