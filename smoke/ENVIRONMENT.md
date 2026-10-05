# Smoke environment

The COCOA checkpoint `cocoaxminixv1x250kxs2xxxtgxrwsik` uses `enable_flash_attn: True`
in the stage-1 transformer, so inference needs `flash_attn`.

Base: the conda env `hgflow` (Python 3.10.14, torch 2.3.1+cu118, pytorch-lightning 1.8.6,
uproot 5.3.1, awkward 2.6.6). Explicit spec with md5: `env/hgflow-conda-explicit.txt`
(pip-installed packages in that env are listed in `env/pip-versions.txt`).

Overlay (keeps the base env untouched):

```bash
python -m venv --system-site-packages "$VENV"      # python = hgflow env python
"$VENV/bin/python" -m pip install --no-deps --require-hashes -r smoke/env/overlay-requirements.txt
```

`env/pip-versions.txt` is `pip list --format=freeze` of the resulting venv.
Not needed for this smoke and not installed: comet_ml (training), vector/pathos/fastjet
(performance helpers), seaborn (live plotting).

Checked on A100 80GB PCIe, driver 615.71.09 (CUDA 13.4 UMD) with the cu118 torch build.

## Run

```bash
export HGPFLOW_PYTHON="$VENV/bin/python" COCOA_DATA=/path/to/cocoa HGPFLOW_RUN_DIR=/path/to/new_run
CUDA_VISIBLE_DEVICES=<one idle GPU UUID> smoke/run_cocoa_smoke.sh
```

The script refuses an existing run directory, a dirty checkout, more than one GPU or a busy GPU.
Each step runs in a systemd user scope (MemoryMax, CPUQuota), with taskset and a wall-time timeout.
Outputs: `seg/` (Mean Shift partitions), `pred/` (event_tree + RNG state), `logs/`, `env/`,
`check_report.json` (`output_valid`), `SHA256SUMS`.
