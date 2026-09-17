#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-$HOME/miniconda3/envs/comp3710lab2/bin/python}"
DEMO_DEVICE="${DEMO_DEVICE:-cuda}"
LFW_DATA_HOME="${LFW_DATA_HOME:-data}"
OASIS_DATA_ROOT="${OASIS_DATA_ROOT:-/home/groups/comp3710/OASIS}"
DEMO_WORKERS="${DEMO_WORKERS:-4}"
MODE="${1:-help}"

cd "$PROJECT_DIR"
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="${MPLCONFIGDIR:-$HOME/tmp/matplotlib}"
mkdir -p "$MPLCONFIGDIR"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python environment not found: $PYTHON_BIN" >&2
  exit 1
fi

require_file() {
  if [[ ! -f "$1" ]]; then
    echo "Required file is missing: $1" >&2
    exit 1
  fi
}

require_device() {
  "$PYTHON_BIN" - "$DEMO_DEVICE" <<'PY'
import sys
import torch

requested = sys.argv[1]
if requested == "cuda" and not torch.cuda.is_available():
    raise SystemExit(
        "CUDA is unavailable. Start an interactive comp3710 GPU allocation first."
    )
print("PyTorch:", torch.__version__)
print("Device:", requested)
if requested == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))
PY
}

show_summary() {
  "$PYTHON_BIN" - <<'PY'
import json
from pathlib import Path


def read(name):
    path = Path("results") / name / "metrics.json"
    if not path.is_file():
        print(f"{name}: missing {path}")
        return None
    return json.loads(path.read_text())


dft = read("part1_dft")
if dft:
    largest = dft["timings"][-1]
    print(
        "Part 1 DFT:",
        f"device={dft['device']}",
        f"N={largest['size']}",
        f"FFT={largest['numpy_fft_seconds']:.6f}s",
        f"tensor={largest['tensor_dft_seconds']:.6f}s",
        f"naive={largest['naive_numpy_seconds']:.6f}s",
    )

eigenfaces = read("part2_eigenfaces")
if eigenfaces:
    print(
        "Part 2 Eigenfaces:",
        f"accuracy={eigenfaces['accuracy']:.4%}",
        f"components={eigenfaces['components']}",
        f"variance={eigenfaces['variance_explained']:.4%}",
    )

lfw = read("part3_lfw_cnn")
if lfw:
    print(
        "Part 3.1 LFW CNN:",
        f"accuracy={lfw['best_test_accuracy']:.4%}",
        f"RF={lfw['random_forest_accuracy']:.4%}",
        f"improvement={lfw['cnn_minus_random_forest_accuracy']:.4%}",
    )

cifar = read("part3_cifar_resnet_fast")
if cifar:
    print(
        "Part 3.2 CIFAR-10:",
        f"accuracy={cifar['best_test_accuracy']:.4%}",
        f"target_epoch={cifar['target_epoch']}",
        f"wall={cifar['time_to_target_seconds']:.3f}s",
        f"AMP={cifar['amp']}",
    )

unet = read("part4_unet")
if unet:
    scores = ", ".join(f"{score:.4f}" for score in unet["test"]["dice_per_class"])
    print(
        "Part 4 U-Net:",
        f"mean_DSC={unet['test']['mean_dice']:.4f}",
        f"per_label=[{scores}]",
        f"all_above_0.9={unet['all_test_labels_above_0_9']}",
    )
PY
}

run_cifar_inference() {
  require_device
  require_file results/part3_cifar_resnet_fast/best.pt
  "$PYTHON_BIN" -m comp3710_lab2.part3_cifar_resnet_fast \
    --data-dir data \
    --device "$DEMO_DEVICE" \
    --batch-size 512 \
    --checkpoint results/part3_cifar_resnet_fast/best.pt \
    --eval-only \
    --output-dir results/demo_cifar_inference
}

run_cifar_epoch() {
  require_device
  require_file results/part3_cifar_resnet_fast/best.pt
  "$PYTHON_BIN" -m comp3710_lab2.part3_cifar_resnet_fast \
    --data-dir data \
    --device "$DEMO_DEVICE" \
    --checkpoint results/part3_cifar_resnet_fast/best.pt \
    --epochs 1 \
    --batch-size 512 \
    --learning-rate 0.01 \
    --mixup-alpha 0.0 \
    --label-smoothing 0.1 \
    --cutout-size 8 \
    --ema-decay 0.99 \
    --ema-update-every 5 \
    --target-accuracy 0.94 \
    --output-dir results/demo_cifar_one_epoch
}

run_unet_inference() {
  require_device
  require_file results/part4_unet/best.pt
  "$PYTHON_BIN" -m comp3710_lab2.part4_unet \
    --data-root "$OASIS_DATA_ROOT" \
    --device "$DEMO_DEVICE" \
    --checkpoint results/part4_unet/best.pt \
    --eval-only \
    --batch-size 8 \
    --workers "$DEMO_WORKERS" \
    --output-dir results/demo_unet_inference
}

run_lfw_inference() {
  require_device
  require_file results/part3_lfw_cnn/best.pt
  "$PYTHON_BIN" -m comp3710_lab2.part3_lfw_cnn \
    --data-home "$LFW_DATA_HOME" \
    --device "$DEMO_DEVICE" \
    --checkpoint results/part3_lfw_cnn/best.pt \
    --eval-only \
    --batch-size 64 \
    --workers "$DEMO_WORKERS" \
    --output-dir results/demo_lfw_inference
}

run_vae_inference() {
  require_device
  local checkpoint=results/part4_vae_beta1_256/best.pt
  require_file "$checkpoint"
  echo "VAE checkpoint: $checkpoint (256x256)"
  "$PYTHON_BIN" -m comp3710_lab2.part4_vae \
    --data-root "$OASIS_DATA_ROOT" \
    --device "$DEMO_DEVICE" \
    --image-size 256 \
    --latent-dim 2 \
    --checkpoint "$checkpoint" \
    --eval-only \
    --batch-size 16 \
    --workers "$DEMO_WORKERS" \
    --output-dir results/demo_vae_inference
}

run_gan_inference() {
  require_device
  require_file results/part4_gan/checkpoint_latest.pt
  "$PYTHON_BIN" -m comp3710_lab2.part4_gan \
    --device "$DEMO_DEVICE" \
    --image-size 128 \
    --latent-dim 128 \
    --base-channels 64 \
    --checkpoint results/part4_gan/checkpoint_latest.pt \
    --sample-only \
    --output-dir results/demo_gan_inference
}

case "$MODE" in
  summary)
    show_summary
    ;;
  cifar-infer)
    run_cifar_inference
    ;;
  cifar-train)
    run_cifar_epoch
    ;;
  lfw-infer)
    run_lfw_inference
    ;;
  vae-infer)
    run_vae_inference
    ;;
  unet-infer)
    run_unet_inference
    ;;
  gan-infer)
    run_gan_inference
    ;;
  all)
    show_summary
    run_lfw_inference
    run_cifar_inference
    run_cifar_epoch
    run_vae_inference
    run_unet_inference
    run_gan_inference
    ;;
  help|-h|--help)
    cat <<'EOF'
Usage: bash scripts/live_demo.sh MODE

Modes:
  summary      Show verified formal metrics; GPU is not required.
  lfw-infer    Load the LFW CNN checkpoint and save labelled predictions.
  cifar-infer  Load the saved CIFAR checkpoint and run test inference.
  cifar-train  Run one CIFAR training epoch without changing formal results.
  vae-infer    Load the best available VAE and save reconstructions/manifold.
  unet-infer   Run U-Net test inference and save a prediction figure.
  gan-infer    Load the GAN checkpoint and save samples/interpolations.
  all          Run all of the above in order.

Set DEMO_DEVICE=cpu to run without a GPU. CUDA is the default.
EOF
    ;;
  *)
    echo "Unknown mode: $MODE" >&2
    exit 2
    ;;
esac
