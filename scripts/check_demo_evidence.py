#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


FORMAL_ARTIFACTS = {
    "Part 1 DFT": (
        "results/part1_dft/metrics.json",
        "results/part1_dft/dft_timing.png",
        "results/part1_dft/dft_frequency_spectrum.png",
        "results/part1_dft/square_wave_reconstructions.png",
    ),
    "Part 2 Eigenfaces": (
        "results/part2_eigenfaces/metrics.json",
        "results/part2_eigenfaces/eigenfaces.png",
        "results/part2_eigenfaces/compactness.png",
        "results/part2_eigenfaces/confusion_matrix.png",
        "results/part2_eigenfaces/pca.npz",
        "results/part2_eigenfaces/random_forest.joblib",
    ),
    "Part 3.1 LFW CNN": (
        "results/part3_lfw_cnn/metrics.json",
        "results/part3_lfw_cnn/best.pt",
        "results/part3_lfw_cnn/training_curves.png",
        "results/part3_lfw_cnn/confusion_matrix.png",
    ),
    "Part 3.2 CIFAR ResNet": (
        "results/part3_cifar_resnet_fast/metrics.json",
        "results/part3_cifar_resnet_fast/best.pt",
        "results/part3_cifar_resnet_fast/training_curves.png",
    ),
    "Part 4.1 VAE": (
        "results/part4_vae_beta1_256/metrics.json",
        "results/part4_vae_beta1_256/best.pt",
        "results/part4_vae_beta1_256/reconstructions_latest.png",
        "results/part4_vae_beta1_256/manifold_best.png",
        "results/part4_vae_beta1_256/training_curves.png",
    ),
    "Part 4.2 U-Net": (
        "results/part4_unet/metrics.json",
        "results/part4_unet/best.pt",
        "results/part4_unet/training_curves.png",
        "results/part4_unet/test_predictions.png",
    ),
    "Part 4.3 GAN": (
        "results/part4_gan/metrics.json",
        "results/part4_gan/checkpoint_latest.pt",
        "results/part4_gan/training_curves.png",
        "results/part4_gan/samples_latest.png",
        "results/part4_gan/latent_interpolation.png",
    ),
}

DEMO_ARTIFACTS = (
    "results/demo_lfw_inference/test_predictions.png",
    "results/demo_vae_inference/reconstructions.png",
    "results/demo_vae_inference/manifold.png",
    "results/demo_gan_inference/generated_samples.png",
    "results/demo_gan_inference/latent_interpolation.png",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit COMP3710 Demo 2 evidence")
    parser.add_argument("--require-demo-inference", action="store_true")
    parser.add_argument("--require-cluster-logs", action="store_true")
    return parser.parse_args()


def read_json(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def check_group(name: str, paths: tuple[str, ...], missing: list[str]) -> None:
    absent = [path for path in paths if not Path(path).is_file()]
    if absent:
        print(f"[MISSING] {name}")
        for path in absent:
            print(f"  - {path}")
        missing.extend(absent)
    else:
        print(f"[OK] {name}: {len(paths)} artifacts")


def check_metrics(failures: list[str]) -> None:
    vae = read_json("results/part4_vae_beta1_256/metrics.json")
    if vae.get("image_size") == 256 and vae.get("latent_dim") == 2:
        print("[OK] VAE configuration: image_size=256, latent_dim=2")
    else:
        message = "VAE metrics do not record image_size=256 and latent_dim=2"
        print(f"[FAIL] {message}")
        failures.append(message)

    cifar = read_json("results/part3_cifar_resnet_fast/metrics.json")
    accuracy = float(cifar["best_test_accuracy"])
    duration = float(cifar["time_to_target_seconds"])
    if accuracy >= 0.94 and duration <= 360:
        print(f"[OK] CIFAR target: {accuracy:.2%} in {duration:.3f}s")
    else:
        message = f"CIFAR target failed: {accuracy:.2%} in {duration:.3f}s"
        print(f"[FAIL] {message}")
        failures.append(message)

    unet = read_json("results/part4_unet/metrics.json")
    foreground = [float(value) for value in unet["test"]["dice_per_class"][1:]]
    if len(foreground) == 3 and all(value > 0.9 for value in foreground):
        scores = ", ".join(f"{value:.4f}" for value in foreground)
        print(f"[OK] U-Net foreground DSC: {scores}")
    else:
        message = f"U-Net foreground DSC failed: {foreground}"
        print(f"[FAIL] {message}")
        failures.append(message)


def main() -> None:
    args = parse_args()
    missing: list[str] = []
    failures: list[str] = []

    for name, paths in FORMAL_ARTIFACTS.items():
        check_group(name, paths, missing)
    check_metrics(failures)

    if args.require_demo_inference:
        check_group("Checkpoint inference outputs", DEMO_ARTIFACTS, missing)

    logs = sorted(Path.cwd().glob("*.out")) + sorted(Path.cwd().glob("*.err"))
    if logs:
        print(f"[OK] Cluster logs: {len(logs)} files")
    elif args.require_cluster_logs:
        message = "No cluster .out/.err logs found in the project root"
        print(f"[MISSING] {message}")
        missing.append(message)
    else:
        print("[PENDING] Cluster logs have not been copied to this project")

    print()
    if missing or failures:
        print(f"Evidence audit failed: {len(missing)} missing, {len(failures)} invalid")
        raise SystemExit(1)
    print("Evidence audit passed")


if __name__ == "__main__":
    main()
