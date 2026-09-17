# Formal Result Evidence

This directory contains a compact, reviewable copy of evidence from the full assignment runs. It intentionally excludes smoke tests, repeated epoch snapshots, downloaded datasets, caches, and model checkpoints.

The `metrics.json` files and cluster logs are copied from the completed runs without redacting the recorded paths, student identifier, job identifiers, node names, or environment details. The figures are the formal outputs referenced by the repository README.

## Included Evidence

| Part | Result summary | Included evidence |
| --- | --- | --- |
| Part 1: DFT | Maximum tensor/FFT error `3.29e-10` | Metrics, CSV timings, timing plot, spectrum, and square-wave reconstructions |
| Part 2: Eigenfaces | 150 components, `94.65%` explained variance, `65.22%` accuracy | Metrics, eigenfaces, compactness plot, and confusion matrix |
| Part 3.1: LFW CNN | `71.74%` accuracy, outperforming the Random Forest by `6.52` percentage points | Metrics, training curves, confusion matrix, and checkpoint-reload predictions |
| Part 3.2: CIFAR-10 ResNet | `94.12%` at epoch 35 in `121.29 s` wall time | Metrics, DAWNBench record, and training curves |
| Part 4.1: OASIS VAE | 256x256 images, two-dimensional latent space, best validation loss `16917.85` | Metrics, reconstruction grid, latent manifold, and training curves |
| Part 4.2: OASIS U-Net | `0.9775` mean test DSC; all foreground labels above `0.96` | Metrics, validation/test predictions, and training curves |
| Part 4.3: OASIS GAN | 80 epochs; generated/real diversity ratio `0.8919` | Metrics, real/generated samples, latent interpolation, and training curves |

## Cluster Logs

`cluster_logs/` contains the original Slurm standard-output and standard-error files from the completed 256x256 VAE run and the multi-model checkpoint-inference run. Empty `.err` files are retained because they document that the jobs produced no standard-error output.

## Excluded Binary Artefacts

Reloadable `.pt`, `.pth`, and `.joblib` files remain outside normal Git history. Several checkpoints are close to GitHub's 100 MiB per-file limit, and the GAN checkpoint exceeds it. The checkpoints are retained locally and on Rangpur for inference during the demonstration.

The complete generated `results/` tree remains the authoritative working copy. This directory is only the curated public evidence set.
