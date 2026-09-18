# COMP3710 Lab Demonstration 2

This repository contains reproducible PyTorch implementations for all four parts of the Pattern Recognition lab:

1. Square-wave reconstruction and explicit NumPy/tensor DFT timing.
2. NumPy SVD eigenfaces with a Random Forest classifier on LFW.
3. The required two-layer LFW CNN and a from-scratch CIFAR-10 ResNet-18.
4. OASIS VAE, categorical four-class UNet, and DCGAN tasks.

The programs save metrics, plots, checkpoints, reconstructions, segmentations, or generated samples under `results/`. Full datasets, generated working directories, and model checkpoints are excluded from normal Git history. A curated set of formal metrics, plots, inference outputs, and cluster logs is tracked under [`evidence/`](evidence/README.md).

## Verified formal results

| Part | Formal result | Evidence directory |
| --- | --- | --- |
| DFT | Maximum tensor/FFT error `3.29e-10` | [`evidence/part1_dft`](evidence/part1_dft) |
| Eigenfaces | 150 components, `94.65%` explained variance, `65.22%` Random Forest accuracy | [`evidence/part2_eigenfaces`](evidence/part2_eigenfaces) |
| LFW CNN | `71.74%` accuracy, `6.52` percentage points above Random Forest | [`evidence/part3_lfw_cnn`](evidence/part3_lfw_cnn) |
| CIFAR-10 | `94.12%` at epoch 35 in `121.29 s` wall time (`115.53 s` benchmark time) | [`evidence/part3_cifar_resnet`](evidence/part3_cifar_resnet) |
| OASIS VAE | 256x256, 40 epochs, `beta=1.0`, two-dimensional manifold; best validation loss `16917.85` | [`evidence/part4_vae`](evidence/part4_vae) |
| OASIS U-Net | `0.9775` mean test DSC; per-label DSC `[0.9993, 0.9652, 0.9659, 0.9796]` | [`evidence/part4_unet`](evidence/part4_unet) |
| OASIS GAN | 80 epochs in `432.42 s`; generated/real mean-pairwise-diversity ratio `0.8919` | [`evidence/part4_gan`](evidence/part4_gan) |

These values come from full Rangpur runs rather than smoke tests. The selected public evidence is copied without changing the recorded metrics. Full `results/` directories and reloadable checkpoints remain available locally and on Rangpur for the live demonstration.

## Setup

The local COMP3710 environment can be prepared with:

```bash
conda activate 3710torch
python -m pip install -e ".[test]"
python -m pytest
```

On Rangpur, activate the course environment already created for this project, then install the project:

```bash
source "$HOME/miniconda3/bin/activate"
conda activate comp3710lab2
cd "$HOME/COMP3710-Assignment2"
python -m pip install -e ".[test]"
```

The local OASIS copy is expected at `data/OASIS`. On Rangpur the code automatically discovers `/home/groups/comp3710/OASIS`. You can override either location with `--data-root` or the `OASIS_ROOT` environment variable.

## Part 1: DFT

```bash
sbatch slurm/dft.sh
```

The tensor implementation explicitly constructs the DFT through trigonometric tensor operations. It does not call `torch.fft`.

## Part 2: Eigenfaces

The first run downloads LFW into `data/`.

```bash
sbatch slurm/eigenfaces.sh
```

Outputs include the eigenface grid, PCA compactness curve, PCA arrays, Random Forest model, classification report, and accuracy.

## Part 3.1: LFW CNN

```bash
sbatch slurm/lfw_cnn.sh
```

The model has exactly two `3x3` convolutional layers with 32 filters per layer, followed by a dense classifier. The formal run reloads the best checkpoint, saves a classification report and confusion matrix, and compares its accuracy with Part 2.

## Part 3.2: CIFAR-10 DAWNBench-style challenge

```bash
python -m comp3710_lab2.part3_cifar_resnet --epochs 100
```

The implementation defines ResNet-18 from scratch and uses CIFAR augmentation, SGD with Nesterov momentum, warm-up plus cosine decay, mixup, label smoothing, and CUDA mixed precision. Run the full experiment on Rangpur:

```bash
sbatch slurm/cifar_resnet.sh
```

Evaluate a saved checkpoint:

```bash
python -m comp3710_lab2.part3_cifar_resnet \
  --checkpoint results/part3_cifar_resnet/best.pt --eval-only
```

Accuracy targets must be demonstrated using the saved `metrics.json` and checkpoint. The code does not claim a 90% or 94% result until a real training run achieves it.

The speed-challenge run keeps the decoded dataset on the GPU, performs crop/flip/cutout augmentation on the GPU, uses a fused optimiser, mixed precision, exponential moving-average weights, and flip test-time augmentation, and delays checkpoint writes until training ends:

```bash
sbatch slurm/cifar_resnet_fast.sh
```

It writes to `results/part3_cifar_resnet_fast` so that the reliable 100-epoch baseline is not overwritten. Its `metrics.json` separately records setup time, full wall time, model-update time, the first epoch reaching 94%, and the exact time to that target. `dawnbench.tsv` stores epoch, hours, and top-1 accuracy in the original benchmark format.

Re-evaluate the fast checkpoint with the same EMA-derived weights and horizontal-flip test-time augmentation:

```bash
python -m comp3710_lab2.part3_cifar_resnet_fast \
  --device cuda \
  --checkpoint results/part3_cifar_resnet_fast/best.pt \
  --eval-only
```

## Part 4 Task 1: OASIS VAE

```bash
python -m comp3710_lab2.part4_vae --image-size 256 --epochs 40 --latent-dim 2 --beta 1.0
```

The formal VAE run uses the native 256x256 OASIS resolution. Submit
`slurm/oasis_vae_256.sh`; its outputs are written to
`results/part4_vae_beta1_256`. The two-dimensional latent space produces
`manifold_best.png`; reconstructions and loss curves are saved beside it. The
standard VAE objective uses `beta=1.0` so that KL regularisation meaningfully
shapes the latent space.

## Part 4 Task 2: OASIS UNet

```bash
python -m comp3710_lab2.part4_unet --epochs 50
```

The dataset maps grayscale mask values `[0, 85, 170, 255]` to four channels. Targets are one-hot tensors and the loss combines categorical cross-entropy with soft Dice. Metrics report DSC for every label and explicitly record whether every test DSC exceeds 0.9.

Evaluate and visualise a checkpoint during the demonstration:

```bash
python -m comp3710_lab2.part4_unet \
  --checkpoint results/part4_unet/best.pt --eval-only
```

## Part 4 Task 3: OASIS GAN

```bash
python -m comp3710_lab2.part4_gan --epochs 80
```

The DCGAN saves a real-image reference grid, a fixed latent sample grid every five epochs, latent interpolations, a resumable checkpoint, discriminator confidence, loss curves, and generated-sample diversity metrics. The final metrics compare generated diversity with the real-image reference batch to support mode-collapse analysis, but realism and diversity must still be visually inspected.

## Quick smoke checks

These commands exercise complete training paths with small subsets. They verify execution, not final accuracy:

```bash
python -m comp3710_lab2.part4_vae --epochs 1 --limit 64 --image-size 64 --workers 0 --output-dir results/smoke_vae
python -m comp3710_lab2.part4_unet --epochs 1 --limit 8 --image-size 64 --base-channels 8 --batch-size 2 --workers 0 --output-dir results/smoke_unet
python -m comp3710_lab2.part4_gan --epochs 1 --limit 64 --image-size 64 --base-channels 16 --batch-size 16 --workers 0 --output-dir results/smoke_gan
python -m comp3710_lab2.part3_cifar_resnet --fake-data --epochs 1 --batch-size 16 --workers 0 --device cpu --output-dir results/smoke_cifar
```

## Evidence and demonstration

Keep the following generated evidence for the demonstrator:

- Terminal output and `metrics.json` from each completed experiment.
- Best checkpoints and training curves.
- Eigenfaces and PCA compactness plot.
- CIFAR-10 accuracy and elapsed time from Rangpur.
- VAE reconstructions and latent manifold.
- UNet per-label test DSC and segmentation examples.
- GAN samples across epochs and loss curves.
- Meaningful Git commit history that reflects work you can explain.

See `docs/DEMO_GUIDE.md` for the explanation checklist.

The use of AI assistance, validation steps, and responsibility boundaries is
documented in [docs/AI_ASSISTANCE_REPORT.md](docs/AI_ASSISTANCE_REPORT.md).

For checkpoint-loading inference across all trained neural models, plus the
required CIFAR one-epoch training demonstration, first obtain an interactive
`comp3710` GPU allocation and then run:

```bash
bash scripts/live_demo.sh summary
bash scripts/live_demo.sh lfw-infer
bash scripts/live_demo.sh cifar-infer
bash scripts/live_demo.sh cifar-train
bash scripts/live_demo.sh vae-infer
bash scripts/live_demo.sh unet-infer
bash scripts/live_demo.sh gan-infer
```

These commands save temporary demonstration outputs under `results/demo_*` and
do not overwrite formal results. `vae-infer` loads the formal 256x256 checkpoint.

Audit the formal evidence at any time with:

```bash
python scripts/check_demo_evidence.py
```

After the inference job and cluster logs have been downloaded, use the strict
check:

```bash
python scripts/check_demo_evidence.py \
  --require-demo-inference \
  --require-cluster-logs
```

## References

- Scikit-learn LFW/eigenfaces example: https://scikit-learn.org/stable/auto_examples/applications/plot_face_recognition.html
- PyTorch CIFAR-10 tutorial: https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
- ResNet paper: https://arxiv.org/abs/1512.03385
- DAWNBench CIFAR-10 submission format and timing definition: https://github.com/stanford-futuredata/dawn-bench-entries
- David Page fast CIFAR-10 recipe, standard PyTorch transcription: https://gist.github.com/KellerJordan/a3cd4b76c531420d8033ece025bc3e9c
- Jordan, "94% on CIFAR-10 in 3.29 Seconds on a Single GPU": https://arxiv.org/abs/2404.00498
- VAE paper: https://arxiv.org/abs/1312.6114
- UNet paper: https://arxiv.org/abs/1505.04597
- DCGAN paper: https://arxiv.org/abs/1511.06434
