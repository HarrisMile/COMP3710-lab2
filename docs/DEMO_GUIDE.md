# Demonstration Guide

Use this guide to prepare explanations. Do not present a metric as achieved unless the corresponding result file was produced by a full run.

## Verified formal results

- Part 1: explicit tensor DFT ran on CUDA; at `N=2048`, NumPy FFT took `0.000102 s`, tensor DFT took `0.000371 s`, and naive NumPy DFT took `4.123061 s`.
- Part 2: 150 eigenfaces explained `94.65%` of the training variance; Random Forest test accuracy was `65.22%`.
- Part 3.1: the required two-layer CNN reached `71.74%`, improving on Random Forest by `6.52` percentage points.
- Part 3.2: CIFAR-10 ResNet-18 reached `94.12%` at epoch 35 in `121.29 s` wall time with CUDA mixed precision.
- Part 4 VAE: the formal 256x256, `beta=1.0`, two-dimensional run is in `results/part4_vae_beta1_256`; its best validation loss was `16917.85` and it provides the final manifold.
- Part 4 U-Net: test mean DSC was `0.9775`; per-label DSC was `[0.9993, 0.9652, 0.9659, 0.9796]`, so every label exceeded `0.9`.
- Part 4 GAN: the 80-epoch OASIS run completed in `432.42 s`. Generated sample pixel variation was `87.85%` of the real reference and mean pairwise diversity was `89.19%` of the real reference.

## Rangpur live demonstration

From `login0`, request an interactive GPU shell. Do this before the demonstration because the allocation may queue:

```bash
cd "$HOME/COMP3710-Assignment2"
srun --partition=comp3710 --account=comp3710 --gres=gpu:1 \
  --time=00:20:00 --pty bash
```

After the prompt changes to an `a100-*` node, run the demonstrations separately so each result is easy to explain:

```bash
cd "$HOME/COMP3710-Assignment2"
bash scripts/live_demo.sh summary
bash scripts/live_demo.sh cifar-infer
bash scripts/live_demo.sh cifar-train
bash scripts/live_demo.sh unet-infer
```

The live commands write only to `results/demo_*`; they do not change the formal checkpoints or metrics. `cifar-train` starts from the verified checkpoint and performs one additional epoch with a small learning rate. `unet-infer` writes `results/demo_unet_inference/test_predictions.png` and `test_metrics.json`.

## Part 1: DFT

- A square wave contains odd harmonics whose amplitudes decrease as `1 / n`.
- More harmonics sharpen edges but retain overshoot near discontinuities (the Gibbs phenomenon).
- The direct DFT is `O(N^2)` because every output frequency uses every input sample.
- FFT is `O(N log N)`. The tensor DFT is still `O(N^2)`, but a GPU can evaluate the matrix operations in parallel.
- The explicit NumPy and tensor results are checked numerically against `numpy.fft.fft`.

## Part 2: Eigenfaces

- Each flattened face is centred by subtracting the training mean face.
- SVD gives right singular vectors that form the PCA basis; reshaping them gives eigenfaces.
- Projection uses `(X - mean) @ components.T`.
- The compactness curve is cumulative explained variance and justifies the selected dimensionality.
- Random Forest operates on PCA features, not raw pixels.
- PCA is linear and task-independent; it preserves variance, not necessarily identity information.

## Part 3.1: LFW CNN

- Input shape is `N x 1 x H x W` and pixel values are normalised.
- Both required convolutional layers use 32 filters and `3x3` kernels.
- ReLU adds non-linearity; max pooling reduces spatial resolution; the dense layer produces class logits.
- Cross-entropy trains all feature extraction and classification parameters end to end.
- Compare test accuracy with the PCA plus Random Forest result using the same split seed.

## Part 3.2: CIFAR-10 ResNet-18

- Residual blocks learn `F(x) + x`, improving gradient flow in deeper networks.
- The CIFAR stem uses a `3x3`, stride-1 convolution because images are only `32x32`.
- The model is defined locally and starts from random weights; no pretrained model is used.
- Crop, flip, AutoAugment, random erasing, mixup, and label smoothing regularise training.
- Warm-up stabilises the initially large learning rate; cosine decay supports late convergence.
- Mixed precision uses faster CUDA tensor cores and reduces GPU memory use.
- The speed run removes the CPU input bottleneck by caching the small CIFAR-10 dataset on the GPU and applying crop, flip, and cutout there.
- Checkpoint files are written after training so network-home storage latency is excluded from the optimisation loop.
- Per-batch losses and accuracies remain as GPU tensors until the epoch ends, avoiding repeated CPU/GPU synchronisation.
- Exponential moving-average weights and horizontal-flip test-time augmentation improve the short-run accuracy used for the speed target.
- `time_to_target_seconds` records the first measured point at which test accuracy reaches 94%; keep the Slurm elapsed time as independent wall-clock evidence.
- Show the saved accuracy, elapsed seconds, checkpoint inference, and one training epoch on Rangpur.

## Part 4 Task 1: VAE

- The encoder predicts latent mean and log variance.
- Reparameterisation samples `z = mean + std * epsilon`, keeping the path differentiable.
- Reconstruction loss preserves image content; KL divergence regularises the latent distribution toward a unit Gaussian.
- A two-dimensional latent space can be sampled as a grid to display a direct manifold.
- Show original/reconstructed pairs, the manifold, and both reconstruction and KL curves.

## Part 4 Task 2: UNet

- The four original grayscale label values are mapped to categorical channels.
- Targets have shape `N x 4 x H x W` and are one-hot at every pixel.
- Contracting layers learn context; skip connections recover fine spatial detail in the decoder.
- The output contains four logits per pixel; softmax converts them to categorical probabilities.
- Categorical cross-entropy supervises class probability and Dice loss combats overlap/class-imbalance issues.
- DSC is `2 * intersection / (predicted pixels + target pixels)` and is reported separately for every label.
- Show test-set inference and verify every reported label DSC rather than only the mean.

## Part 4 Task 3: GAN

- The generator maps random latent vectors to synthetic MRI slices.
- The discriminator distinguishes real from generated images.
- Alternating optimisation creates a minimax game; neither loss alone proves image quality.
- Fixed latent vectors make epoch-to-epoch visual comparison meaningful.
- Mode collapse appears when many latent vectors produce nearly identical images.
- Compare `real_samples.png` with `samples_epoch_*.png` for anatomical realism and improvement over training.
- Use `latent_interpolation.png` to show that different latent codes produce smooth, non-identical brain images.
- Use `final_to_real_diversity_ratio` in `metrics.json` as supporting evidence against mode collapse. Ratios near zero indicate collapsed or nearly constant outputs; the images remain the primary evidence.
- In the formal run, the closest generated pair was similar but not identical (`0.002745` mean absolute pixel difference and `0.0638` maximum local difference). This is consistent with partial similarity between brain slices rather than complete sample duplication.
- The late-training diversity remained stable: over epochs 40-80, mean pairwise L1 stayed between `0.0541` and `0.0633`.
- Show samples across epochs, the loss/diversity curves, and explain why the combined visual and numerical evidence supports both realism and diversity.

## Repository and ownership

- Be ready to navigate each model and explain tensor shapes, loss terms, optimiser, and evaluation.
- Keep meaningful commits made after reviewing and understanding each component.
- Cite external and AI assistance accurately according to course policy.
- Do not commit datasets, checkpoints, cluster logs, or generated result directories.
