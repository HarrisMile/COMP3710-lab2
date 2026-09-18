# AI Assistance and Verification Report

## 1. Purpose of AI Use

I used OpenAI ChatGPT and Codex as supporting tools during COMP3710 Assignment 2. Their main roles were to help me interpret technical requirements, discuss implementation approaches, identify possible bugs, improve testing, prepare Slurm commands, and organise documentation.

AI was used as a programming assistant and technical tutor rather than as a replacement for running, validating, and understanding the assignment. I remained responsible for deciding whether to accept, modify, or reject its suggestions.

## 2. Areas Where AI Assisted

AI assistance was used in the following areas:

- Discussing implementations of the DFT, PCA/eigenfaces, CNN, ResNet, VAE, U-Net, and GAN tasks.
- Drafting or refining parts of the Python implementation, test cases, inference modes, and Slurm scripts.
- Explaining concepts such as FFT complexity, PCA projection, convolutional kernels, residual connections, VAE reparameterisation, Dice loss, skip connections, GAN alternating optimisation, and mode collapse.
- Diagnosing cluster issues such as missing paths, queued jobs, time limits, and incomplete output files.
- Suggesting performance improvements for the CIFAR-10 speed task, including mixed precision, GPU-side data handling, EMA weights, and test-time augmentation.
- Improving result checking, checkpoint-loading workflows, README content, demonstration commands, and viva preparation material.

Some AI suggestions required correction or further testing. I did not assume that generated code was correct simply because it ran.

## 3. Work Performed and Verified by Me

I personally executed the project on both my local machine and the Rangpur cluster. This included preparing the environment, submitting Slurm jobs, monitoring job states, inspecting standard-output and standard-error logs, downloading results, and checking the generated figures and metrics.

I reviewed the implementation at the level of model structure, preprocessing, tensor shapes, loss functions, optimisation, evaluation, and checkpoint loading. I also tested whether saved models could be loaded again for inference instead of relying only on screenshots or previously printed metrics.

The final project passed 14 automated tests. These tests cover numerical DFT correctness, model output shapes, OASIS label conversion, one-hot masks, differentiable losses, VAE loss behaviour, and GAN sample-diversity calculations.

I distinguished small smoke tests from formal experiments. Smoke tests were used only to verify that a pipeline could execute; all reported assessment results came from completed formal runs.

## 4. Independent Experimental Evidence

The results reported in the repository were produced by actual program executions rather than invented or manually entered values. Examples include:

- The explicit tensor DFT agreed with the FFT to a maximum error of approximately `3.29e-10`.
- The eigenfaces experiment used 150 principal components, explaining approximately `94.65%` of the training variance.
- The LFW CNN achieved `71.74%` test accuracy, compared with `65.22%` for PCA plus Random Forest.
- The CIFAR-10 ResNet reached `94.12%` accuracy at epoch 35 in approximately `121.29` seconds of wall time, meeting the 360-second target.
- The formal VAE used 256x256 OASIS images, `beta=1.0`, and a two-dimensional latent space.
- The U-Net achieved a test mean DSC of approximately `0.9775`, with all three foreground-label DSC values above `0.96`.
- The GAN completed 80 epochs and produced generated samples, latent interpolations, loss curves, and numerical diversity evidence.

I retained the corresponding metrics, figures, checkpoints, and cluster logs. The public repository contains a curated evidence directory, while large model checkpoints remain available locally and on Rangpur for live inference.

## 5. Validation and Critical Review

I checked AI-assisted work using several independent forms of evidence:

1. Automated unit tests.
2. Syntax checking of Python and Slurm scripts.
3. Full training runs on Rangpur GPUs.
4. Numerical comparisons with reference implementations.
5. Inspection of training and validation curves.
6. Checkpoint reloading and inference.
7. Confusion matrices and per-class metrics.
8. Visual inspection of reconstructions, segmentations, generated samples, and latent interpolations.
9. Slurm job records and `.out/.err` logs.

When results were unsuccessful, I treated them as debugging evidence rather than final results. For example, the original CIFAR run exceeded the cluster time limit, so it was not presented as satisfying the speed target. The final claim was made only after the faster implementation completed and achieved the required accuracy within the target time.

## 6. Limitations and Responsibility

AI-generated suggestions can contain bugs, unsuitable assumptions, or implementations that do not satisfy assignment-specific requirements. For that reason, I verified the final behaviour against the assignment requirements and retained evidence from real executions.

I understand the main components of each model and can explain the preprocessing, architecture, loss, training process, evaluation metrics, results, and limitations. I can also make and explain simple modifications to the code.

I take responsibility for the final submitted code, experimental configuration, reported results, repository content, and demonstration. AI assistance is disclosed because it contributed to development and documentation, but the final results were obtained and verified through my own execution and review of the project.
