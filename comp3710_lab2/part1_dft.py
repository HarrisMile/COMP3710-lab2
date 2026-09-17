from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from .common import make_output_dir, save_json, select_device, seed_everything


def square_wave(
    t: np.ndarray | torch.Tensor,
    frequency: float,
) -> np.ndarray | torch.Tensor:
    if isinstance(t, torch.Tensor):
        return torch.sign(torch.sin(2 * torch.pi * frequency * t))
    return np.sign(np.sin(2 * np.pi * frequency * t))


def square_wave_fourier(
    t: np.ndarray | torch.Tensor,
    frequency: float,
    harmonics: int,
) -> np.ndarray | torch.Tensor:
    if isinstance(t, torch.Tensor):
        return square_wave_fourier_torch(t, frequency, harmonics)
    result = np.zeros_like(t, dtype=np.float64)
    for harmonic_index in range(harmonics):
        order = 2 * harmonic_index + 1
        result += np.sin(2 * np.pi * order * frequency * t) / order
    return (4 / np.pi) * result


def naive_dft_numpy(signal: np.ndarray) -> np.ndarray:
    signal = np.asarray(signal)
    length = signal.shape[0]
    result = np.zeros(length, dtype=np.complex128)
    for frequency_index in range(length):
        for sample_index in range(length):
            angle = -2j * np.pi * frequency_index * sample_index / length
            result[frequency_index] += signal[sample_index] * np.exp(angle)
    return result


def square_wave_fourier_torch(
    t: torch.Tensor,
    frequency: float,
    harmonics: int,
) -> torch.Tensor:
    orders = torch.arange(1, 2 * harmonics, 2, device=t.device, dtype=t.dtype)
    phases = 2 * torch.pi * frequency * orders[:, None] * t[None, :]
    return (4 / torch.pi) * (torch.sin(phases) / orders[:, None]).sum(dim=0)


def tensor_dft(signal: torch.Tensor) -> torch.Tensor:
    """Explicit O(N^2) DFT using tensor matrix operations, not torch.fft."""
    signal = signal.flatten()
    length = signal.numel()
    indices = torch.arange(length, device=signal.device, dtype=signal.dtype)
    angles = -2 * torch.pi * torch.outer(indices, indices) / length
    real = torch.cos(angles) @ signal
    imaginary = torch.sin(angles) @ signal
    return torch.complex(real, imaginary)


def _synchronise(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def _measure_seconds(operation, repeats: int, device: torch.device | None = None):
    samples = []
    result = None
    for _ in range(repeats):
        if device is not None:
            _synchronise(device)
        start = time.perf_counter()
        result = operation()
        if device is not None:
            _synchronise(device)
        samples.append(time.perf_counter() - start)
    return result, float(np.median(samples))


def benchmark(
    size: int,
    device: torch.device,
    repeats: int,
) -> tuple[dict[str, float], float]:
    time_axis = np.linspace(0, 1, size, endpoint=False, dtype=np.float64)
    signal = square_wave_fourier(time_axis, frequency=5, harmonics=50)

    naive_result, naive_seconds = _measure_seconds(
        lambda: naive_dft_numpy(signal),
        repeats,
    )
    fft_result, fft_seconds = _measure_seconds(lambda: np.fft.fft(signal), repeats)

    tensor_signal = torch.tensor(signal, dtype=torch.float64, device=device)
    _ = tensor_dft(tensor_signal)
    tensor_result, tensor_seconds = _measure_seconds(
        lambda: tensor_dft(tensor_signal),
        repeats,
        device,
    )

    maximum_error = float(
        np.max(np.abs(tensor_result.detach().cpu().numpy() - fft_result))
    )
    if not np.allclose(naive_result, fft_result, rtol=1e-7, atol=1e-7):
        raise RuntimeError("Naive DFT does not match NumPy FFT")
    if not np.allclose(
        tensor_result.detach().cpu().numpy(), fft_result, rtol=1e-6, atol=1e-6
    ):
        raise RuntimeError("Tensor DFT does not match NumPy FFT")
    return {
        "size": size,
        "naive_numpy_seconds": naive_seconds,
        "numpy_fft_seconds": fft_seconds,
        "tensor_dft_seconds": tensor_seconds,
    }, maximum_error


def plot_frequency_spectrum(
    size: int,
    device: torch.device,
    output_dir: Path,
) -> None:
    fundamental = 1.0
    time_axis = torch.arange(size, device=device, dtype=torch.float64) / size
    signal = square_wave_fourier_torch(
        time_axis,
        frequency=fundamental,
        harmonics=50,
    )
    transformed = tensor_dft(signal)
    frequencies = np.fft.fftfreq(size, d=1.0 / size)[: size // 2]
    magnitude = (2.0 / size) * transformed.abs()[: size // 2].detach().cpu().numpy()

    figure, axes = plt.subplots(2, 1, figsize=(11, 8))
    axes[0].plot(time_axis.detach().cpu().numpy(), signal.detach().cpu().numpy())
    axes[0].set_title("50-harmonic square-wave reconstruction")
    axes[0].set_xlabel("Time (seconds)")
    axes[0].set_ylabel("Amplitude")
    axes[0].set_xlim(0, 1)
    axes[0].grid(alpha=0.3)

    axes[1].stem(frequencies, magnitude, basefmt=" ")
    for order in range(1, 20, 2):
        axes[1].axvline(
            order * fundamental,
            color="tab:red",
            linestyle="--",
            alpha=0.35,
        )
    axes[1].set_title("Explicit tensor DFT magnitude spectrum")
    axes[1].set_xlabel("Frequency (Hz)")
    axes[1].set_ylabel("Magnitude")
    axes[1].set_xlim(0, 50)
    axes[1].grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(output_dir / "dft_frequency_spectrum.png", dpi=160)
    plt.close(figure)


def plot_reconstructions(output_dir: Path) -> None:
    t = np.linspace(0, 1, 2000, endpoint=False)
    original = square_wave(t, frequency=5)
    harmonics = [1, 3, 5, 20, 50]
    figure, axes = plt.subplots(3, 2, figsize=(12, 10), sharex=True)
    axes = axes.flatten()
    axes[0].plot(t, original, color="black", linewidth=1)
    axes[0].set_title("Original square wave")
    for axis, count in zip(axes[1:], harmonics):
        axis.plot(t, original, color="black", alpha=0.25, label="Target")
        axis.plot(t, square_wave_fourier(t, 5, count), label=f"{count} harmonics")
        axis.set_title(f"Fourier reconstruction: {count} harmonics")
        axis.legend()
    for axis in axes:
        axis.set_xlim(0, 0.4)
        axis.set_ylim(-1.5, 1.5)
        axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_dir / "square_wave_reconstructions.png", dpi=160)
    plt.close(figure)


def plot_timings(rows: list[dict[str, float]], output_dir: Path) -> None:
    figure, axis = plt.subplots(figsize=(8, 5))
    sizes = [int(row["size"]) for row in rows]
    for key, label in (
        ("naive_numpy_seconds", "Naive NumPy DFT"),
        ("numpy_fft_seconds", "NumPy FFT"),
        ("tensor_dft_seconds", "Tensor DFT"),
    ):
        axis.plot(sizes, [row[key] for row in rows], marker="o", label=label)
    axis.set_xlabel("Signal length N")
    axis.set_ylabel("Execution time (seconds, log scale)")
    axis.set_yscale("log")
    axis.grid(alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "dft_timing.png", dpi=160)
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Part 1: DFT comparison")
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=[64, 128, 256, 512, 1024, 2048],
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output-dir", default="results/part1_dft")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    output_dir = make_output_dir(args.output_dir)
    device = select_device(args.device)
    rows = []
    maximum_errors = []
    for size in args.sizes:
        row, error = benchmark(size, device, args.repeats)
        rows.append(row)
        maximum_errors.append(error)
        print(row)
    timing_order = {
        str(int(row["size"])): sorted(
            (
                ("NumPy FFT", row["numpy_fft_seconds"]),
                ("GPU/tensor explicit DFT", row["tensor_dft_seconds"]),
                ("naive NumPy DFT", row["naive_numpy_seconds"]),
            ),
            key=lambda item: item[1],
        )
        for row in rows
    }
    with (output_dir / "timings.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    plot_reconstructions(output_dir)
    plot_timings(rows, output_dir)
    plot_frequency_spectrum(max(args.sizes), device, output_dir)
    save_json(
        {
            "device": str(device),
            "timing_repeats": args.repeats,
            "maximum_tensor_fft_error": max(maximum_errors),
            "timings": rows,
            "fastest_to_slowest": timing_order,
            "interpretation": (
                "NumPy FFT uses an O(N log N) algorithm. Both explicit DFT methods "
                "perform O(N^2) work; tensor operations can parallelise that work on "
                "the GPU, while the nested Python loops run serially and are normally "
                "the slowest as N grows. Small inputs may be dominated by launch overhead."
            ),
        },
        output_dir / "metrics.json",
    )


if __name__ == "__main__":
    main()
