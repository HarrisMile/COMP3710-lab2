import numpy as np
import torch

from comp3710_lab2.part1_dft import (
    square_wave,
    naive_dft_numpy,
    square_wave_fourier,
    square_wave_fourier_torch,
    tensor_dft,
)


def test_dft_implementations_match_fft() -> None:
    rng = np.random.default_rng(42)
    signal = rng.normal(size=16)
    expected = np.fft.fft(signal)
    assert np.allclose(naive_dft_numpy(signal), expected)
    actual = tensor_dft(torch.tensor(signal, dtype=torch.float64)).numpy()
    assert np.allclose(actual, expected)


def test_numpy_and_torch_square_wave_reconstruction_match() -> None:
    t = np.linspace(0, 1, 64, endpoint=False)
    numpy_result = square_wave_fourier(t, frequency=5, harmonics=10)
    torch_result = square_wave_fourier_torch(
        torch.tensor(t, dtype=torch.float64), frequency=5, harmonics=10
    )
    assert np.allclose(torch_result.numpy(), numpy_result)


def test_square_wave_dispatches_to_numpy_and_torch() -> None:
    t = np.linspace(0, 1, 64, endpoint=False)
    numpy_result = square_wave(t, frequency=5)
    torch_result = square_wave(torch.tensor(t), frequency=5)
    assert isinstance(numpy_result, np.ndarray)
    assert isinstance(torch_result, torch.Tensor)
    assert np.allclose(torch_result.numpy(), numpy_result)
