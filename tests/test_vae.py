import torch

from comp3710_lab2.part4_vae import vae_loss


def test_vae_loss_is_finite_and_differentiable() -> None:
    logits = torch.randn(2, 1, 8, 8, requires_grad=True)
    reconstruction = torch.sigmoid(logits)
    inputs = torch.rand_like(reconstruction)
    mean = torch.randn(2, 2, requires_grad=True)
    log_variance = torch.randn(2, 2, requires_grad=True)

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        loss, reconstruction_loss, kl_loss = vae_loss(
            reconstruction, inputs, mean, log_variance, beta=1.0
        )

    assert torch.isfinite(loss)
    assert torch.isfinite(reconstruction_loss)
    assert torch.isfinite(kl_loss)
    loss.backward()
    assert logits.grad is not None
    assert mean.grad is not None
    assert log_variance.grad is not None
