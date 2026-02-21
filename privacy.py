import torch
import torch.nn as nn


class PrivacyCode:
    """Simple DP-style gradient perturbation utility."""

    def __init__(self, noise_multiplier: float = 1.0, delta: float = 1e-5, max_grad_norm: float = 1.0):
        self.noise_multiplier = noise_multiplier
        self.delta = delta
        self.max_grad_norm = max_grad_norm

    def apply_privacy(self, model: nn.Module) -> None:
        # Clip model gradients first, then add Gaussian noise.
        torch.nn.utils.clip_grad_norm_(model.parameters(), self.max_grad_norm)
        for param in model.parameters():
            if param.grad is not None:
                noise = torch.randn_like(param.grad) * self.noise_multiplier
                param.grad = param.grad + noise


if __name__ == "__main__":
    model = nn.Linear(10, 2)
    x = torch.randn(4, 10)
    y = torch.randint(0, 2, (4,))
    criterion = nn.CrossEntropyLoss()

    logits = model(x)
    loss = criterion(logits, y)
    loss.backward()

    shield = PrivacyCode(noise_multiplier=1.0, delta=1e-5, max_grad_norm=1.0)
    shield.apply_privacy(model)
    print("Privacy step applied.")
