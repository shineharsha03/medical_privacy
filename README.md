# Medical Privacy

Minimal PyTorch utilities for privacy-aware training workflows in medical AI projects.

## What This Repo Includes
- `privacy.py`: a lightweight gradient privacy helper.
- Gradient clipping before gradient perturbation.
- Gaussian noise injection on parameter gradients.

## Core Class
`PrivacyCode` supports:
- `noise_multiplier`: controls added Gaussian noise scale.
- `delta`: placeholder privacy parameter for future accounting integration.
- `max_grad_norm`: gradient clipping threshold.

## Quick Start
```python
import torch
import torch.nn as nn
from privacy import PrivacyCode

model = nn.Linear(10, 2)
x = torch.randn(4, 10)
y = torch.randint(0, 2, (4,))
criterion = nn.CrossEntropyLoss()

logits = model(x)
loss = criterion(logits, y)
loss.backward()

shield = PrivacyCode(noise_multiplier=1.0, delta=1e-5, max_grad_norm=1.0)
shield.apply_privacy(model)
```

## Notes
- This is a simple educational privacy utility, not a full formal DP pipeline.
- For production differential privacy, integrate a privacy accountant (for example with Opacus).