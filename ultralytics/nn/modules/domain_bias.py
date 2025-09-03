# Ultralytics \U0001f680 AGPL-3.0 License - https://ultralytics.com/license
"""Domain-specific bias modules for lightweight domain bias (LDB).

This module implements a BiasToken that adds learnable token and spatial weights
as additive bias to feature maps, and a DomainBiasManager that manages sets of
BiasTokens for different insertion points and domains.
"""

from typing import Dict

import torch
import torch.nn as nn


class BiasToken(nn.Module):
    """Learnable bias token and spatial weight as described in LDB."""

    def __init__(self, channels: int):
        """Initialise BiasToken with channel dimension."""
        super().__init__()
        self.v = nn.Parameter(torch.zeros(channels))
        self.gamma_conv = nn.Conv2d(channels, 1, kernel_size=1)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:  # noqa: D401
        """Apply domain bias to the input feature map."""
        gamma = torch.sigmoid(self.gamma_conv(feat))
        v = self.v.view(1, -1, 1, 1)
        return feat + v * gamma


class DomainBiasManager(nn.Module):
    """Manage BiasTokens for multiple domains and insertion points."""

    def __init__(self, insertion_cfg: Dict[str, int]):
        """Create manager with mapping of insertion names to channel counts."""
        super().__init__()
        self.insertion_cfg = insertion_cfg
        self.registry = nn.ModuleDict()

    def add_domain(self, domain_name: str):
        """Register a new domain and initialise its BiasTokens."""
        if domain_name in self.registry:
            return
        tokens = nn.ModuleDict({k: BiasToken(c) for k, c in self.insertion_cfg.items()})
        self.registry[domain_name] = tokens

    def apply(self, domain_name: str, insert_name: str, feat: torch.Tensor) -> torch.Tensor:
        """Apply the BiasToken for given domain and insertion point."""
        if domain_name not in self.registry:
            raise KeyError(f"[DomainBias] domain {domain_name} not registered")
        if insert_name not in self.registry[domain_name]:
            raise KeyError(f"[DomainBias] insertion {insert_name} not registered (domain={domain_name})")
        return self.registry[domain_name][insert_name](feat)
