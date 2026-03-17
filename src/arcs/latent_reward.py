"""Latent-space reward predictor and refinement for VCG.

Trains a lightweight MLP that predicts SPICE simulation reward from
the VCG latent code z and spec embedding.  At inference time, gradient
ascent on z maximizes predicted reward while the VCG constraint
projection maintains structural validity.

Architecture:
    z (latent_dim) ⊕ spec_embed (d_model) → MLP(3 layers) → scalar reward

Usage:
    1. Train:  predictor.fit(dataloader, vcg_model)
    2. Refine: refined_z = predictor.refine(z, spec_embed, ...)
    3. Decode: circuits = vcg_model.decode(refined_z, ...)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class LatentRewardConfig:
    """Configuration for the latent reward predictor."""

    latent_dim: int = 64
    spec_dim: int = 256       # d_model from VCG spec encoder
    hidden_dim: int = 256
    n_layers: int = 3
    dropout: float = 0.1

    # Refinement hyperparameters
    n_refine_steps: int = 15
    refine_lr: float = 0.01
    constraint_weight: float = 10.0  # penalty for constraint violations during refinement
    max_z_drift: float = 3.0         # max L2 distance from initial z
    drift_weight: float = 5.0        # weight for drift penalty (scaled to reward range [0,8])


class LatentRewardPredictor(nn.Module):
    """MLP that predicts SPICE simulation reward from latent code.

    Input:  z (B, latent_dim) + spec_embed (B, spec_dim)
    Output: predicted_reward (B,) — scalar in [0, 8] range
    """

    def __init__(self, config: LatentRewardConfig):
        super().__init__()
        self.config = config

        input_dim = config.latent_dim + config.spec_dim
        layers = []
        in_d = input_dim
        for _ in range(config.n_layers - 1):
            layers.extend([
                nn.Linear(in_d, config.hidden_dim),
                nn.SiLU(),
                nn.Dropout(config.dropout),
            ])
            in_d = config.hidden_dim
        layers.append(nn.Linear(in_d, 1))
        self.net = nn.Sequential(*layers)

    def forward(
        self,
        z: torch.Tensor,
        spec_embed: torch.Tensor,
    ) -> torch.Tensor:
        """Predict reward from latent code and spec embedding.

        Returns:
            (B,) predicted reward scalars
        """
        x = torch.cat([z, spec_embed], dim=-1)
        return self.net(x).squeeze(-1)


class LatentRefinement(nn.Module):
    """Gradient-based latent refinement using reward predictor.

    After VCG decodes z → circuit, refine z via gradient ascent
    on predicted reward while maintaining constraint satisfaction.
    """

    def __init__(
        self,
        reward_predictor: LatentRewardPredictor,
        config: LatentRewardConfig,
    ):
        super().__init__()
        self.reward_predictor = reward_predictor
        self.config = config

    @torch.no_grad()
    def refine(
        self,
        z: torch.Tensor,                # (B, latent_dim)
        spec_embed: torch.Tensor,       # (B, d_model)
        constraints_fn: Optional[callable] = None,  # (z) → violation scalar
    ) -> Tuple[torch.Tensor, dict]:
        """Refine latent code to maximize predicted reward.

        Args:
            z: initial latent codes from VCG prior sampling
            spec_embed: spec conditioning (frozen during refinement)
            constraints_fn: optional callable that takes z and returns
                a scalar constraint violation (used as penalty)

        Returns:
            refined_z: (B, latent_dim) optimized latent codes
            stats: dict with refinement statistics
        """
        z_init = z.clone()
        z_opt = z.clone().detach().requires_grad_(True)

        optimizer = torch.optim.Adam([z_opt], lr=self.config.refine_lr)

        best_reward = float("-inf")
        best_z = z.clone()
        rewards_history = []

        for step in range(self.config.n_refine_steps):
            optimizer.zero_grad()

            with torch.enable_grad():
                # Predict reward
                pred_reward = self.reward_predictor(z_opt, spec_embed)
                objective = pred_reward.mean()

                # Constraint penalty
                if constraints_fn is not None:
                    violation = constraints_fn(z_opt)
                    objective = objective - self.config.constraint_weight * violation

                # Drift penalty — keep z close to initial sample
                # Weighted to scale consistently with reward range [0, 8]
                drift = (z_opt - z_init).norm(dim=-1).mean()
                drift_penalty = self.config.drift_weight * F.relu(drift - self.config.max_z_drift)
                objective = objective - drift_penalty

                # Maximize objective → minimize negative
                loss = -objective

            loss.backward()
            optimizer.step()

            with torch.no_grad():
                current_reward = pred_reward.mean().item()
                rewards_history.append(current_reward)
                if current_reward > best_reward:
                    best_reward = current_reward
                    best_z = z_opt.clone()

        stats = {
            "initial_reward": rewards_history[0] if rewards_history else 0.0,
            "final_reward": best_reward,
            "reward_improvement": best_reward - (rewards_history[0] if rewards_history else 0.0),
            "n_steps": self.config.n_refine_steps,
            "final_drift": (best_z - z_init).norm(dim=-1).mean().item(),
        }

        return best_z.detach(), stats


class LatentRewardTrainer:
    """Trains the reward predictor on (z, spec_embed, reward) triples.

    Training data comes from encoding existing circuit samples through
    the VCG encoder and pairing with their SPICE simulation rewards.
    """

    def __init__(
        self,
        predictor: LatentRewardPredictor,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
    ):
        self.predictor = predictor
        self.optimizer = torch.optim.AdamW(
            predictor.parameters(), lr=lr, weight_decay=weight_decay,
        )
        self.step_count = 0

    def train_step(
        self,
        z: torch.Tensor,           # (B, latent_dim)
        spec_embed: torch.Tensor,  # (B, d_model)
        reward: torch.Tensor,      # (B,) ground-truth SPICE rewards
    ) -> dict:
        """Single training step.

        Returns:
            stats dict with loss and correlation metrics
        """
        self.predictor.train()
        self.optimizer.zero_grad()

        pred = self.predictor(z.detach(), spec_embed.detach())
        loss = F.mse_loss(pred, reward)

        loss.backward()
        self.optimizer.step()
        self.step_count += 1

        with torch.no_grad():
            # Pearson correlation
            pred_centered = pred - pred.mean()
            reward_centered = reward - reward.mean()
            corr = (pred_centered * reward_centered).sum() / (
                pred_centered.norm() * reward_centered.norm() + 1e-8
            )

        return {
            "loss": loss.item(),
            "correlation": corr.item(),
            "pred_mean": pred.mean().item(),
            "reward_mean": reward.mean().item(),
            "step": self.step_count,
        }

    def state_dict(self) -> dict:
        return {
            "predictor": self.predictor.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "step_count": self.step_count,
        }

    def load_state_dict(self, d: dict) -> None:
        self.predictor.load_state_dict(d["predictor"])
        self.optimizer.load_state_dict(d["optimizer"])
        self.step_count = d["step_count"]
