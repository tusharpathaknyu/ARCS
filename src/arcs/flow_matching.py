"""Constrained Circuit Flow Matching (CCFM).

A novel generative model for electronic circuits that replaces the VAE core
of ValidCircuitGen with Conditional Flow Matching (Lipman et al. 2023),
augmented with differentiable circuit-constraint guidance during sampling.

Key contributions:
    1. **Latent Flow Matching**: Learns a velocity field v_θ(z_t, t, spec)
       that transports Gaussian noise z_0 ~ N(0,I) to valid circuit
       embeddings z_1 along optimal-transport paths.

    2. **Constraint-Guided Sampling**: During ODE integration, the velocity
       is projected towards the feasible set using differentiable circuit
       constraints (Kirchhoff, topology, component bounds), ensuring
       physically valid circuits WITHOUT post-hoc rejection.

    3. **Spec-Conditioned Generation**: Circuit specifications (Vin, Vout,
       efficiency targets) condition the flow via cross-attention, enabling
       spec → circuit inverse design.

Unlike diffusion models, flow matching uses straight probability paths
(simpler training, fewer integration steps) and directly predicts the
velocity v = dz/dt rather than the noise ε.

Architecture:
    ┌────────────┐      ┌───────────────┐      ┌──────────────┐
    │ z_0 ~ N(0,I)│ ──→ │ FlowNetwork   │ ──→ │ z_1 (circuit) │
    └────────────┘      │ v_θ(z_t,t,c)  │      └──────┬───────┘
                        │ + constraint  │             │
    ┌────────────┐      │   guidance    │      ┌──────▼───────┐
    │ Spec cond. │ ──→ └───────────────┘      │ VCGDecoder   │
    │ (c)        │                             │ → circuit    │
    └────────────┘                             └──────────────┘

Training:
    Sample (z_0, z_1, t) where z_1 = encoder(circuit), z_0 ~ N(0,I)
    z_t = (1-t)·z_0 + t·z_1      (linear interpolation)
    u_t = z_1 - z_0               (target velocity)
    Loss = || v_θ(z_t, t, c) - u_t ||²

Sampling (ODE integration with Euler + constraint guidance):
    z_0 ~ N(0,I)
    for t in [0, 1]:
        v = v_θ(z_t, t, c)
        ∇c = constraint_gradient(decode(z_t))    # guidance
        v_guided = v - λ·∇c                       # steer toward valid
        z_{t+dt} = z_t + v_guided · dt

Parameters: ~1.2M (flow network only, reuses VCG encoder/decoder ~4M)
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from arcs.model import RMSNorm
from arcs.valid_circuit_gen import (
    VCGConfig,
    VCGDecoder,
    VCGEncoder,
    SpecEncoder,
    CircuitConstraints,
    ConstraintProjection,
    ValidCircuitGenModel,
    CircuitGraphDataset,
    CircuitGraph,
    graph_to_token_sequence,
    TOPOLOGY_TO_IDX,
    N_TOPOLOGIES,
)
from arcs.tokenizer import CircuitTokenizer


# ═══════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class FlowMatchingConfig:
    """Hyperparameters for Constrained Circuit Flow Matching."""

    # Latent dimension (must match VCG)
    latent_dim: int = 64

    # Flow network architecture
    flow_d_model: int = 256
    flow_n_layers: int = 4
    flow_n_heads: int = 4
    flow_d_ff: int = 512
    flow_dropout: float = 0.1

    # Time embedding
    time_embed_dim: int = 64
    n_time_freqs: int = 32          # sinusoidal frequency count

    # Spec conditioning
    spec_d_model: int = 256         # must match VCG d_model

    # ODE sampling
    n_sample_steps: int = 50        # Euler integration steps
    guidance_strength: float = 1.0  # constraint guidance weight λ
    guidance_start_t: float = 0.3   # start guidance after t=0.3

    # Training
    sigma_min: float = 1e-4         # minimum noise for stability
    ot_plan: bool = True            # use OT-conditional path (vs VP)

    # VCG config (for decoder reuse)
    vcg_config: Optional[VCGConfig] = None

    def to_dict(self) -> dict:
        d = asdict(self)
        if self.vcg_config is not None:
            d["vcg_config"] = self.vcg_config.to_dict()
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "FlowMatchingConfig":
        vcg_dict = d.pop("vcg_config", None)
        cfg = cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
        if vcg_dict is not None:
            cfg.vcg_config = VCGConfig.from_dict(vcg_dict)
        return cfg


# ═══════════════════════════════════════════════════════════════════════════
# Time Embedding
# ═══════════════════════════════════════════════════════════════════════════


class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal embedding for the flow time variable t ∈ [0, 1].

    Produces a fixed-dim vector from scalar t using sin/cos frequencies,
    analogous to positional embeddings in transformers.
    """

    def __init__(self, d_embed: int, n_freqs: int = 32):
        super().__init__()
        self.d_embed = d_embed
        self.n_freqs = n_freqs

        # Learnable projection from raw sinuosids to d_embed
        self.proj = nn.Sequential(
            nn.Linear(2 * n_freqs, d_embed),
            nn.SiLU(),
            nn.Linear(d_embed, d_embed),
        )

        # Frequency bands (fixed, log-spaced)
        freqs = torch.exp(
            torch.linspace(
                math.log(1.0), math.log(1000.0), n_freqs
            )
        )
        self.register_buffer("freqs", freqs)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: (B,) or (B, 1) time values in [0, 1]
        Returns:
            (B, d_embed) embedding
        """
        if t.dim() == 1:
            t = t.unsqueeze(-1)  # (B, 1)

        # (B, n_freqs) each
        angles = t * self.freqs.unsqueeze(0) * 2 * math.pi
        raw = torch.cat([angles.sin(), angles.cos()], dim=-1)  # (B, 2*n_freqs)
        return self.proj(raw)


# ═══════════════════════════════════════════════════════════════════════════
# Flow Transformer Block
# ═══════════════════════════════════════════════════════════════════════════


class AdaptiveLayerNorm(nn.Module):
    """Adaptive Layer Norm conditioned on time + spec.

    Used in DiT-style architectures: the norm parameters (scale, shift)
    are predicted from the conditioning signal.

    y = γ(c) · LayerNorm(x) + β(c)
    """

    def __init__(self, d_model: int, d_cond: int):
        super().__init__()
        self.norm = nn.LayerNorm(d_model, elementwise_affine=False)
        self.adaLN = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_cond, 2 * d_model),
        )
        # Init to identity: γ=1, β=0
        nn.init.zeros_(self.adaLN[-1].weight)
        nn.init.zeros_(self.adaLN[-1].bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, ..., d_model) input
            cond: (B, d_cond) conditioning
        """
        params = self.adaLN(cond)
        while params.dim() < x.dim():
            params = params.unsqueeze(1)
        gamma, beta = params.chunk(2, dim=-1)
        return (1.0 + gamma) * self.norm(x) + beta


class FlowTransformerBlock(nn.Module):
    """Transformer block with adaptive layer norm conditioning.

    Follows the DiT (Diffusion Transformer) design pattern: the time and
    spec conditioning modulate the layer norm parameters rather than being
    concatenated to the input.
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int, d_cond: int,
                 dropout: float = 0.1):
        super().__init__()
        self.adaLN1 = AdaptiveLayerNorm(d_model, d_cond)
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True,
        )
        self.adaLN2 = AdaptiveLayerNorm(d_model, d_cond)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.SiLU(),
            nn.Linear(d_ff, d_model, bias=False),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, L, d_model) — here L=1 for latent token
            cond: (B, d_cond) — time + spec conditioning
        """
        # Self-attention with adaptive norm
        normed = self.adaLN1(x, cond)
        attn_out, _ = self.attn(normed, normed, normed)
        x = x + self.drop(attn_out)

        # FFN with adaptive norm
        x = x + self.drop(self.ffn(self.adaLN2(x, cond)))
        return x


# ═══════════════════════════════════════════════════════════════════════════
# Flow Velocity Network
# ═══════════════════════════════════════════════════════════════════════════


class FlowVelocityNetwork(nn.Module):
    """Predicts the velocity field v_θ(z_t, t, c) for flow matching.

    Architecture:
        z_t (latent_dim) ──→ Linear ──→ d_model tokens
        t   (scalar)     ──→ SinEmbed ──→ time_embed
        c   (spec_embed) ──→ Linear   ──→ spec_proj
        cond = time_embed + spec_proj
        DiT blocks (adaptive LN conditioned on cond)
        Output ──→ Linear ──→ velocity (latent_dim)

    The network operates on a single "latent token" (since z is a vector,
    not a sequence), but uses transformer blocks for their expressiveness
    and conditioning mechanism.
    """

    def __init__(self, config: FlowMatchingConfig):
        super().__init__()
        self.config = config
        d = config.flow_d_model
        d_cond = config.time_embed_dim + config.spec_d_model

        # Input projection: z_t → token
        self.z_proj = nn.Sequential(
            nn.Linear(config.latent_dim, d),
            nn.SiLU(),
            nn.Linear(d, d),
        )

        # Time embedding
        self.time_embed = SinusoidalTimeEmbedding(
            config.time_embed_dim, config.n_time_freqs
        )

        # Spec conditioning projection
        self.spec_proj = nn.Linear(config.spec_d_model, config.spec_d_model)

        # Topology embedding for additional conditioning
        self.topo_embed = nn.Embedding(N_TOPOLOGIES + 1, config.spec_d_model)

        # DiT-style transformer blocks
        self.blocks = nn.ModuleList([
            FlowTransformerBlock(d, config.flow_n_heads, config.flow_d_ff,
                                d_cond, config.flow_dropout)
            for _ in range(config.flow_n_layers)
        ])

        # Output projection: d_model → velocity
        self.out_norm = nn.LayerNorm(d)
        self.out_proj = nn.Sequential(
            nn.Linear(d, d),
            nn.SiLU(),
            nn.Linear(d, config.latent_dim),
        )

        # Initialize output to near-zero (start with identity flow)
        nn.init.zeros_(self.out_proj[-1].weight)
        nn.init.zeros_(self.out_proj[-1].bias)

    def forward(
        self,
        z_t: torch.Tensor,            # (B, latent_dim) noisy latent
        t: torch.Tensor,              # (B,) time in [0, 1]
        spec_embed: torch.Tensor,     # (B, spec_d_model)
        topology_idx: torch.Tensor,   # (B,) int
    ) -> torch.Tensor:
        """Predict velocity v = dz/dt.

        Args:
            z_t: noisy latent at time t
            t: flow time
            spec_embed: spec conditioning vector
            topology_idx: topology index for conditioning

        Returns:
            v: (B, latent_dim) predicted velocity
        """
        # Build conditioning vector
        t_emb = self.time_embed(t)                                    # (B, time_dim)
        s_emb = self.spec_proj(spec_embed) + self.topo_embed(topology_idx)  # (B, spec_dim)
        cond = torch.cat([t_emb, s_emb], dim=-1)                     # (B, d_cond)

        # Project z_t to token sequence (length 1)
        h = self.z_proj(z_t).unsqueeze(1)  # (B, 1, d_model)

        # Transformer blocks
        for block in self.blocks:
            h = block(h, cond)

        # Output velocity
        h = self.out_norm(h.squeeze(1))  # (B, d_model)
        v = self.out_proj(h)             # (B, latent_dim)
        return v

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ═══════════════════════════════════════════════════════════════════════════
# Constraint Guidance
# ═══════════════════════════════════════════════════════════════════════════


class ConstraintGuidance(nn.Module):
    """Differentiable constraint guidance for flow sampling.

    During ODE integration, computes the gradient of circuit violations
    w.r.t. the latent z, and steers the flow away from invalid regions.

    v_guided = v_θ(z_t, t, c) - λ · ∇_z Σ_i w_i · c_i(decode(z_t))

    where c_i are circuit constraints (no floating nodes, connectivity, etc.)
    and w_i are per-constraint weights.
    """

    def __init__(
        self,
        decoder: VCGDecoder,
        constraints: CircuitConstraints,
        config: FlowMatchingConfig,
    ):
        super().__init__()
        self.decoder = decoder
        self.constraints = constraints
        self.config = config

        # Per-constraint weights (learnable)
        self.log_weights = nn.Parameter(torch.zeros(5))

    @property
    def weights(self) -> torch.Tensor:
        return F.softplus(self.log_weights)

    def compute_guidance(
        self,
        z: torch.Tensor,              # (B, latent_dim)
        spec_embed: torch.Tensor,     # (B, d_model)
        topology_idx: torch.Tensor,   # (B,) int
        active_mask: torch.Tensor,    # (B, N)
        bounds_min: torch.Tensor,     # (B, N)
        bounds_max: torch.Tensor,     # (B, N)
    ) -> Tuple[torch.Tensor, float]:
        """Compute constraint gradient w.r.t. z.

        Returns:
            grad_z: (B, latent_dim) gradient direction
            violation: scalar total violation
        """
        z_grad = z.detach().requires_grad_(True)

        # Decode z → soft graph
        soft_X, soft_A, soft_V = self.decoder(z_grad, spec_embed, topology_idx)

        # Compute constraint violations
        violations = self.constraints.all_constraints(
            soft_A, soft_X, soft_V, active_mask, bounds_min, bounds_max,
        )  # (B, 5)

        # Weighted violation
        total_violation = (self.weights * violations.mean(dim=0)).sum()

        # Gradient of violation w.r.t. z
        grad_z = torch.autograd.grad(
            total_violation, z_grad,
            create_graph=False,
            retain_graph=False,
        )[0]

        return grad_z, total_violation.item()


# ═══════════════════════════════════════════════════════════════════════════
# Full CCFM Model
# ═══════════════════════════════════════════════════════════════════════════


class ConstrainedFlowMatchingModel(nn.Module):
    """Constrained Circuit Flow Matching (CCFM).

    Combines:
        - VCG encoder/decoder (frozen or fine-tuned)
        - Flow velocity network (trained)
        - Constraint guidance (trained guidance weights)

    Training:
        1. Encode training circuits → z_1 (via VCG encoder)
        2. Sample z_0 ~ N(0,I), t ~ U(0,1)
        3. z_t = (1-t)·z_0 + t·z_1
        4. Loss = || v_θ(z_t, t, c) - (z_1 - z_0) ||²

    Sampling:
        1. z_0 ~ N(0,I)
        2. Euler integration with constraint guidance
        3. Decode z_1 → circuit graph via VCG decoder + projection
    """

    def __init__(
        self,
        flow_config: FlowMatchingConfig,
        vcg_model: Optional[ValidCircuitGenModel] = None,
    ):
        super().__init__()
        self.flow_config = flow_config

        # VCG components (reuse or create fresh)
        if vcg_model is not None:
            self.spec_encoder = vcg_model.spec_encoder
            self.vcg_encoder = vcg_model.encoder
            self.vcg_decoder = vcg_model.decoder
            vcg_config = vcg_model.config
        else:
            vcg_config = flow_config.vcg_config or VCGConfig(
                latent_dim=flow_config.latent_dim,
                d_model=flow_config.spec_d_model,
            )
            self.spec_encoder = SpecEncoder(vcg_config)
            self.vcg_encoder = VCGEncoder(vcg_config)
            self.vcg_decoder = VCGDecoder(vcg_config)

        self.flow_config.vcg_config = vcg_config

        # Constraints
        self.constraints = CircuitConstraints(vcg_config)
        self.projection = ConstraintProjection(vcg_config)

        # Flow velocity network
        self.flow_net = FlowVelocityNetwork(flow_config)

        # Constraint guidance
        self.guidance = ConstraintGuidance(
            self.vcg_decoder, self.constraints, flow_config
        )

    def encode_to_latent(
        self, batch: dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode circuit graphs to latent vectors (z_1 for training).

        Returns:
            z_1: (B, latent_dim) — latent circuit embedding (mu, deterministic)
            spec_embed: (B, d_model) — spec conditioning vector
        """
        spec_embed = self.spec_encoder(
            batch["spec_types"], batch["spec_values"], batch["spec_mask"],
        )
        mu, logvar = self.vcg_encoder(
            batch["node_types"], batch["values"], batch["adjacency"],
            batch["active_mask"], batch["topology_idx"], spec_embed,
        )
        # Use mu directly (no sampling) for FM training targets
        return mu, spec_embed

    def compute_flow_loss(
        self, batch: dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, dict]:
        """Compute conditional flow matching loss.

        L_CFM = E_{t,z_0,z_1} || v_θ(z_t, t, c) - u_t ||²

        where z_t = (1-t)·z_0 + t·z_1 and u_t = z_1 - z_0.

        Returns:
            loss: scalar flow matching loss
            stats: dict with per-component losses
        """
        B = batch["node_types"].shape[0]
        device = batch["node_types"].device

        # Encode training data → z_1
        with torch.no_grad():
            z_1, spec_embed = self.encode_to_latent(batch)
            spec_embed = spec_embed.detach()
            z_1 = z_1.detach()

        # Sample noise z_0 ~ N(0, I)
        z_0 = torch.randn_like(z_1)

        # Sample time t ~ U(0, 1)
        t = torch.rand(B, device=device)

        # Interpolate: z_t = (1-t)·z_0 + t·z_1
        t_expand = t.unsqueeze(-1)  # (B, 1)
        z_t = (1.0 - t_expand) * z_0 + t_expand * z_1

        # Add small noise for stability (σ_min)
        z_t = z_t + self.flow_config.sigma_min * torch.randn_like(z_t)

        # Target velocity: u_t = z_1 - z_0
        u_t = z_1 - z_0

        # Predict velocity
        v_pred = self.flow_net(
            z_t, t, spec_embed, batch["topology_idx"],
        )

        # MSE loss
        flow_loss = F.mse_loss(v_pred, u_t)

        # Optional: consistency regularization — predicted z_1 should be
        # close to actual z_1 (helps early training convergence)
        z_1_pred = z_t + (1.0 - t_expand) * v_pred  # predicted endpoint
        consistency_loss = F.mse_loss(z_1_pred, z_1) * 0.1

        total_loss = flow_loss + consistency_loss

        stats = {
            "loss/flow": flow_loss.item(),
            "loss/consistency": consistency_loss.item(),
            "loss/total": total_loss.item(),
            "v_pred_norm": v_pred.norm(dim=-1).mean().item(),
            "u_t_norm": u_t.norm(dim=-1).mean().item(),
        }

        return total_loss, stats

    @torch.no_grad()
    def sample(
        self,
        spec_types: torch.Tensor,     # (B, S) int
        spec_values: torch.Tensor,    # (B, S) float
        spec_mask: torch.Tensor,      # (B, S) bool/float
        topology_idx: torch.Tensor,   # (B,) int
        active_mask: torch.Tensor,    # (B, N) float
        bounds_min: torch.Tensor,     # (B, N)
        bounds_max: torch.Tensor,     # (B, N)
        n_steps: Optional[int] = None,
        guidance_strength: Optional[float] = None,
        temperature: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """Generate circuits via ODE integration with constraint guidance.

        Args:
            spec_*:  specification conditioning
            topology_idx: target topology
            active_mask, bounds_*: for constraint guidance
            n_steps: ODE integration steps (default: config)
            guidance_strength: constraint guidance λ (default: config)
            temperature: noise scaling (1.0 = standard)

        Returns:
            soft_X: (B, N, n_types) node type probabilities
            soft_A: (B, N, N) adjacency probabilities
            soft_V: (B, N) component values (log10)
            info:   dict with sampling statistics
        """
        n_steps = n_steps or self.flow_config.n_sample_steps
        λ = guidance_strength if guidance_strength is not None else self.flow_config.guidance_strength

        B = spec_types.shape[0]
        device = spec_types.device

        # Compute spec embedding
        spec_embed = self.spec_encoder(spec_types, spec_values, spec_mask)

        # Start from noise
        z = torch.randn(B, self.flow_config.latent_dim, device=device) * temperature

        dt = 1.0 / n_steps
        total_violation = 0.0
        guidance_applied = 0

        # Euler integration: z_0 → z_1
        for step in range(n_steps):
            t_val = step / n_steps
            t = torch.full((B,), t_val, device=device)

            # Predict velocity
            v = self.flow_net(z, t, spec_embed, topology_idx)

            # Apply constraint guidance after guidance_start_t
            if λ > 0 and t_val >= self.flow_config.guidance_start_t:
                with torch.enable_grad():
                    grad_z, viol = self.guidance.compute_guidance(
                        z, spec_embed, topology_idx,
                        active_mask, bounds_min, bounds_max,
                    )
                # Steer away from constraint violations
                v = v - λ * grad_z
                total_violation += viol
                guidance_applied += 1

            # Euler step
            z = z + v * dt

        # Decode final latent → soft graph
        soft_X, soft_A, soft_V = self.vcg_decoder(z, spec_embed, topology_idx)

        info = {
            "mean_violation": total_violation / max(guidance_applied, 1),
            "guidance_steps": guidance_applied,
            "z_norm": z.norm(dim=-1).mean().item(),
        }

        return soft_X, soft_A, soft_V, info

    @torch.no_grad()
    def sample_with_projection(
        self,
        spec_types: torch.Tensor,
        spec_values: torch.Tensor,
        spec_mask: torch.Tensor,
        topology_idx: torch.Tensor,
        active_mask: torch.Tensor,
        bounds_min: torch.Tensor,
        bounds_max: torch.Tensor,
        n_steps: Optional[int] = None,
        guidance_strength: Optional[float] = None,
        temperature: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """Sample + apply constraint projection for guaranteed validity.

        Returns projected (discretized) soft graph.
        """
        soft_X, soft_A, soft_V, info = self.sample(
            spec_types, spec_values, spec_mask, topology_idx,
            active_mask, bounds_min, bounds_max,
            n_steps, guidance_strength, temperature,
        )

        # Project onto constraint set
        soft_X, soft_A, soft_V, proj_stats = self.projection.project(
            soft_X, soft_A, soft_V, active_mask, bounds_min, bounds_max,
        )
        info.update({f"proj_{k}": v for k, v in proj_stats.items()})

        return soft_X, soft_A, soft_V, info

    def count_parameters(self) -> int:
        """Total parameters (flow net + guidance)."""
        return sum(p.numel() for p in self.parameters())

    def count_trainable_parameters(self) -> int:
        """Trainable parameters only."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def freeze_vcg(self) -> None:
        """Freeze VCG encoder/decoder, only train flow network + guidance."""
        for p in self.spec_encoder.parameters():
            p.requires_grad = False
        for p in self.vcg_encoder.parameters():
            p.requires_grad = False
        for p in self.vcg_decoder.parameters():
            p.requires_grad = False

    def unfreeze_vcg(self) -> None:
        """Unfreeze VCG components for end-to-end fine-tuning."""
        for p in self.spec_encoder.parameters():
            p.requires_grad = True
        for p in self.vcg_encoder.parameters():
            p.requires_grad = True
        for p in self.vcg_decoder.parameters():
            p.requires_grad = True

    def save(self, path: str | Path) -> None:
        """Save full model state."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "flow_config": self.flow_config.to_dict(),
                "state_dict": self.state_dict(),
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path, device: torch.device = torch.device("cpu")) -> "ConstrainedFlowMatchingModel":
        """Load model from checkpoint."""
        ckpt = torch.load(path, map_location=device, weights_only=False)
        config = FlowMatchingConfig.from_dict(ckpt["flow_config"])
        model = cls(config)
        model.load_state_dict(ckpt["state_dict"])
        model.to(device)
        return model


# ═══════════════════════════════════════════════════════════════════════════
# Training Utilities
# ═══════════════════════════════════════════════════════════════════════════


def train_flow_matching(
    model: ConstrainedFlowMatchingModel,
    train_dataset: CircuitGraphDataset,
    val_dataset: Optional[CircuitGraphDataset] = None,
    n_epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-4,
    weight_decay: float = 0.01,
    freeze_vcg: bool = True,
    device: torch.device = torch.device("cpu"),
    output_dir: str | Path = "checkpoints/ccfm",
    log_interval: int = 50,
) -> dict:
    """Train the CCFM model.

    Args:
        model: CCFM model
        train_dataset: training circuit graphs
        val_dataset: optional validation circuit graphs
        n_epochs: training epochs
        batch_size: batch size
        lr: learning rate
        weight_decay: AdamW weight decay
        freeze_vcg: if True, only train flow net + guidance weights
        device: torch device
        output_dir: checkpoint output directory
        log_interval: log every N steps

    Returns:
        dict with training history
    """
    import logging
    import time
    from torch.utils.data import DataLoader

    logger = logging.getLogger(__name__)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = model.to(device)
    if freeze_vcg:
        model.freeze_vcg()
        logger.info("VCG encoder/decoder frozen — training flow network only")
    else:
        logger.info("End-to-end training (VCG + flow network)")

    n_trainable = model.count_trainable_parameters()
    n_total = model.count_parameters()
    logger.info(f"Parameters: {n_trainable:,} trainable / {n_total:,} total")

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=weight_decay,
        betas=(0.9, 0.95),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_epochs, eta_min=lr / 10
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=True,
    )

    history: dict[str, list] = {
        "train_loss": [], "val_loss": [],
        "flow_loss": [], "consistency_loss": [],
    }
    best_val_loss = float("inf")
    global_step = 0

    for epoch in range(1, n_epochs + 1):
        model.train()
        epoch_losses: list[float] = []
        t0 = time.time()

        for batch_dict in train_loader:
            # Move batch to device
            batch_dev = {
                k: v.to(device) for k, v in batch_dict.items()
            }

            optimizer.zero_grad()
            loss, stats = model.compute_flow_loss(batch_dev)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                filter(lambda p: p.requires_grad, model.parameters()),
                max_norm=1.0,
            )
            optimizer.step()

            epoch_losses.append(loss.item())
            global_step += 1

            if global_step % log_interval == 0:
                logger.info(
                    f"  step {global_step:5d} | "
                    f"flow={stats['loss/flow']:.4f} | "
                    f"consist={stats['loss/consistency']:.4f} | "
                    f"v_norm={stats['v_pred_norm']:.3f}"
                )

        epoch_loss = np.mean(epoch_losses)
        history["train_loss"].append(epoch_loss)
        dt = time.time() - t0

        # Validation
        val_loss = float("inf")
        if val_dataset is not None and len(val_dataset) > 0:
            model.eval()
            val_loader = DataLoader(
                val_dataset, batch_size=batch_size, shuffle=False,
            )
            val_losses: list[float] = []
            with torch.no_grad():
                for batch_dict in val_loader:
                    batch_dev = {k: v.to(device) for k, v in batch_dict.items()}
                    loss, _ = model.compute_flow_loss(batch_dev)
                    val_losses.append(loss.item())
            val_loss = np.mean(val_losses) if val_losses else float("inf")
        history["val_loss"].append(val_loss)

        logger.info(
            f"Epoch {epoch:3d}/{n_epochs} | "
            f"train_loss={epoch_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"lr={scheduler.get_last_lr()[0]:.2e} | "
            f"{dt:.1f}s"
        )

        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model.save(output_dir / "best_ccfm.pt")
            logger.info(f"  → New best val_loss: {best_val_loss:.4f}")

        scheduler.step()

        # Periodic checkpoint
        if epoch % 20 == 0:
            model.save(output_dir / f"ccfm_epoch{epoch}.pt")

    # Final save
    model.save(output_dir / "final_ccfm.pt")

    # Save history
    with open(output_dir / "ccfm_history.json", "w") as f:
        json.dump(history, f, indent=2)

    return {
        "n_epochs": n_epochs,
        "best_val_loss": best_val_loss,
        "final_train_loss": history["train_loss"][-1],
        "history": history,
    }
