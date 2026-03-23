"""ARCS Phase 3: SPICE-in-the-Loop Reinforcement Learning.

Fine-tunes the pre-trained ARCS model using REINFORCE with baseline,
where the reward signal comes from actual ngspice simulation of
generated circuits.

Pipeline:
    1. Model generates circuit token sequence (with log-probs)
    2. Decode tokens → DecodedCircuit → parameter dict
    3. Build SPICE netlist from parameters → simulate with ngspice
    4. Extract metrics (efficiency, vout_error, ripple) → scalar reward
    5. REINFORCE update: advantage * log_prob + KL penalty

Usage:
    PYTHONPATH=src python -m arcs.rl \\
        --checkpoint checkpoints/arcs_small/best_model.pt \\
        --output checkpoints/arcs_rl \\
        --steps 5000 --batch-size 8
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import math
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from arcs.datagen import compute_derived_metrics, is_valid_result
from arcs.evaluate import DecodedCircuit, decode_generated_sequence
from arcs.model import ARCSConfig, ARCSModel
from arcs.model_enhanced import create_model, load_model
from arcs.simulate import (
    COMPONENT_TO_PARAM,
    SimulationOutcome,
    components_to_params,
    compute_reward,
    simulate_decoded_circuit,
    _get_spec_to_cond,
)
from arcs.spice import NGSpiceRunner
from arcs.templates import (
    OPERATING_CONDITIONS,
    POWER_CONVERTER_BOUNDS,
    get_topology,
)
from arcs.tokenizer import CircuitTokenizer, TokenType
from arcs import DEFAULT_TEMPERATURE, DEFAULT_TOP_K

logger = logging.getLogger(__name__)


# simulate_decoded_circuit, SimulationOutcome, compute_reward imported from arcs.simulate


# compute_reward imported from arcs.simulate (domain-aware: power/signal/mirror/regulator)


# ---------------------------------------------------------------------------
# Sampling with log-probabilities (for REINFORCE)
# ---------------------------------------------------------------------------


@torch.no_grad()
def sample_with_logprobs(
    model: ARCSModel,
    prefix: torch.Tensor,
    tokenizer: CircuitTokenizer,
    max_new_tokens: int = 64,
    temperature: float = 0.8,
    top_k: int = 50,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate tokens autoregressively and collect per-token log-probs.

    Args:
        model: ARCS model
        prefix: (1, T_prefix) prefix token IDs
        tokenizer: CircuitTokenizer
        max_new_tokens: Max tokens to generate
        temperature: Sampling temperature
        top_k: Top-k filtering (0 = no filtering)

    Returns:
        generated: (T_gen,) generated token IDs (excluding prefix)
        log_probs: (T_gen,) log probability of each sampled token
        entropy:   (T_gen,) entropy of the distribution at each step
    """
    model.eval()
    device = prefix.device
    sequence = prefix.clone()

    gen_tokens = []
    gen_logprobs = []
    gen_entropy = []

    end_id = tokenizer.name_to_id.get("END", 2)

    for _ in range(max_new_tokens):
        # Forward pass (no grad by decorator)
        # For GraphTransformerARCSModel, pass tokenizer for graph features
        if hasattr(model, 'compute_graph_features'):
            logits, _ = model(sequence, tokenizer=tokenizer)
        else:
            logits, _ = model(sequence)  # (1, T, vocab)
        next_logits = logits[0, -1, :] / temperature  # (vocab,)

        # Top-k filtering
        if top_k > 0:
            values, _ = torch.topk(next_logits, min(top_k, next_logits.size(0)))
            threshold = values[-1]
            next_logits[next_logits < threshold] = float("-inf")

        probs = F.softmax(next_logits, dim=-1)
        log_prob_dist = F.log_softmax(next_logits, dim=-1)

        # Sample
        token = torch.multinomial(probs, num_samples=1).squeeze(-1)

        # Collect log-prob and entropy
        gen_tokens.append(token.item())
        gen_logprobs.append(log_prob_dist[token].item())
        # Safe entropy: only sum over non-zero probability entries
        safe_probs = probs[probs > 0]
        ent = -(safe_probs * safe_probs.log()).sum().item()
        gen_entropy.append(ent)

        # Append to sequence
        sequence = torch.cat([sequence, token.unsqueeze(0).unsqueeze(0)], dim=1)

        # Stop at END
        if token.item() == end_id:
            break

    return (
        torch.tensor(gen_tokens, device=device),
        torch.tensor(gen_logprobs, device=device),
        torch.tensor(gen_entropy, device=device),
    )


def compute_kl_penalty(
    model: ARCSModel,
    ref_model: ARCSModel,
    sequence: torch.Tensor,
    gen_start_idx: int,
) -> torch.Tensor:
    """Compute per-token KL(ref || policy) for the generated portion.

    This penalizes the policy for deviating too far from the pre-trained model.

    Args:
        model: Current policy model
        ref_model: Frozen reference (pre-trained) model
        sequence: (1, T_total) full sequence (prefix + generated)
        gen_start_idx: Index where generation starts (prefix length)

    Returns:
        kl: scalar mean KL divergence over generated tokens
    """
    with torch.no_grad():
        ref_logits, _ = ref_model(sequence)
        pol_logits, _ = model(sequence)

    # Only compute KL over generated tokens
    ref_logprobs = F.log_softmax(ref_logits[0, gen_start_idx - 1 : -1, :], dim=-1)
    pol_logprobs = F.log_softmax(pol_logits[0, gen_start_idx - 1 : -1, :], dim=-1)

    # KL(ref || policy) = sum ref * (log ref - log policy)
    ref_probs = ref_logprobs.exp()
    kl = (ref_probs * (ref_logprobs - pol_logprobs)).sum(dim=-1).mean()

    return kl


# ---------------------------------------------------------------------------
# RL training specs sampler
# ---------------------------------------------------------------------------


def sample_training_specs(
    tokenizer: CircuitTokenizer,
    device: torch.device,
) -> tuple[str, dict[str, float], torch.Tensor]:
    """Sample a random topology + spec combination for RL training.

    Returns:
        topology: str (e.g., "buck")
        specs: dict of target specs
        prefix: (1, T_prefix) prefix tensor ready for model.generate()
    """
    topologies = list(OPERATING_CONDITIONS.keys())
    topo = np.random.choice(topologies)
    conditions = dict(OPERATING_CONDITIONS[topo])

    # Add small random perturbation to specs (±20%) for diversity
    specs = {}
    for key, val in conditions.items():
        if key == "fsw":
            specs[key] = val  # Keep switching frequency fixed
        else:
            factor = np.random.uniform(0.8, 1.2)
            specs[key] = val * factor

    # Build prefix tokens
    prefix_ids = [tokenizer.name_to_id["START"]]
    topo_key = f"TOPO_{topo.upper()}"
    if topo_key in tokenizer.name_to_id:
        prefix_ids.append(tokenizer.name_to_id[topo_key])
    prefix_ids.append(tokenizer.sep_id)

    for spec_name in ["vin", "vout", "iout", "fsw"]:
        if spec_name in specs:
            spec_key = f"SPEC_{spec_name.upper()}"
            if spec_key in tokenizer.name_to_id:
                prefix_ids.append(tokenizer.name_to_id[spec_key])
                prefix_ids.append(tokenizer.encode_value(abs(specs[spec_name])))
    prefix_ids.append(tokenizer.sep_id)

    prefix = torch.tensor([prefix_ids], device=device)
    return topo, specs, prefix


def sample_specs_for_topology(
    topology: str,
    tokenizer: CircuitTokenizer,
    device: torch.device,
) -> tuple[str, dict[str, float], torch.Tensor]:
    """Sample specs for a *specific* topology (used by GRPO).

    Like ``sample_training_specs`` but takes an explicit topology name
    instead of sampling uniformly.
    """
    conditions = dict(OPERATING_CONDITIONS[topology])

    specs: dict[str, float] = {}
    for key, val in conditions.items():
        if key == "fsw":
            specs[key] = val  # keep switching frequency fixed
        else:
            factor = np.random.uniform(0.8, 1.2)
            specs[key] = val * factor

    # Build prefix tokens
    prefix_ids = [tokenizer.name_to_id["START"]]
    topo_key = f"TOPO_{topology.upper()}"
    if topo_key in tokenizer.name_to_id:
        prefix_ids.append(tokenizer.name_to_id[topo_key])
    prefix_ids.append(tokenizer.sep_id)

    for spec_name in ["vin", "vout", "iout", "fsw"]:
        if spec_name in specs:
            spec_key = f"SPEC_{spec_name.upper()}"
            if spec_key in tokenizer.name_to_id:
                prefix_ids.append(tokenizer.name_to_id[spec_key])
                prefix_ids.append(tokenizer.encode_value(abs(specs[spec_name])))
    prefix_ids.append(tokenizer.sep_id)

    prefix = torch.tensor([prefix_ids], device=device)
    return topology, specs, prefix


# ---------------------------------------------------------------------------
# REINFORCE Trainer
# ---------------------------------------------------------------------------


@dataclass
class RLConfig:
    """RL training hyperparameters."""

    lr: float = 1e-5
    kl_coeff: float = 0.1
    entropy_coeff: float = 0.01
    reward_baseline_decay: float = 0.99
    max_grad_norm: float = 0.5
    temperature: float = DEFAULT_TEMPERATURE
    top_k: int = DEFAULT_TOP_K
    max_gen_tokens: int = 64
    batch_size: int = 8
    n_steps: int = 5000
    log_interval: int = 10
    save_interval: int = 500
    eval_interval: int = 100
    n_eval_samples: int = 50
    # Stability controls
    struct_bonus: float = 1.0         # structure validity reward weight
    kl_target: float = 0.5           # target KL divergence (adaptive coeff)
    adaptive_kl: bool = False        # enable adaptive KL coefficient
    validity_early_stop: float = 0.0 # stop if struct_valid drops below this (0 = disabled)
    # GRPO (Group Relative Policy Optimization)
    grpo: bool = False               # enable GRPO mode
    group_size: int = 4              # circuits per topology per step
    n_topos_per_step: int = 3        # topologies sampled per GRPO step
    grpo_clip_adv: float = 5.0       # clip advantage magnitude
    grpo_eps: float = 1e-4           # std normalization epsilon
    # v4: topology-aware control
    per_topology_eval_samples: int = 2          # samples per topology during per-topology eval
    per_topology_early_stop_patience: int = 0   # number of eval rounds w/o macro sim-valid improvement (0=disabled)
    per_topology_early_stop_delta: float = 0.0  # required macro sim-valid improvement to reset patience


class ARCSRLTrainer:
    """REINFORCE with baseline trainer for ARCS.

    Uses SPICE simulation as the reward signal to fine-tune the
    pre-trained model to generate circuits that actually work.
    """

    def __init__(
        self,
        model: ARCSModel,
        ref_model: ARCSModel,
        tokenizer: CircuitTokenizer,
        config: RLConfig,
        device: torch.device,
        output_dir: str | Path,
    ):
        self.model = model
        self.ref_model = ref_model
        self.tokenizer = tokenizer
        self.config = config
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Freeze reference model
        self.ref_model.eval()
        for p in self.ref_model.parameters():
            p.requires_grad = False

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.lr,
            betas=(0.9, 0.95),
            weight_decay=0.01,
        )

        # SPICE runner
        self.runner = NGSpiceRunner()

        # Reward baseline (running mean)
        self.baseline = 0.0

        # History
        self.history: dict[str, list] = defaultdict(list)
        self.global_step = 0
        self.best_topology_macro_sim_valid = 0.0
        self.topology_no_improve_count = 0

        # Whether model needs tokenizer for forward (graph transformer)
        self._is_graph_model = hasattr(self.model, 'compute_graph_features')

    def _model_forward(self, model, seq):
        """Forward pass, passing tokenizer if model is graph-aware."""
        if self._is_graph_model:
            return model(seq, tokenizer=self.tokenizer)
        return model(seq)

    def train_step(self) -> dict[str, float]:
        """REINFORCE step with proper gradient computation.

        For each episode in the batch:
            1. Sample specs → prefix
            2. Generate tokens (no grad), then recompute log-probs (with grad)
            3. Simulate → reward
            4. Compute loss = -advantage * sum(log_probs) + KL + entropy
        """
        self.model.train()
        self.optimizer.zero_grad()

        batch_rewards = []
        batch_kls = []
        batch_entropies = []
        batch_infos = []
        total_policy_loss = torch.tensor(0.0, device=self.device)

        for _ in range(self.config.batch_size):
            # Sample specs
            topo, specs, prefix = sample_training_specs(
                self.tokenizer, self.device
            )
            prefix_len = prefix.size(1)

            # Generate WITH gradients
            gen_tokens, log_probs, entropy = self._generate_with_grad(prefix)

            # Decode & simulate (no grad needed)
            all_ids = prefix[0].tolist() + gen_tokens.tolist()
            decoded = decode_generated_sequence(all_ids, self.tokenizer)
            outcome = simulate_decoded_circuit(decoded, self.runner)
            reward = compute_reward(decoded, outcome, specs, self.config.struct_bonus)

            # KL from ref model
            full_seq = torch.cat([prefix, gen_tokens.unsqueeze(0)], dim=1)
            with torch.no_grad():
                ref_logits, _ = self._model_forward(self.ref_model, full_seq)
            pol_logits, _ = self._model_forward(self.model, full_seq)

            ref_lp = F.log_softmax(
                ref_logits[0, prefix_len - 1 : -1, :], dim=-1
            )
            pol_lp = F.log_softmax(
                pol_logits[0, prefix_len - 1 : -1, :], dim=-1
            )
            ref_p = ref_lp.exp()
            kl = (ref_p * (ref_lp - pol_lp)).sum(dim=-1).mean()

            # Advantage
            advantage = reward - self.baseline

            # Policy loss: -advantage * mean(log_probs)
            # Use mean (not sum) over tokens to normalize by sequence length,
            # keeping gradients at a consistent scale regardless of gen length.
            if log_probs.numel() > 0:
                pg_loss = -(advantage * log_probs.mean())
            else:
                pg_loss = torch.tensor(0.0, device=self.device)

            # Total episode loss
            episode_loss = pg_loss + self.config.kl_coeff * kl - self.config.entropy_coeff * entropy.mean()
            total_policy_loss = total_policy_loss + episode_loss

            batch_rewards.append(reward)
            batch_kls.append(kl.item())
            batch_entropies.append(entropy.mean().item())

            info = {
                "topology": topo,
                "valid_structure": decoded.valid_structure,
                "sim_success": outcome.success,
                "sim_valid": outcome.valid,
                "n_components": len(decoded.components),
                "reward": reward,
            }
            if outcome.success:
                info["efficiency"] = outcome.metrics.get("efficiency", 0)
                info["vout_error"] = outcome.metrics.get("vout_error_pct", 100)
                info["ripple"] = outcome.metrics.get("ripple_ratio", 1.0)
            batch_infos.append(info)

        # Average loss and backprop
        total_policy_loss = total_policy_loss / self.config.batch_size

        # Safety clamp: prevent inf/nan from corrupting model weights
        if not torch.isfinite(total_policy_loss):
            logging.warning(
                f"Non-finite loss detected ({total_policy_loss.item():.4f}), "
                "skipping gradient update"
            )
            self.optimizer.zero_grad()
        else:
            total_policy_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.max_grad_norm
            )
            self.optimizer.step()

        # Update baseline
        mean_reward = np.mean(batch_rewards)
        self.baseline = (
            self.config.reward_baseline_decay * self.baseline
            + (1 - self.config.reward_baseline_decay) * mean_reward
        )

        # Aggregate stats
        stats = {
            "reward_mean": mean_reward,
            "reward_std": float(np.std(batch_rewards)),
            "kl_mean": float(np.mean(batch_kls)),
            "entropy_mean": float(np.mean(batch_entropies)),
            "loss": total_policy_loss.item(),
            "baseline": self.baseline,
            "sim_success_rate": np.mean(
                [info["sim_success"] for info in batch_infos]
            ),
            "valid_structure_rate": np.mean(
                [info["valid_structure"] for info in batch_infos]
            ),
            "sim_valid_rate": np.mean(
                [info.get("sim_valid", False) for info in batch_infos]
            ),
        }

        # Per-topology reward breakdown
        topo_rewards = defaultdict(list)
        for info in batch_infos:
            topo_rewards[info["topology"]].append(info["reward"])

        return stats

    # ------------------------------------------------------------------
    # GRPO: Group Relative Policy Optimization
    # ------------------------------------------------------------------

    def train_step_grpo(self) -> dict[str, float]:
        """GRPO training step with per-topology advantage normalization.

        Unlike vanilla REINFORCE which uses a single global EMA baseline,
        GRPO generates a *group* of circuits for each sampled topology and
        computes advantage = (r - μ_group) / (σ_group + ε).  This prevents
        cross-topology reward-scale interference (e.g. power converters
        with max reward ~8 vs signal circuits with max ~3).

        Per step:
            1. Sample ``n_topos_per_step`` topologies (without replacement).
            2. For each topology generate ``group_size`` circuits.
            3. Compute per-group z-scored advantages.
            4. REINFORCE loss aggregated across all groups.

        Effective batch = n_topos_per_step × group_size.
        """
        self.model.train()
        self.optimizer.zero_grad()

        all_rewards: list[float] = []
        all_kls: list[float] = []
        all_entropies: list[float] = []
        all_infos: list[dict] = []
        total_loss = torch.tensor(0.0, device=self.device)
        n_episodes = 0

        # ------ sample topologies for this step ------
        topologies = list(OPERATING_CONDITIONS.keys())
        n_to_sample = min(self.config.n_topos_per_step, len(topologies))
        selected_topos = list(
            np.random.choice(topologies, size=n_to_sample, replace=False)
        )

        for topo in selected_topos:
            # ---- generate a group of circuits for this topology ----
            group_rewards: list[float] = []
            group_episodes: list[
                tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            ] = []  # (log_probs, kl, entropy)
            group_infos: list[dict] = []

            for _ in range(self.config.group_size):
                _, specs, prefix = sample_specs_for_topology(
                    topo, self.tokenizer, self.device
                )
                prefix_len = prefix.size(1)

                # Generate WITH gradients
                gen_tokens, log_probs, entropy = self._generate_with_grad(
                    prefix
                )

                # Decode & simulate (no grad needed)
                all_ids = prefix[0].tolist() + gen_tokens.tolist()
                decoded = decode_generated_sequence(all_ids, self.tokenizer)
                outcome = simulate_decoded_circuit(decoded, self.runner)
                reward = compute_reward(
                    decoded, outcome, specs, self.config.struct_bonus
                )

                # KL from ref model
                full_seq = torch.cat(
                    [prefix, gen_tokens.unsqueeze(0)], dim=1
                )
                with torch.no_grad():
                    ref_logits, _ = self._model_forward(
                        self.ref_model, full_seq
                    )
                pol_logits, _ = self._model_forward(self.model, full_seq)

                ref_lp = F.log_softmax(
                    ref_logits[0, prefix_len - 1 : -1, :], dim=-1
                )
                pol_lp = F.log_softmax(
                    pol_logits[0, prefix_len - 1 : -1, :], dim=-1
                )
                kl = (ref_lp.exp() * (ref_lp - pol_lp)).sum(-1).mean()

                group_rewards.append(reward)
                group_episodes.append((log_probs, kl, entropy))

                info: dict = {
                    "topology": topo,
                    "valid_structure": decoded.valid_structure,
                    "sim_success": outcome.success,
                    "sim_valid": outcome.valid,
                    "n_components": len(decoded.components),
                    "reward": reward,
                }
                if outcome.success:
                    info["efficiency"] = outcome.metrics.get("efficiency", 0)
                    info["vout_error"] = outcome.metrics.get(
                        "vout_error_pct", 100
                    )
                    info["ripple"] = outcome.metrics.get("ripple_ratio", 1.0)
                group_infos.append(info)

            # ---- compute group-relative advantages ----
            grp = np.array(group_rewards)
            grp_mean = grp.mean()
            # Use ddof=1 (Bessel correction) for unbiased sample std;
            # ddof=0 underestimates std for small groups, inflating advantages.
            grp_std = grp.std(ddof=1) if len(grp) > 1 else 0.0

            for i, (log_probs, kl, entropy) in enumerate(group_episodes):
                adv = (group_rewards[i] - grp_mean) / (
                    grp_std + self.config.grpo_eps
                )
                adv = float(
                    np.clip(
                        adv,
                        -self.config.grpo_clip_adv,
                        self.config.grpo_clip_adv,
                    )
                )

                if log_probs.numel() > 0:
                    pg_loss = -(adv * log_probs.mean())
                else:
                    pg_loss = torch.tensor(0.0, device=self.device)

                episode_loss = (
                    pg_loss
                    + self.config.kl_coeff * kl
                    - self.config.entropy_coeff * entropy.mean()
                )
                total_loss = total_loss + episode_loss
                n_episodes += 1

            all_rewards.extend(group_rewards)
            all_kls.extend([ep[1].item() for ep in group_episodes])
            all_entropies.extend(
                [ep[2].mean().item() for ep in group_episodes]
            )
            all_infos.extend(group_infos)

        # ---- backprop ----
        if n_episodes > 0:
            total_loss = total_loss / n_episodes

        if not torch.isfinite(total_loss):
            logger.warning("Non-finite GRPO loss, skipping gradient update")
            self.optimizer.zero_grad()
        else:
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.max_grad_norm
            )
            self.optimizer.step()

        # Update baseline (for logging & backward compat)
        mean_reward = float(np.mean(all_rewards)) if all_rewards else 0.0
        self.baseline = (
            self.config.reward_baseline_decay * self.baseline
            + (1 - self.config.reward_baseline_decay) * mean_reward
        )

        # Per-topology reward breakdown
        topo_rewards: dict[str, list[float]] = defaultdict(list)
        for info in all_infos:
            topo_rewards[info["topology"]].append(info["reward"])

        stats: dict[str, float | int | dict] = {
            "reward_mean": mean_reward,
            "reward_std": float(np.std(all_rewards)) if all_rewards else 0.0,
            "kl_mean": (
                float(np.mean(all_kls)) if all_kls else 0.0
            ),
            "entropy_mean": (
                float(np.mean(all_entropies)) if all_entropies else 0.0
            ),
            "loss": total_loss.item(),
            "baseline": self.baseline,
            "sim_success_rate": (
                float(np.mean([i["sim_success"] for i in all_infos]))
                if all_infos
                else 0.0
            ),
            "valid_structure_rate": (
                float(np.mean([i["valid_structure"] for i in all_infos]))
                if all_infos
                else 0.0
            ),
            "sim_valid_rate": (
                float(
                    np.mean([i.get("sim_valid", False) for i in all_infos])
                )
                if all_infos
                else 0.0
            ),
            "n_topologies": len(selected_topos),
        }

        return stats

    def _generate_with_grad(
        self,
        prefix: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate tokens with gradient-tracked log-probs.

        Uses teacher-forcing-like approach: sample tokens, but compute
        log-probs through the full model forward pass at the end.
        This allows backprop through the log-prob computation.
        """
        device = prefix.device
        end_id = self.tokenizer.name_to_id.get("END", 2)
        sequence = prefix.clone()

        gen_token_ids = []
        entropies = []

        # First: sample tokens without grad (fast)
        with torch.no_grad():
            for _ in range(self.config.max_gen_tokens):
                logits, _ = self._model_forward(self.model, sequence)
                next_logits = logits[0, -1, :] / self.config.temperature

                if self.config.top_k > 0:
                    vals, _ = torch.topk(
                        next_logits,
                        min(self.config.top_k, next_logits.size(0)),
                    )
                    next_logits[next_logits < vals[-1]] = float("-inf")

                probs = F.softmax(next_logits, dim=-1)
                log_p = F.log_softmax(next_logits, dim=-1)
                token = torch.multinomial(probs, 1).squeeze(-1)

                gen_token_ids.append(token.item())
                safe_p = probs[probs > 0]
                ent = -(safe_p * safe_p.log()).sum().item()
                entropies.append(ent)

                sequence = torch.cat(
                    [sequence, token.unsqueeze(0).unsqueeze(0)], dim=1
                )
                if token.item() == end_id:
                    break

        if not gen_token_ids:
            return (
                torch.tensor([], dtype=torch.long, device=device),
                torch.tensor([], device=device),
                torch.tensor([], device=device),
            )

        # Second: compute log-probs WITH grad via a single forward pass
        gen_tensor = torch.tensor(gen_token_ids, dtype=torch.long, device=device)
        full_seq = torch.cat([prefix, gen_tensor.unsqueeze(0)], dim=1)

        # Full model forward with grad
        logits, _ = self._model_forward(self.model, full_seq)  # (1, T, vocab)

        # Extract log-probs for the generated tokens
        # logits[:, prefix_len-1 : prefix_len-1+n_gen, :] → predicted distributions
        # for tokens at positions prefix_len ... prefix_len+n_gen-1
        prefix_len = prefix.size(1)
        n_gen = len(gen_token_ids)
        pred_logits = logits[0, prefix_len - 1 : prefix_len - 1 + n_gen, :]
        pred_logits = pred_logits / self.config.temperature

        # Apply same top-k mask
        if self.config.top_k > 0:
            for t in range(n_gen):
                vals, _ = torch.topk(
                    pred_logits[t],
                    min(self.config.top_k, pred_logits.size(-1)),
                )
                pred_logits[t][pred_logits[t] < vals[-1]] = float("-inf")

        log_probs_all = F.log_softmax(pred_logits, dim=-1)  # (n_gen, vocab)
        log_probs = log_probs_all[
            torch.arange(n_gen, device=device), gen_tensor
        ]  # (n_gen,)

        # Clamp to avoid -inf when a generated token falls outside the
        # top-k set during the second (gradient) pass. -20 ≈ prob 2e-9,
        # effectively zero but still finite for stable gradient computation.
        log_probs = log_probs.clamp(min=-20.0)

        # Compute entropy from the grad-enabled forward pass so the entropy
        # bonus term actually produces gradients (the sampling-loop entropies
        # were detached via .item() and couldn't influence training).
        probs_all = log_probs_all.exp()
        # Mask out zero-prob entries to avoid NaN in log
        safe_probs = probs_all.clamp(min=1e-10)
        grad_entropies = -(probs_all * safe_probs.log()).sum(dim=-1)  # (n_gen,)

        return gen_tensor, log_probs, grad_entropies

    def evaluate(self, n_samples: int | None = None) -> dict[str, float]:
        """Evaluate current policy by generating and simulating circuits.

        Returns aggregate metrics without gradient computation.
        """
        n = n_samples or self.config.n_eval_samples
        self.model.eval()

        rewards = []
        sim_successes = 0
        sim_valids = 0
        struct_valids = 0
        efficiencies = []
        vout_errors = []
        topo_counts = defaultdict(int)

        for _ in range(n):
            topo, specs, prefix = sample_training_specs(
                self.tokenizer, self.device
            )

            gen_tokens, _, _ = sample_with_logprobs(
                self.model,
                prefix,
                self.tokenizer,
                max_new_tokens=self.config.max_gen_tokens,
                temperature=self.config.temperature,
                top_k=self.config.top_k,
            )

            all_ids = prefix[0].tolist() + gen_tokens.tolist()
            decoded = decode_generated_sequence(all_ids, self.tokenizer)

            if decoded.valid_structure:
                struct_valids += 1
                outcome = simulate_decoded_circuit(decoded, self.runner)
                reward = compute_reward(decoded, outcome, specs, self.config.struct_bonus)

                if outcome.success:
                    sim_successes += 1
                    if outcome.valid:
                        sim_valids += 1
                    efficiencies.append(
                        outcome.metrics.get("efficiency", 0)
                    )
                    vout_errors.append(
                        outcome.metrics.get("vout_error_pct", 100)
                    )
            else:
                reward = 0.0

            rewards.append(reward)
            topo_counts[topo] += 1

        return {
            "eval_reward_mean": np.mean(rewards),
            "eval_reward_std": float(np.std(rewards)),
            "eval_struct_valid_rate": struct_valids / max(n, 1),
            "eval_sim_success_rate": sim_successes / max(n, 1),
            "eval_sim_valid_rate": sim_valids / max(n, 1),
            "eval_mean_efficiency": (
                np.mean(efficiencies) if efficiencies else 0.0
            ),
            "eval_mean_vout_error": (
                np.mean(vout_errors) if vout_errors else 100.0
            ),
        }

    def evaluate_per_topology(
        self,
        n_per_topology: int | None = None,
    ) -> dict[str, dict[str, float]]:
        """Evaluate policy per topology for topology-aware control.

        Returns a dict keyed by topology containing:
            reward_mean, struct_valid_rate, sim_success_rate, sim_valid_rate
        """
        n = n_per_topology or self.config.per_topology_eval_samples
        n = max(1, int(n))

        self.model.eval()
        results: dict[str, dict[str, float]] = {}

        for topo in OPERATING_CONDITIONS.keys():
            rewards: list[float] = []
            struct_valids = 0
            sim_successes = 0
            sim_valids = 0

            for _ in range(n):
                _, specs, prefix = sample_specs_for_topology(
                    topo, self.tokenizer, self.device
                )

                gen_tokens, _, _ = sample_with_logprobs(
                    self.model,
                    prefix,
                    self.tokenizer,
                    max_new_tokens=self.config.max_gen_tokens,
                    temperature=self.config.temperature,
                    top_k=self.config.top_k,
                )

                all_ids = prefix[0].tolist() + gen_tokens.tolist()
                decoded = decode_generated_sequence(all_ids, self.tokenizer)

                if decoded.valid_structure:
                    struct_valids += 1
                    outcome = simulate_decoded_circuit(decoded, self.runner)
                    reward = compute_reward(decoded, outcome, specs, self.config.struct_bonus)
                    if outcome.success:
                        sim_successes += 1
                        if outcome.valid:
                            sim_valids += 1
                else:
                    reward = 0.0

                rewards.append(reward)

            results[topo] = {
                "reward_mean": float(np.mean(rewards)) if rewards else 0.0,
                "struct_valid_rate": struct_valids / n,
                "sim_success_rate": sim_successes / n,
                "sim_valid_rate": sim_valids / n,
            }

        return results

    def _update_topology_early_stop_state(
        self,
        per_topology_eval: dict[str, dict[str, float]],
    ) -> tuple[bool, float]:
        """Update topology-aware patience state and return stop flag + macro sim-valid.

        Macro sim-valid is the unweighted mean of per-topology sim-valid rates.
        """
        if not per_topology_eval:
            return False, 0.0

        macro_sim_valid = float(
            np.mean([v["sim_valid_rate"] for v in per_topology_eval.values()])
        )

        if (
            macro_sim_valid
            > self.best_topology_macro_sim_valid + self.config.per_topology_early_stop_delta
        ):
            self.best_topology_macro_sim_valid = macro_sim_valid
            self.topology_no_improve_count = 0
        else:
            self.topology_no_improve_count += 1

        should_stop = (
            self.config.per_topology_early_stop_patience > 0
            and self.topology_no_improve_count >= self.config.per_topology_early_stop_patience
        )
        return should_stop, macro_sim_valid

    def save_checkpoint(self, path: str | Path | None = None) -> Path:
        """Save model + optimizer + training state."""
        if path is None:
            path = self.output_dir / f"rl_checkpoint_step{self.global_step}.pt"
        else:
            path = Path(path)

        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "config": self.model.config.to_dict(),
                "rl_config": {
                    "lr": self.config.lr,
                    "kl_coeff": self.config.kl_coeff,
                    "entropy_coeff": self.config.entropy_coeff,
                    "temperature": self.config.temperature,
                    "top_k": self.config.top_k,
                    "batch_size": self.config.batch_size,
                },
                "global_step": self.global_step,
                "baseline": self.baseline,
                "history": dict(self.history),
                "best_topology_macro_sim_valid": self.best_topology_macro_sim_valid,
                "topology_no_improve_count": self.topology_no_improve_count,
            },
            path,
        )
        return path

    def resume_from_checkpoint(self, path: str | Path) -> float:
        """Resume training from a saved RL checkpoint. Returns best_reward."""
        path = Path(path)
        logger.info(f"Resuming RL from checkpoint: {path}")
        ckpt = torch.load(path, map_location=self.device, weights_only=False)

        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.global_step = ckpt["global_step"]
        self.baseline = ckpt.get("baseline", 0.0)
        self.best_topology_macro_sim_valid = ckpt.get("best_topology_macro_sim_valid", 0.0)
        self.topology_no_improve_count = ckpt.get("topology_no_improve_count", 0)

        # Restore history
        saved_history = ckpt.get("history", {})
        for k, v in saved_history.items():
            self.history[k] = v

        # Restore KL coeff if adaptive
        rl_cfg = ckpt.get("rl_config", {})
        if "kl_coeff" in rl_cfg:
            self.config.kl_coeff = rl_cfg["kl_coeff"]

        # Derive best_reward from eval history
        best_reward = float("-inf")
        for ev in self.history.get("eval", []):
            r = ev.get("eval_reward_mean", float("-inf"))
            if r > best_reward:
                best_reward = r

        logger.info(
            f"  Resumed at step {self.global_step}, "
            f"baseline={self.baseline:.3f}, best_reward={best_reward:.3f}"
        )
        return best_reward

    def train(self, resume_best_reward: float | None = None) -> dict:
        """Full RL training loop with adaptive KL & validity early stopping.

        Args:
            resume_best_reward: If resuming, pass the best reward from the
                checkpoint so we skip the initial eval and start from the
                saved global_step.
        """
        mode = "GRPO" if self.config.grpo else "REINFORCE"
        eff_batch = (
            f"{self.config.n_topos_per_step}×{self.config.group_size}"
            if self.config.grpo
            else str(self.config.batch_size)
        )

        start_step = self.global_step + 1  # 1 for fresh, global_step+1 for resume

        logger.info(
            f"Starting RL training ({mode}): steps {start_step}..{self.config.n_steps}, "
            f"batch={eff_batch}, lr={self.config.lr}, "
            f"kl={self.config.kl_coeff}, struct_bonus={self.config.struct_bonus}, "
            f"adaptive_kl={self.config.adaptive_kl}, temp={self.config.temperature}"
        )

        if resume_best_reward is not None:
            best_reward = resume_best_reward
            logger.info(f"Resumed from step {self.global_step}, best_reward={best_reward:.3f}")
        else:
            # Initial evaluation
            eval_results = self.evaluate()
            logger.info(f"Initial eval: {eval_results}")
            self.history["eval"].append({"step": 0, **eval_results})
            best_reward = eval_results["eval_reward_mean"]

        t0 = time.time()

        for step in range(start_step, self.config.n_steps + 1):
            self.global_step = step
            if self.config.grpo:
                stats = self.train_step_grpo()
            else:
                stats = self.train_step()

            # Record history
            for k, v in stats.items():
                self.history[k].append(v)

            # Adaptive KL coefficient: if KL > 1.5× target, increase coeff;
            # if KL < 0.5× target, decrease slightly.
            if self.config.adaptive_kl and step % self.config.log_interval == 0:
                kl_now = stats["kl_mean"]
                if kl_now > 1.5 * self.config.kl_target:
                    self.config.kl_coeff = min(self.config.kl_coeff * 1.5, 10.0)
                    logger.info(f"  Adaptive KL: raised coeff → {self.config.kl_coeff:.4f}")
                elif kl_now < 0.5 * self.config.kl_target:
                    self.config.kl_coeff = max(self.config.kl_coeff * 0.8, 0.01)
                    logger.info(f"  Adaptive KL: lowered coeff → {self.config.kl_coeff:.4f}")

            # Log
            if step % self.config.log_interval == 0:
                dt = time.time() - t0
                steps_per_sec = step / dt
                logger.info(
                    f"Step {step:5d}/{self.config.n_steps} | "
                    f"reward={stats['reward_mean']:.3f}±{stats['reward_std']:.2f} | "
                    f"loss={stats['loss']:.4f} | "
                    f"kl={stats['kl_mean']:.4f} | "
                    f"kl_coeff={self.config.kl_coeff:.3f} | "
                    f"sim_ok={stats['sim_success_rate']:.0%} | "
                    f"valid={stats['sim_valid_rate']:.0%} | "
                    f"struct={stats['valid_structure_rate']:.0%} | "
                    f"base={stats['baseline']:.3f} | "
                    f"{steps_per_sec:.2f} steps/s"
                )

            # Evaluate
            if step % self.config.eval_interval == 0:
                eval_results = self.evaluate()
                self.history["eval"].append({"step": step, **eval_results})
                logger.info(
                    f"  EVAL step {step}: "
                    f"reward={eval_results['eval_reward_mean']:.3f} | "
                    f"struct_valid={eval_results['eval_struct_valid_rate']:.0%} | "
                    f"sim_valid={eval_results['eval_sim_valid_rate']:.0%} | "
                    f"eff={eval_results['eval_mean_efficiency']:.3f} | "
                    f"vout_err={eval_results['eval_mean_vout_error']:.1f}%"
                )

                if eval_results["eval_reward_mean"] > best_reward:
                    best_reward = eval_results["eval_reward_mean"]
                    self.save_checkpoint(self.output_dir / "best_rl_model.pt")
                    logger.info(
                        f"  New best reward: {best_reward:.3f}"
                    )

                # Topology-aware evaluation / early stop
                if self.config.per_topology_early_stop_patience > 0:
                    topo_eval = self.evaluate_per_topology()
                    macro_struct = float(
                        np.mean([v["struct_valid_rate"] for v in topo_eval.values()])
                    )
                    macro_sim = float(
                        np.mean([v["sim_valid_rate"] for v in topo_eval.values()])
                    )
                    worst_topo, worst_stats = min(
                        topo_eval.items(),
                        key=lambda kv: kv[1]["sim_valid_rate"],
                    )
                    logger.info(
                        "  TOPO EVAL step %d: macro_struct=%.0f%% macro_sim_valid=%.0f%% "
                        "worst=%s(%.0f%%)",
                        step,
                        100 * macro_struct,
                        100 * macro_sim,
                        worst_topo,
                        100 * worst_stats["sim_valid_rate"],
                    )

                    self.history["eval_topology"].append({
                        "step": step,
                        "macro_struct_valid_rate": macro_struct,
                        "macro_sim_valid_rate": macro_sim,
                        "worst_topology": worst_topo,
                        "worst_topology_sim_valid_rate": worst_stats["sim_valid_rate"],
                        "per_topology": topo_eval,
                    })

                    should_stop_topo, macro_sim_valid = self._update_topology_early_stop_state(topo_eval)
                    if should_stop_topo:
                        logger.warning(
                            "  EARLY STOP (topology-aware): macro_sim_valid %.0f%% did not improve "
                            "for %d eval rounds (best=%.0f%%, delta=%.2f)",
                            100 * macro_sim_valid,
                            self.topology_no_improve_count,
                            100 * self.best_topology_macro_sim_valid,
                            self.config.per_topology_early_stop_delta,
                        )
                        break

                # Validity early stopping
                if (self.config.validity_early_stop > 0
                        and eval_results["eval_struct_valid_rate"] < self.config.validity_early_stop):
                    logger.warning(
                        f"  EARLY STOP: struct_valid "
                        f"{eval_results['eval_struct_valid_rate']:.0%} < "
                        f"{self.config.validity_early_stop:.0%} threshold"
                    )
                    break

            # Save periodic checkpoint
            if step % self.config.save_interval == 0:
                self.save_checkpoint()

        # Final save
        self.save_checkpoint(self.output_dir / "final_rl_model.pt")

        # Save history
        with open(self.output_dir / "rl_history.json", "w") as f:
            json.dump(dict(self.history), f, indent=2, default=str)

        total_time = time.time() - t0
        logger.info(
            f"RL training complete. {self.config.n_steps} steps in {total_time:.0f}s. "
            f"Best reward: {best_reward:.3f}"
        )

        return {
            "total_steps": self.config.n_steps,
            "total_time": total_time,
            "best_reward": best_reward,
            "final_eval": eval_results,
        }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="ARCS Phase 3: SPICE-in-the-Loop RL Training"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to pre-trained model checkpoint (.pt)",
    )
    parser.add_argument("--output", type=str, default="checkpoints/arcs_rl")
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--kl-coeff", type=float, default=0.1)
    parser.add_argument("--entropy-coeff", type=float, default=0.01)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--save-interval", type=int, default=500)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--n-eval-samples", type=int, default=50)
    parser.add_argument("--struct-bonus", type=float, default=1.0,
                        help="Structure validity reward weight (v2: use 2.0)")
    parser.add_argument("--kl-target", type=float, default=0.5,
                        help="Target KL for adaptive coefficient")
    parser.add_argument("--adaptive-kl", action="store_true",
                        help="Enable adaptive KL coefficient")
    parser.add_argument("--validity-early-stop", type=float, default=0.0,
                        help="Stop if struct_valid drops below this (0=disabled)")
    # GRPO options
    parser.add_argument("--grpo", action="store_true",
                        help="Enable Group Relative Policy Optimization")
    parser.add_argument("--group-size", type=int, default=4,
                        help="GRPO: circuits per topology per step")
    parser.add_argument("--n-topos-per-step", type=int, default=3,
                        help="GRPO: topologies sampled per step")
    parser.add_argument("--grpo-clip-adv", type=float, default=5.0,
                        help="GRPO: clip advantage magnitude")
    parser.add_argument("--per-topology-eval-samples", type=int, default=2,
                        help="Per-topology eval samples each eval round")
    parser.add_argument("--per-topology-early-stop-patience", type=int, default=0,
                        help="Stop if macro per-topology sim-valid doesn't improve for N eval rounds (0=disabled)")
    parser.add_argument("--per-topology-early-stop-delta", type=float, default=0.0,
                        help="Minimum macro per-topology sim-valid improvement to reset patience")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to RL checkpoint to resume from (e.g. checkpoints/arcs_grpo_v2/rl_checkpoint_step500.pt)")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    # Seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Device
    if args.device == "auto":
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    logger.info(f"Using device: {device}")

    # Load pre-trained model (auto-detects model type from checkpoint)
    logger.info(f"Loading checkpoint: {args.checkpoint}")
    model, config, model_type = load_model(args.checkpoint, device=device)
    model.train()  # load_model sets eval mode; switch back to train
    logger.info(
        f"Loaded {model_type} model "
        f"({model.count_parameters():,} params)"
    )

    # Create frozen reference model (same architecture, frozen weights)
    ref_model, _, _ = load_model(args.checkpoint, device=device)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False

    tokenizer = CircuitTokenizer()

    # RL config
    rl_config = RLConfig(
        lr=args.lr,
        kl_coeff=args.kl_coeff,
        entropy_coeff=args.entropy_coeff,
        temperature=args.temperature,
        top_k=args.top_k,
        batch_size=args.batch_size,
        n_steps=args.steps,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        eval_interval=args.eval_interval,
        n_eval_samples=args.n_eval_samples,
        struct_bonus=args.struct_bonus,
        kl_target=args.kl_target,
        adaptive_kl=args.adaptive_kl,
        validity_early_stop=args.validity_early_stop,
        grpo=args.grpo,
        group_size=args.group_size,
        n_topos_per_step=args.n_topos_per_step,
        grpo_clip_adv=args.grpo_clip_adv,
        per_topology_eval_samples=args.per_topology_eval_samples,
        per_topology_early_stop_patience=args.per_topology_early_stop_patience,
        per_topology_early_stop_delta=args.per_topology_early_stop_delta,
    )

    # Train
    trainer = ARCSRLTrainer(
        model=model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        config=rl_config,
        device=device,
        output_dir=args.output,
    )

    # Save args
    with open(Path(args.output) / "rl_args.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    # Resume from RL checkpoint if specified
    resume_best_reward = None
    if args.resume:
        resume_best_reward = trainer.resume_from_checkpoint(args.resume)

    results = trainer.train(resume_best_reward=resume_best_reward)

    logger.info(f"\nFinal results: {json.dumps(results, indent=2, default=str)}")


if __name__ == "__main__":
    main()
