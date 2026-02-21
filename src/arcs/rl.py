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
from arcs.spice import NGSpiceRunner
from arcs.templates import (
    OPERATING_CONDITIONS,
    POWER_CONVERTER_BOUNDS,
    get_topology,
)
from arcs.tokenizer import CircuitTokenizer, TokenType

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Token → Parameter inverse mapping
# ---------------------------------------------------------------------------

# Per-topology: ordered list of (component_type, param_name) pairs.
# This mirrors the iteration order in tokenizer._params_to_components(),
# which iterates over the params dict.  For Python 3.7+ dicts preserve
# insertion order, so this matches DataGenerator's param sampling order.

COMPONENT_TO_PARAM: dict[str, list[tuple[str, str]]] = {
    "buck": [
        ("INDUCTOR", "inductance"),
        ("CAPACITOR", "capacitance"),
        ("RESISTOR", "esr"),
        ("MOSFET_N", "r_dson"),
    ],
    "boost": [
        ("INDUCTOR", "inductance"),
        ("CAPACITOR", "capacitance"),
        ("RESISTOR", "esr"),
        ("MOSFET_N", "r_dson"),
    ],
    "buck_boost": [
        ("INDUCTOR", "inductance"),
        ("CAPACITOR", "capacitance"),
        ("RESISTOR", "esr"),
        ("MOSFET_N", "r_dson"),
    ],
    "cuk": [
        ("INDUCTOR", "inductance_1"),
        ("INDUCTOR", "inductance_2"),
        ("CAPACITOR", "cap_coupling"),
        ("CAPACITOR", "capacitance"),
        ("RESISTOR", "esr"),
        ("MOSFET_N", "r_dson"),
    ],
    "sepic": [
        ("INDUCTOR", "inductance_1"),
        ("INDUCTOR", "inductance_2"),
        ("CAPACITOR", "cap_coupling"),
        ("CAPACITOR", "capacitance"),
        ("RESISTOR", "esr"),
        ("MOSFET_N", "r_dson"),
    ],
    "flyback": [
        ("INDUCTOR", "inductance_primary"),
        ("TRANSFORMER", "turns_ratio"),
        ("CAPACITOR", "capacitance"),
        ("RESISTOR", "esr"),
        ("MOSFET_N", "r_dson"),
    ],
    "forward": [
        ("INDUCTOR", "inductance_primary"),
        ("TRANSFORMER", "turns_ratio"),
        ("INDUCTOR", "inductance_output"),
        ("CAPACITOR", "capacitance"),
        ("RESISTOR", "esr"),
        ("MOSFET_N", "r_dson"),
    ],
}


def components_to_params(
    topology: str,
    components: list[tuple[str, float]],
) -> dict[str, float] | None:
    """Inverse of tokenizer._params_to_components().

    Maps a list of (component_type, value) pairs back to a parameter dict
    suitable for TopologyTemplate.generate_netlist().

    Returns None if the component list doesn't match the expected topology
    template (wrong types or wrong count).
    """
    expected = COMPONENT_TO_PARAM.get(topology)
    if expected is None:
        return None

    # Strategy: match in order.  The tokenizer emits components in
    # parameter-dict iteration order, and the model was trained on that.
    # We iterate through expected slots and consume matching components.
    params: dict[str, float] = {}
    comp_idx = 0

    for expected_type, param_name in expected:
        if comp_idx >= len(components):
            # Not enough components — use default from bounds
            break
        comp_type, comp_val = components[comp_idx]
        if comp_type.upper() == expected_type:
            params[param_name] = comp_val
            comp_idx += 1
        else:
            # Type mismatch — try to continue (model may have
            # generated components in a different order due to shuffle aug)
            # Search remaining components for a match
            found = False
            for j in range(comp_idx, len(components)):
                if components[j][0].upper() == expected_type:
                    params[param_name] = components[j][1]
                    # Remove consumed component and reset
                    components = components[:j] + components[j + 1 :]
                    found = True
                    break
            if not found:
                # Missing required component — use middle of bounds
                bounds = POWER_CONVERTER_BOUNDS.get(topology)
                if bounds:
                    for b in bounds:
                        if b.name == param_name:
                            params[param_name] = math.sqrt(b.min_val * b.max_val)
                            break

    return params if len(params) >= 3 else None  # Need at least 3 params


# ---------------------------------------------------------------------------
# SPICE simulation of decoded circuits
# ---------------------------------------------------------------------------


@dataclass
class SimulationOutcome:
    """Result of simulating a decoded circuit."""

    success: bool
    metrics: dict[str, float] = field(default_factory=dict)
    valid: bool = False
    error: str = ""
    sim_time: float = 0.0


def simulate_decoded_circuit(
    decoded: DecodedCircuit,
    runner: NGSpiceRunner | None = None,
    custom_conditions: dict[str, float] | None = None,
) -> SimulationOutcome:
    """Full pipeline: DecodedCircuit → SPICE netlist → simulate → metrics.

    Args:
        decoded: Decoded circuit with topology, specs, and components
        runner: NGSpiceRunner instance (creates one if None)
        custom_conditions: Override operating conditions (uses specs from decoded
                          circuit if available, else topology defaults)

    Returns:
        SimulationOutcome with metrics and validity flag
    """
    if not decoded.valid_structure or not decoded.topology:
        return SimulationOutcome(success=False, error="Invalid structure")

    # Get topology template
    try:
        template = get_topology(decoded.topology)
    except ValueError as e:
        return SimulationOutcome(success=False, error=str(e))

    # Map components back to parameters
    params = components_to_params(decoded.topology, list(decoded.components))
    if params is None:
        return SimulationOutcome(
            success=False, error="Could not map components to params"
        )

    # Determine operating conditions
    conditions = dict(template.operating_conditions)
    # Override with decoded specs if available
    spec_to_cond = {
        "vin": "vin",
        "vout": "vout",
        "iout": "iout",
        "fsw": "fsw",
    }
    if decoded.specs:
        for spec_key, cond_key in spec_to_cond.items():
            if spec_key in decoded.specs:
                conditions[cond_key] = decoded.specs[spec_key]
    if custom_conditions:
        conditions.update(custom_conditions)

    # Build netlist (need to also pass conditions into the template)
    # The template.generate_netlist uses self.operating_conditions,
    # so we temporarily override it
    old_conds = template.operating_conditions
    template.operating_conditions = conditions
    try:
        netlist = template.generate_netlist(params)
    except Exception as e:
        template.operating_conditions = old_conds
        return SimulationOutcome(success=False, error=f"Netlist error: {e}")
    finally:
        template.operating_conditions = old_conds

    # Simulate
    if runner is None:
        runner = NGSpiceRunner()
    try:
        sim_result = runner.run(netlist, template.metric_names)
    except Exception as e:
        return SimulationOutcome(success=False, error=f"Sim error: {e}")

    if not sim_result.success:
        return SimulationOutcome(
            success=False,
            error=sim_result.error_message or "Simulation failed",
            sim_time=sim_result.sim_time_seconds,
        )

    # Compute derived metrics
    try:
        metrics = compute_derived_metrics(
            sim_result.metrics, conditions, decoded.topology
        )
        valid = is_valid_result(metrics, conditions)
    except Exception as e:
        return SimulationOutcome(
            success=True,
            metrics=sim_result.metrics,
            valid=False,
            error=f"Metric error: {e}",
            sim_time=sim_result.sim_time_seconds,
        )

    return SimulationOutcome(
        success=True,
        metrics=metrics,
        valid=valid,
        sim_time=sim_result.sim_time_seconds,
    )


# ---------------------------------------------------------------------------
# Reward function
# ---------------------------------------------------------------------------


def compute_reward(
    decoded: DecodedCircuit,
    outcome: SimulationOutcome,
    target_specs: dict[str, float] | None = None,
) -> float:
    """Compute scalar reward from simulation outcome.

    Reward components (max ≈ 8.0):
        +1.0  — valid circuit structure (has topo, components, END)
        +1.0  — SPICE simulation converges
        +3.0  — Vout accuracy:  3.0 × max(0, 1 - vout_error/10)
                (full credit at <1% error, zero at >10% error)
        +2.0  — Efficiency:     2.0 × efficiency
                (0.9 eff → 1.8 points)
        +1.0  — Low ripple:     1.0 × max(0, 1 - ripple_ratio×10)
                (full credit at <1% ripple, zero at >10%)

    Without simulation (structure-only): max 1.0
    With failed simulation: max 2.0
    With successful simulation: max 8.0
    """
    reward = 0.0

    # Structure reward
    if decoded.valid_structure:
        reward += 1.0
    else:
        return 0.0  # No further reward for broken structure

    # Simulation convergence reward
    if not outcome.success:
        return reward  # 1.0 for structure only

    reward += 1.0  # Sim converged

    # Vout accuracy (max +3.0)
    vout_error = outcome.metrics.get("vout_error_pct", 100.0)
    vout_score = max(0.0, 1.0 - vout_error / 10.0)
    reward += 3.0 * vout_score

    # Efficiency (max +2.0)
    eff = outcome.metrics.get("efficiency", 0.0)
    eff = max(0.0, min(1.0, eff))
    reward += 2.0 * eff

    # Ripple (max +1.0)
    ripple = outcome.metrics.get("ripple_ratio", 1.0)
    ripple_score = max(0.0, 1.0 - ripple * 10.0)
    reward += 1.0 * ripple_score

    return reward


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
    temperature: float = 0.8
    top_k: int = 50
    max_gen_tokens: int = 64
    batch_size: int = 8
    n_steps: int = 5000
    log_interval: int = 10
    save_interval: int = 500
    eval_interval: int = 100
    n_eval_samples: int = 50


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
            reward = compute_reward(decoded, outcome, specs)

            # KL from ref model
            full_seq = torch.cat([prefix, gen_tokens.unsqueeze(0)], dim=1)
            with torch.no_grad():
                ref_logits, _ = self.ref_model(full_seq)
            pol_logits, _ = self.model(full_seq)

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
                logits, _ = self.model(sequence)
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
        logits, _ = self.model(full_seq)  # (1, T, vocab)

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

        return gen_tensor, log_probs, torch.tensor(entropies, device=device)

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
                reward = compute_reward(decoded, outcome, specs)

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
            },
            path,
        )
        return path

    def train(self) -> dict:
        """Full RL training loop."""
        logger.info(
            f"Starting RL training: {self.config.n_steps} steps, "
            f"batch={self.config.batch_size}, lr={self.config.lr}, "
            f"kl={self.config.kl_coeff}, temp={self.config.temperature}"
        )

        # Initial evaluation
        eval_results = self.evaluate()
        logger.info(f"Initial eval: {eval_results}")
        self.history["eval"].append({"step": 0, **eval_results})

        best_reward = eval_results["eval_reward_mean"]
        t0 = time.time()

        for step in range(1, self.config.n_steps + 1):
            self.global_step = step
            stats = self.train_step()

            # Record history
            for k, v in stats.items():
                self.history[k].append(v)

            # Log
            if step % self.config.log_interval == 0:
                dt = time.time() - t0
                steps_per_sec = step / dt
                logger.info(
                    f"Step {step:5d}/{self.config.n_steps} | "
                    f"reward={stats['reward_mean']:.3f}±{stats['reward_std']:.2f} | "
                    f"loss={stats['loss']:.4f} | "
                    f"kl={stats['kl_mean']:.4f} | "
                    f"sim_ok={stats['sim_success_rate']:.0%} | "
                    f"valid={stats['sim_valid_rate']:.0%} | "
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
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--save-interval", type=int, default=500)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--n-eval-samples", type=int, default=50)
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

    # Load pre-trained model
    logger.info(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = ARCSConfig.from_dict(ckpt["config"])

    model = ARCSModel(config).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    logger.info(
        f"Loaded model (epoch {ckpt.get('epoch', '?')}, "
        f"{model.count_parameters():,} params)"
    )

    # Create frozen reference model (deep copy)
    ref_model = ARCSModel(config).to(device)
    ref_model.load_state_dict(ckpt["model_state_dict"])
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

    results = trainer.train()

    logger.info(f"\nFinal results: {json.dumps(results, indent=2, default=str)}")


if __name__ == "__main__":
    main()
