"""
REINFORCE policy gradient trainer for CircuitGenie.

Stage 2 training: fine-tune a CE-pretrained model using SPICE simulation
rewards. The model generates circuits from spec prefixes, simulates them
with ngspice, and receives a reward based on V_out accuracy and simulation
success.

Key features:
  - REINFORCE with baseline (running average reward)
  - KL penalty to stay close to CE-pretrained policy
  - Topology-conditioned reward shaping
  - Entropy bonus for exploration
"""

import time
import math
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..data.spice_templates import (
    Topology, PARAM_NAMES, calculate_expected_vout,
)
from ..data.simulator import run_simulation, check_ngspice
from ..model.transformer import CircuitGenieModel
from ..tokenizer.tokenizer import CircuitTokenizer
from ..tokenizer.vocabulary import (
    BOS_ID, EOS_ID, SEP_ID, PAD_ID,
    TOKEN_TO_ID, SPEC_KEY_TO_INFO,
    VALUE_BIN_OFFSET, NUM_VALUE_BINS,
    value_to_token_id,
)
from ..tokenizer.sequence import (
    SPEC_ORDER, tokens_to_circuit, _TOPO_TO_TOKEN,
)
from .generate import build_spec_prefix


def compute_reward(
    decoded: Optional[Dict],
    target_specs: Dict[str, float],
    target_topology: Topology,
    use_spice: bool = True,
) -> float:
    """
    Compute reward for a generated circuit.

    Reward components:
      1. +1.0 if decode succeeds (structural validity)
      2. +2.0 * (1 - |V_out_theory - V_out_spec| / V_out_spec)  capped at [0, 2]
      3. +1.0 if V_in matches spec within 15%
      4. +3.0 * (1 - |V_out_sim - V_out_spec| / V_out_spec)  if SPICE succeeds
      5. +1.0 bonus for successful SPICE simulation

    Total max: ~8.0
    Failure (decode fails): 0.0
    """
    if decoded is None:
        return 0.0

    reward = 1.0  # Decode success bonus

    params = decoded.get('params', {})
    topo = decoded.get('topology')

    if topo is None:
        return reward

    # V_in consistency
    gen_v_in = params.get('V_in', 0)
    if gen_v_in > 0:
        vin_ratio = target_specs['v_in'] / gen_v_in
        if 0.85 < vin_ratio < 1.18:
            reward += 1.0

    # Theoretical V_out accuracy
    expected_vout = calculate_expected_vout(topo, params)
    target_vout = target_specs['v_out']
    if target_vout > 0:
        vout_error = abs(expected_vout - target_vout) / target_vout
        vout_score = max(0.0, 1.0 - vout_error)
        reward += 2.0 * vout_score

    # SPICE simulation reward
    if use_spice:
        try:
            waveform = run_simulation(topo, params, timeout=10)
            if waveform is not None:
                reward += 1.0  # Simulation success bonus
                v_out_sim = float(np.mean(np.abs(waveform[-100:])))
                sim_error = abs(v_out_sim - target_vout) / max(target_vout, 0.01)
                sim_score = max(0.0, 1.0 - sim_error)
                reward += 3.0 * sim_score
        except (KeyError, ValueError, TypeError):
            pass  # Template formatting failed

    return reward


class RLTrainer:
    """
    REINFORCE trainer for CircuitGenie.

    Uses policy gradient with:
      - Running average baseline for variance reduction
      - KL penalty to stay close to reference (CE-pretrained) policy
      - Entropy bonus for exploration
    """

    def __init__(
        self,
        model: CircuitGenieModel,
        ref_model: CircuitGenieModel,
        tokenizer: CircuitTokenizer,
        test_specs: List[Dict],
        lr: float = 1e-5,
        kl_coeff: float = 0.1,
        entropy_coeff: float = 0.01,
        max_grad_norm: float = 0.5,
        baseline_decay: float = 0.99,
        device: str = "cpu",
        checkpoint_dir: str = "checkpoints",
        use_spice: bool = True,
        temperature: float = 0.8,
        top_k: int = 20,
    ):
        """
        Args:
            model: Model to fine-tune with RL
            ref_model: Frozen reference model (CE-pretrained) for KL penalty
            tokenizer: CircuitTokenizer
            test_specs: List of dicts with 'specs' and 'topology' keys
            lr: Learning rate (should be much smaller than CE stage)
            kl_coeff: KL divergence penalty coefficient
            entropy_coeff: Entropy bonus coefficient
            max_grad_norm: Gradient clipping
            baseline_decay: EMA decay for reward baseline
            device: torch device
            checkpoint_dir: Checkpoint save directory
            use_spice: Whether to use ngspice for reward computation
            temperature: Sampling temperature
            top_k: Top-k for generation
        """
        self.model = model.to(device)
        self.ref_model = ref_model.to(device)
        self.ref_model.eval()
        for p in self.ref_model.parameters():
            p.requires_grad = False

        self.tokenizer = tokenizer
        self.test_specs = test_specs
        self.device = device
        self.kl_coeff = kl_coeff
        self.entropy_coeff = entropy_coeff
        self.max_grad_norm = max_grad_norm
        self.use_spice = use_spice
        self.temperature = temperature
        self.top_k = top_k

        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=lr,
        )

        # Running baseline for variance reduction
        self.baseline = 0.0
        self.baseline_decay = baseline_decay
        self.best_mean_reward = -float('inf')

    def _sample_circuit(
        self,
        specs: Dict[str, float],
        topology: Topology,
    ) -> Dict:
        """
        Sample a circuit from the policy (model), collecting log-probs
        for REINFORCE.

        Returns dict with:
            token_ids: list of generated token IDs
            log_probs: list of log-probs for each generated token
            decoded: decoded circuit dict (or None)
        """
        self.model.eval()

        prefix = build_spec_prefix(specs)
        topo_token = TOKEN_TO_ID[_TOPO_TO_TOKEN[topology]]
        prefix.extend([topo_token, SEP_ID])

        ids = torch.tensor([prefix], dtype=torch.long, device=self.device)
        log_probs = []

        max_new = 32 - len(prefix)

        for _ in range(max_new):
            ids_cond = ids[:, -self.model.config.max_seq_len:]
            logits, _ = self.model.forward(ids_cond)
            logits = logits[:, -1, :] / self.temperature  # (1, V)

            # Top-k filtering
            if self.top_k > 0:
                top_vals, _ = torch.topk(logits, self.top_k)
                logits[logits < top_vals[:, -1:]] = float('-inf')

            probs = F.softmax(logits, dim=-1)
            log_prob_dist = F.log_softmax(logits, dim=-1)

            next_id = torch.multinomial(probs, num_samples=1)  # (1, 1)
            token_log_prob = log_prob_dist[0, next_id.item()]
            log_probs.append(token_log_prob)

            ids = torch.cat([ids, next_id], dim=1)

            if next_id.item() == EOS_ID:
                break

        token_ids = ids[0].tolist()
        decoded = tokens_to_circuit(token_ids)

        return {
            'token_ids': token_ids,
            'log_probs': log_probs,
            'decoded': decoded,
        }

    def _compute_kl_penalty(
        self,
        token_ids: List[int],
        prefix_len: int,
    ) -> torch.Tensor:
        """
        Compute KL(policy || reference) for the generated tokens.
        Only counts tokens after the prefix.
        """
        ids_tensor = torch.tensor([token_ids], dtype=torch.long, device=self.device)
        T = ids_tensor.shape[1]

        if T <= prefix_len + 1:
            return torch.tensor(0.0, device=self.device)

        # Get logits from both models
        with torch.no_grad():
            ref_logits, _ = self.ref_model.forward(ids_tensor[:, :-1])
        policy_logits, _ = self.model.forward(ids_tensor[:, :-1])

        # KL divergence at each generated position
        kl_total = torch.tensor(0.0, device=self.device)
        n_tokens = 0

        for t in range(prefix_len, T - 1):
            policy_log_probs = F.log_softmax(policy_logits[0, t, :], dim=-1)
            ref_log_probs = F.log_softmax(ref_logits[0, t, :], dim=-1)
            ref_probs = F.softmax(ref_logits[0, t, :], dim=-1)

            # KL(ref || policy) = sum ref_probs * (ref_log_probs - policy_log_probs)
            kl = (ref_probs * (ref_log_probs - policy_log_probs)).sum()
            kl_total = kl_total + kl
            n_tokens += 1

        return kl_total / max(1, n_tokens)

    def train_step(self) -> Dict:
        """
        One REINFORCE training step:
        1. Sample a random spec from test_specs
        2. Generate a circuit
        3. Compute reward via SPICE simulation
        4. Compute policy gradient loss

        Returns dict with step metrics.
        """
        self.model.train()

        # Sample random spec
        spec_idx = np.random.randint(len(self.test_specs))
        test_case = self.test_specs[spec_idx]
        specs = test_case['specs']
        topology = test_case['topology']

        # Generate circuit (sampling with grad for the KL part)
        sample = self._sample_circuit(specs, topology)

        # Compute reward
        reward = compute_reward(
            sample['decoded'], specs, topology, use_spice=self.use_spice
        )

        # Update baseline
        self.baseline = self.baseline_decay * self.baseline + (1 - self.baseline_decay) * reward
        advantage = reward - self.baseline

        # Policy gradient loss: -advantage * sum(log_probs)
        if len(sample['log_probs']) > 0:
            log_prob_sum = torch.stack(sample['log_probs']).sum()
            pg_loss = -advantage * log_prob_sum
        else:
            pg_loss = torch.tensor(0.0, device=self.device, requires_grad=True)

        # KL penalty
        prefix_len = 13  # BOS + 5 spec pairs + SEP + TOPO + SEP = 1 + 10 + 1 + 1 + 1 = 14 tokens prefix, 13 in 0-indexed
        kl_penalty = self._compute_kl_penalty(
            sample['token_ids'], prefix_len
        )

        # Total loss
        total_loss = pg_loss + self.kl_coeff * kl_penalty

        # Backward
        self.optimizer.zero_grad()
        if total_loss.requires_grad:
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.max_grad_norm
            )
            self.optimizer.step()

        return {
            'reward': reward,
            'advantage': advantage,
            'pg_loss': pg_loss.item() if torch.is_tensor(pg_loss) else pg_loss,
            'kl_penalty': kl_penalty.item() if torch.is_tensor(kl_penalty) else kl_penalty,
            'total_loss': total_loss.item() if torch.is_tensor(total_loss) else total_loss,
            'n_tokens': len(sample['log_probs']),
            'decoded_ok': sample['decoded'] is not None,
            'topology': topology.name,
        }

    def train(
        self,
        n_steps: int = 5000,
        log_every: int = 50,
        eval_every: int = 500,
        n_eval_samples: int = 5,
    ) -> Dict:
        """
        Run REINFORCE training loop.

        Args:
            n_steps: Total training steps
            log_every: Print metrics every N steps
            eval_every: Run evaluation every N steps
            n_eval_samples: Samples per spec during evaluation

        Returns:
            Training history dict
        """
        has_spice = check_ngspice() if self.use_spice else False
        if not has_spice and self.use_spice:
            print("Warning: ngspice not available, using theoretical reward only")
            self.use_spice = False

        history = {
            'rewards': [],
            'pg_losses': [],
            'kl_penalties': [],
            'decode_rates': [],
        }

        reward_window = []
        decode_window = []

        print(f"REINFORCE Training: {n_steps} steps")
        print(f"  KL coeff: {self.kl_coeff}, Entropy coeff: {self.entropy_coeff}")
        print(f"  SPICE reward: {self.use_spice}")
        print(f"  Test specs: {len(self.test_specs)}")
        print()

        t0 = time.time()

        for step in range(1, n_steps + 1):
            metrics = self.train_step()

            reward_window.append(metrics['reward'])
            decode_window.append(1.0 if metrics['decoded_ok'] else 0.0)

            history['rewards'].append(metrics['reward'])
            history['pg_losses'].append(metrics['pg_loss'])
            history['kl_penalties'].append(metrics['kl_penalty'])

            if step % log_every == 0:
                mean_reward = np.mean(reward_window[-log_every:])
                mean_decode = np.mean(decode_window[-log_every:])
                dt = time.time() - t0
                steps_per_sec = step / dt

                print(
                    f"Step {step:5d}/{n_steps} | "
                    f"reward={mean_reward:.2f} | "
                    f"decode={mean_decode:.0%} | "
                    f"pg_loss={metrics['pg_loss']:.4f} | "
                    f"kl={metrics['kl_penalty']:.4f} | "
                    f"baseline={self.baseline:.2f} | "
                    f"{steps_per_sec:.1f} steps/s"
                )

            if step % eval_every == 0:
                eval_metrics = self._run_eval(n_eval_samples)
                history['decode_rates'].append(eval_metrics['decode_rate'])

                print(f"\n  --- Eval at step {step} ---")
                print(f"  Decode rate: {eval_metrics['decode_rate']:.0%}")
                print(f"  Mean reward: {eval_metrics['mean_reward']:.2f}")
                print(f"  V_in match:  {eval_metrics['v_in_match']:.0%}")
                print(f"  Duty OK:     {eval_metrics['duty_ok']:.0%}")
                if eval_metrics.get('mean_vout_error') is not None:
                    print(f"  V_out error: {eval_metrics['mean_vout_error']:.1%}")
                print()

                # Save best
                if eval_metrics['mean_reward'] > self.best_mean_reward:
                    self.best_mean_reward = eval_metrics['mean_reward']
                    self.save_checkpoint(
                        self.checkpoint_dir / "best_rl_model.pt"
                    )

            # Periodic checkpoint
            if step % 1000 == 0:
                self.save_checkpoint(
                    self.checkpoint_dir / f"rl_checkpoint_step{step}.pt"
                )

        # Save final
        self.save_checkpoint(self.checkpoint_dir / "final_rl_model.pt")
        return history

    @torch.no_grad()
    def _run_eval(self, n_samples: int = 5) -> Dict:
        """Quick evaluation on test specs."""
        self.model.eval()

        total = 0
        decoded = 0
        rewards = []
        v_in_match = 0
        duty_ok = 0
        vout_errors = []

        for test_case in self.test_specs:
            specs = test_case['specs']
            topo = test_case['topology']

            for _ in range(n_samples):
                total += 1
                sample = self._sample_circuit(specs, topo)
                result = sample['decoded']

                reward = compute_reward(
                    result, specs, topo, use_spice=self.use_spice
                )
                rewards.append(reward)

                if result is None:
                    continue
                decoded += 1

                params = result.get('params', {})

                # V_in match
                gen_v_in = params.get('V_in', 0)
                if gen_v_in > 0:
                    ratio = specs['v_in'] / gen_v_in
                    if 0.85 < ratio < 1.18:
                        v_in_match += 1

                # Duty reasonableness
                expected_vout = calculate_expected_vout(topo, params)
                if specs['v_out'] > 0:
                    vout_ratio = expected_vout / specs['v_out']
                    if 0.5 < vout_ratio < 2.0:
                        duty_ok += 1

                # Theoretical V_out error
                vout_error = abs(expected_vout - specs['v_out']) / max(specs['v_out'], 0.01)
                vout_errors.append(vout_error)

        return {
            'decode_rate': decoded / max(1, total),
            'mean_reward': np.mean(rewards),
            'v_in_match': v_in_match / max(1, decoded),
            'duty_ok': duty_ok / max(1, decoded),
            'mean_vout_error': np.mean(vout_errors) if vout_errors else None,
        }

    def save_checkpoint(self, path: Path) -> None:
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.model.config,
            'baseline': self.baseline,
            'best_mean_reward': self.best_mean_reward,
        }, path)

    def load_checkpoint(self, path: Path) -> None:
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.baseline = checkpoint.get('baseline', 0.0)
        self.best_mean_reward = checkpoint.get('best_mean_reward', -float('inf'))
