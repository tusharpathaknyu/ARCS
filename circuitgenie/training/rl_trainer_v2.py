"""
REINFORCE policy gradient trainer for CircuitGenie v2 (Eulerian walk).

Stage 2 training: fine-tune a CE-pretrained model using SPICE simulation
rewards. The model generates circuits from spec prefixes (no topology hint),
producing Eulerian walks that encode both topology and wiring.

Key differences from v1 RL trainer:
  - Uses v2 tokenizer (Eulerian walk representation)
  - No topology token in prefix — model must infer topology from walk
  - Longer sequences (up to 64 tokens)
  - Topology identified from generated walk structure
"""

import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

from ..data.spice_templates import (
    Topology, calculate_expected_vout,
)
from ..data.simulator import run_simulation, check_ngspice
from ..model.transformer import CircuitGenieModel
from ..tokenizer.vocabulary_v2 import (
    BOS_ID, EOS_ID, SEP_ID, PAD_ID, WALK_END_ID,
    TOKEN_TO_ID_V2, SPEC_KEY_TO_INFO_V2,
    value_to_token_id_v2,
)
from ..tokenizer.sequence_v2 import (
    SPEC_ORDER, tokens_to_circuit_v2, _identify_topology_from_walk,
)


def build_spec_prefix_v2(specs: Dict[str, float]) -> List[int]:
    """
    Build the spec prefix token sequence for v2.

    Returns: [BOS, SPEC_V_IN, VAL_xx, SPEC_V_OUT, VAL_xx, ..., SEP]
    (12 tokens: 1 BOS + 5 spec pairs + 1 SEP)
    """
    tokens = [BOS_ID]
    for spec_key in SPEC_ORDER:
        spec_token_name, qtype = SPEC_KEY_TO_INFO_V2[spec_key]
        spec_token_id = TOKEN_TO_ID_V2[spec_token_name]
        val_token_id = value_to_token_id_v2(specs[spec_key], qtype)
        tokens.extend([spec_token_id, val_token_id])
    tokens.append(SEP_ID)
    return tokens


def compute_reward_v2(
    decoded: Optional[Dict],
    target_specs: Dict[str, float],
    use_spice: bool = True,
) -> float:
    """
    Compute reward for a generated v2 circuit.

    Reward components:
      1. +1.0 if decode succeeds (structural validity)
      2. +2.0 * (1 - |V_out_theory - V_out_spec| / V_out_spec)  capped at [0, 2]
      3. +1.0 if V_in matches spec within 15%
      4. +3.0 * (1 - |V_out_sim - V_out_spec| / V_out_spec)  if SPICE succeeds
      5. +1.0 bonus for successful SPICE simulation

    Total max: ~8.0
    """
    if decoded is None:
        return 0.0

    reward = 1.0  # Decode success bonus

    params = decoded.get('params', {})
    topo = decoded.get('topology')

    if topo is None:
        return reward

    # Refine topology from walk structure
    walk_tokens = decoded.get('walk_tokens', [])
    if walk_tokens:
        refined = _identify_topology_from_walk(walk_tokens)
        if refined is not None:
            topo = refined
            decoded['topology'] = topo

    # V_in consistency
    gen_v_in = params.get('V_in', 0)
    if gen_v_in > 0:
        vin_ratio = target_specs['v_in'] / gen_v_in
        if 0.85 < vin_ratio < 1.18:
            reward += 1.0

    # Theoretical V_out accuracy
    try:
        expected_vout = calculate_expected_vout(topo, params)
        target_vout = target_specs['v_out']
        if target_vout > 0:
            vout_error = abs(expected_vout - target_vout) / target_vout
            vout_score = max(0.0, 1.0 - vout_error)
            reward += 2.0 * vout_score
    except Exception:
        pass

    # SPICE simulation reward
    if use_spice:
        try:
            waveform = run_simulation(topo, params, timeout=10)
            if waveform is not None:
                reward += 1.0  # Simulation success bonus
                target_vout = target_specs['v_out']
                v_out_sim = float(np.mean(np.abs(waveform[-100:])))
                sim_error = abs(v_out_sim - target_vout) / max(target_vout, 0.01)
                sim_score = max(0.0, 1.0 - sim_error)
                reward += 3.0 * sim_score
        except (KeyError, ValueError, TypeError):
            pass

    return reward


class RLTrainerV2:
    """
    REINFORCE trainer for CircuitGenie v2 (Eulerian walk).

    Uses policy gradient with:
      - Running average baseline for variance reduction
      - KL penalty to stay close to reference (CE-pretrained) policy
      - No topology hint — model must generate both walk and values
    """

    def __init__(
        self,
        model: CircuitGenieModel,
        ref_model: CircuitGenieModel,
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
        self.model = model.to(device)
        self.ref_model = ref_model.to(device)
        self.ref_model.eval()
        for p in self.ref_model.parameters():
            p.requires_grad = False

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

        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        # Running baseline for variance reduction
        self.baseline = 0.0
        self.baseline_decay = baseline_decay
        self.best_mean_reward = -float('inf')

    def _sample_circuit(self, specs: Dict[str, float]) -> Dict:
        """
        Sample a circuit from the policy (model), collecting log-probs.

        For v2, prefix is just [BOS, specs..., SEP] (12 tokens).
        Model generates the full Eulerian walk + value block.
        """
        self.model.eval()

        prefix = build_spec_prefix_v2(specs)
        ids = torch.tensor([prefix], dtype=torch.long, device=self.device)
        log_probs = []

        max_new = 64 - len(prefix)  # v2 uses 64 max seq len

        for _ in range(max_new):
            ids_cond = ids[:, -self.model.config.max_seq_len:]
            logits, _ = self.model.forward(ids_cond)
            logits = logits[:, -1, :] / self.temperature

            # Top-k filtering
            if self.top_k > 0:
                top_vals, _ = torch.topk(logits, self.top_k)
                logits[logits < top_vals[:, -1:]] = float('-inf')

            probs = F.softmax(logits, dim=-1)
            log_prob_dist = F.log_softmax(logits, dim=-1)

            next_id = torch.multinomial(probs, num_samples=1)
            token_log_prob = log_prob_dist[0, next_id.item()]
            log_probs.append(token_log_prob)

            ids = torch.cat([ids, next_id], dim=1)

            if next_id.item() == EOS_ID:
                break

        token_ids = ids[0].tolist()
        decoded = tokens_to_circuit_v2(token_ids)

        # Refine topology from walk if possible
        if decoded is not None:
            walk_tokens = decoded.get('walk_tokens', [])
            if walk_tokens:
                refined = _identify_topology_from_walk(walk_tokens)
                if refined is not None:
                    decoded['topology'] = refined

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
        """Compute KL(policy || reference) for generated tokens."""
        ids_tensor = torch.tensor(
            [token_ids], dtype=torch.long, device=self.device
        )
        T = ids_tensor.shape[1]

        if T <= prefix_len + 1:
            return torch.tensor(0.0, device=self.device)

        with torch.no_grad():
            ref_logits, _ = self.ref_model.forward(ids_tensor[:, :-1])
        policy_logits, _ = self.model.forward(ids_tensor[:, :-1])

        kl_total = torch.tensor(0.0, device=self.device)
        n_tokens = 0

        for t in range(prefix_len, T - 1):
            policy_log_probs = F.log_softmax(policy_logits[0, t, :], dim=-1)
            ref_log_probs = F.log_softmax(ref_logits[0, t, :], dim=-1)
            ref_probs = F.softmax(ref_logits[0, t, :], dim=-1)

            kl = (ref_probs * (ref_log_probs - policy_log_probs)).sum()
            kl_total = kl_total + kl
            n_tokens += 1

        return kl_total / max(1, n_tokens)

    def train_step(self) -> Dict:
        """One REINFORCE training step."""
        self.model.train()

        # Sample random spec
        spec_idx = np.random.randint(len(self.test_specs))
        test_case = self.test_specs[spec_idx]
        specs = test_case['specs']

        # Generate circuit (no topology hint for v2)
        sample = self._sample_circuit(specs)

        # Compute reward
        reward = compute_reward_v2(
            sample['decoded'], specs, use_spice=self.use_spice
        )

        # Update baseline
        self.baseline = (
            self.baseline_decay * self.baseline
            + (1 - self.baseline_decay) * reward
        )
        advantage = reward - self.baseline

        # Policy gradient loss
        if len(sample['log_probs']) > 0:
            log_prob_sum = torch.stack(sample['log_probs']).sum()
            pg_loss = -advantage * log_prob_sum
        else:
            pg_loss = torch.tensor(
                0.0, device=self.device, requires_grad=True
            )

        # KL penalty (prefix is 12 tokens: BOS + 5 pairs + SEP)
        prefix_len = 12
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
            'pg_loss': (
                pg_loss.item() if torch.is_tensor(pg_loss) else pg_loss
            ),
            'kl_penalty': (
                kl_penalty.item()
                if torch.is_tensor(kl_penalty)
                else kl_penalty
            ),
            'total_loss': (
                total_loss.item()
                if torch.is_tensor(total_loss)
                else total_loss
            ),
            'n_tokens': len(sample['log_probs']),
            'decoded_ok': sample['decoded'] is not None,
        }

    def train(
        self,
        n_steps: int = 5000,
        log_every: int = 50,
        eval_every: int = 500,
        n_eval_samples: int = 5,
    ) -> Dict:
        """Run REINFORCE training loop."""
        has_spice = check_ngspice() if self.use_spice else False
        if not has_spice and self.use_spice:
            print(
                "Warning: ngspice not available, using theoretical reward only"
            )
            self.use_spice = False

        history = {
            'rewards': [],
            'pg_losses': [],
            'kl_penalties': [],
            'decode_rates': [],
        }

        reward_window = []
        decode_window = []

        print(f"REINFORCE v2 Training: {n_steps} steps")
        print(f"  KL coeff: {self.kl_coeff}")
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
                    print(
                        f"  V_out error: {eval_metrics['mean_vout_error']:.1%}"
                    )
                if eval_metrics.get('spice_success') is not None:
                    print(
                        f"  SPICE rate:  {eval_metrics['spice_success']:.0%}"
                    )
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
        spice_ok = 0

        for test_case in self.test_specs:
            specs = test_case['specs']

            for _ in range(n_samples):
                total += 1
                sample = self._sample_circuit(specs)
                result = sample['decoded']

                reward = compute_reward_v2(
                    result, specs, use_spice=self.use_spice
                )
                rewards.append(reward)

                if result is None:
                    continue
                decoded += 1

                params = result.get('params', {})
                topo = result.get('topology')

                if topo is None:
                    continue

                # V_in match
                gen_v_in = params.get('V_in', 0)
                if gen_v_in > 0:
                    ratio = specs['v_in'] / gen_v_in
                    if 0.85 < ratio < 1.18:
                        v_in_match += 1

                # Theoretical V_out
                try:
                    expected_vout = calculate_expected_vout(topo, params)
                    if specs['v_out'] > 0:
                        vout_ratio = expected_vout / specs['v_out']
                        if 0.5 < vout_ratio < 2.0:
                            duty_ok += 1

                    vout_error = abs(
                        expected_vout - specs['v_out']
                    ) / max(specs['v_out'], 0.01)
                    vout_errors.append(vout_error)
                except Exception:
                    pass

                # SPICE check
                if self.use_spice:
                    try:
                        waveform = run_simulation(topo, params, timeout=10)
                        if waveform is not None:
                            spice_ok += 1
                    except Exception:
                        pass

        return {
            'decode_rate': decoded / max(1, total),
            'mean_reward': np.mean(rewards),
            'v_in_match': v_in_match / max(1, decoded),
            'duty_ok': duty_ok / max(1, decoded),
            'mean_vout_error': np.mean(vout_errors) if vout_errors else None,
            'spice_success': spice_ok / max(1, decoded) if self.use_spice else None,
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
        checkpoint = torch.load(
            path, map_location=self.device, weights_only=False
        )
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.baseline = checkpoint.get('baseline', 0.0)
        self.best_mean_reward = checkpoint.get(
            'best_mean_reward', -float('inf')
        )
