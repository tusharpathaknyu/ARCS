"""Smoke test for arcs.rl Phase 3 module."""

import torch
from arcs.rl import (
    COMPONENT_TO_PARAM,
    components_to_params,
    simulate_decoded_circuit,
    compute_reward,
    sample_training_specs,
    sample_specs_for_topology,
    sample_with_logprobs,
    RLConfig,
    ARCSRLTrainer,
)
from arcs.model import ARCSModel, ARCSConfig
from arcs.tokenizer import CircuitTokenizer
from arcs.evaluate import decode_generated_sequence


def test_components_to_params():
    print("=== Test 1: components_to_params ===")

    # Buck: 4 components → 4 params
    comps = [("INDUCTOR", 100e-6), ("CAPACITOR", 47e-6), ("RESISTOR", 0.01), ("MOSFET_N", 0.05)]
    params = components_to_params("buck", comps)
    print(f"  buck: {params}")
    assert params is not None
    assert "inductance" in params
    assert "capacitance" in params
    assert abs(params["inductance"] - 100e-6) < 1e-10

    # Cuk: 6 components → 6 params
    comps2 = [("INDUCTOR", 50e-6), ("INDUCTOR", 30e-6), ("CAPACITOR", 1e-6),
              ("CAPACITOR", 47e-6), ("RESISTOR", 0.01), ("MOSFET_N", 0.03)]
    params2 = components_to_params("cuk", comps2)
    print(f"  cuk: {params2}")
    assert params2 is not None
    assert "inductance_1" in params2
    assert "inductance_2" in params2
    assert "cap_coupling" in params2

    # Flyback: 5 components → 5 params
    comps3 = [("INDUCTOR", 500e-6), ("TRANSFORMER", 2.0), ("CAPACITOR", 100e-6),
              ("RESISTOR", 0.02), ("MOSFET_N", 0.04)]
    params3 = components_to_params("flyback", comps3)
    print(f"  flyback: {params3}")
    assert params3 is not None
    assert "turns_ratio" in params3
    assert abs(params3["turns_ratio"] - 2.0) < 1e-10

    print("  PASSED")


def test_decode_and_simulate():
    print("\n=== Test 2: Decode & Simulate ===")
    tokenizer = CircuitTokenizer()

    # Build a token sequence for a known-good buck circuit
    token_ids = [
        tokenizer.name_to_id["START"],
        tokenizer.name_to_id["TOPO_BUCK"],
        tokenizer.sep_id,
        tokenizer.name_to_id["SPEC_VIN"], tokenizer.encode_value(12.0),
        tokenizer.name_to_id["SPEC_VOUT"], tokenizer.encode_value(5.0),
        tokenizer.name_to_id["SPEC_IOUT"], tokenizer.encode_value(1.0),
        tokenizer.name_to_id["SPEC_FSW"], tokenizer.encode_value(100000),
        tokenizer.sep_id,
        tokenizer.name_to_id["COMP_INDUCTOR"], tokenizer.encode_value(100e-6),
        tokenizer.name_to_id["COMP_CAPACITOR"], tokenizer.encode_value(47e-6),
        tokenizer.name_to_id["COMP_RESISTOR"], tokenizer.encode_value(0.01),
        tokenizer.name_to_id["COMP_MOSFET_N"], tokenizer.encode_value(0.05),
        tokenizer.name_to_id["END"],
    ]

    decoded = decode_generated_sequence(token_ids, tokenizer)
    print(f"  Topology: {decoded.topology}")
    print(f"  Specs: {decoded.specs}")
    print(f"  Components: {decoded.components}")
    print(f"  Valid structure: {decoded.valid_structure}")
    assert decoded.topology == "buck"
    assert decoded.valid_structure

    # Simulate
    outcome = simulate_decoded_circuit(decoded)
    print(f"  Sim success: {outcome.success}")
    if outcome.success:
        eff = outcome.metrics.get("efficiency", 0)
        verr = outcome.metrics.get("vout_error_pct", 100)
        rip = outcome.metrics.get("ripple_ratio", 1)
        print(f"  eff={eff:.3f}, vout_err={verr:.1f}%, ripple={rip:.3f}")
        print(f"  Valid: {outcome.valid}")
    else:
        print(f"  Error: {outcome.error}")

    # Compute reward
    reward = compute_reward(decoded, outcome)
    print(f"  Reward: {reward:.2f} / 8.0")
    assert reward >= 1.0  # At least structure reward
    print("  PASSED")


def test_sample_specs():
    print("\n=== Test 3: sample_training_specs ===")
    tokenizer = CircuitTokenizer()
    device = torch.device("cpu")

    topo, specs, prefix = sample_training_specs(tokenizer, device)
    print(f"  Topo: {topo}")
    print(f"  Specs: {specs}")
    print(f"  Prefix shape: {prefix.shape}")
    assert prefix.ndim == 2
    assert prefix.size(0) == 1
    assert prefix.size(1) >= 5  # START + TOPO + SEP + at least some specs
    print("  PASSED")


def test_model_generation():
    print("\n=== Test 4: Model generation with logprobs ===")
    tokenizer = CircuitTokenizer()
    device = torch.device("cpu")
    config = ARCSConfig.small()
    model = ARCSModel(config).to(device)

    topo, specs, prefix = sample_training_specs(tokenizer, device)
    gen_tokens, log_probs, entropy = sample_with_logprobs(
        model, prefix, tokenizer, max_new_tokens=30, temperature=0.8, top_k=50
    )
    print(f"  Generated {len(gen_tokens)} tokens")
    print(f"  Log-probs shape: {log_probs.shape}")
    print(f"  Mean entropy: {entropy.mean().item():.3f}")
    assert len(gen_tokens) > 0
    assert len(gen_tokens) == len(log_probs)
    print("  PASSED")


def test_rl_trainer_init():
    print("\n=== Test 5: ARCSRLTrainer init ===")
    tokenizer = CircuitTokenizer()
    device = torch.device("cpu")
    config = ARCSConfig.small()

    model = ARCSModel(config).to(device)
    ref_model = ARCSModel(config).to(device)
    ref_model.load_state_dict(model.state_dict())

    rl_config = RLConfig(n_steps=2, batch_size=1, log_interval=1, eval_interval=1, n_eval_samples=2)
    trainer = ARCSRLTrainer(
        model=model, ref_model=ref_model, tokenizer=tokenizer,
        config=rl_config, device=device, output_dir="/tmp/arcs_rl_test"
    )
    print(f"  Trainer initialized")
    print(f"  Baseline: {trainer.baseline}")

    # Run 1 training step
    print("  Running 1 train step (this will simulate circuits)...")
    stats = trainer.train_step()
    print(f"  Step stats: reward={stats['reward_mean']:.3f}, loss={stats['loss']:.4f}")
    print("  PASSED")


# -----------------------------------------------------------------------
# GRPO tests
# -----------------------------------------------------------------------

def test_sample_specs_for_topology():
    """sample_specs_for_topology produces prefix for a given topology."""
    tokenizer = CircuitTokenizer()
    device = torch.device("cpu")

    topo, specs, prefix = sample_specs_for_topology("buck", tokenizer, device)
    assert topo == "buck"
    assert "vin" in specs and "vout" in specs
    assert prefix.ndim == 2 and prefix.size(0) == 1

    # Specs should be perturbed from base ±20%
    from arcs.templates import OPERATING_CONDITIONS
    base_vin = OPERATING_CONDITIONS["buck"]["vin"]
    assert 0.8 * base_vin <= specs["vin"] <= 1.2 * base_vin


def test_grpo_config():
    """RLConfig with GRPO fields."""
    cfg = RLConfig(grpo=True, group_size=6, n_topos_per_step=2)
    assert cfg.grpo is True
    assert cfg.group_size == 6
    assert cfg.n_topos_per_step == 2
    assert cfg.grpo_clip_adv == 5.0
    assert cfg.grpo_eps == 1e-4


def test_grpo_trainer_step():
    """ARCSRLTrainer.train_step_grpo runs without error."""
    tokenizer = CircuitTokenizer()
    device = torch.device("cpu")
    config = ARCSConfig.small()

    model = ARCSModel(config).to(device)
    ref_model = ARCSModel(config).to(device)
    ref_model.load_state_dict(model.state_dict())

    rl_config = RLConfig(
        grpo=True,
        group_size=2,           # small for speed
        n_topos_per_step=2,     # 2 topos × 2 per group = 4 episodes
        n_steps=1,
        log_interval=1,
        eval_interval=1,
        n_eval_samples=2,
    )
    trainer = ARCSRLTrainer(
        model=model, ref_model=ref_model, tokenizer=tokenizer,
        config=rl_config, device=device, output_dir="/tmp/arcs_grpo_test",
    )

    stats = trainer.train_step_grpo()
    assert "reward_mean" in stats
    assert "n_topologies" in stats
    assert stats["n_topologies"] == 2
    assert isinstance(stats["loss"], float)
    print(f"  GRPO step: reward={stats['reward_mean']:.3f}, "
          f"topos={stats['n_topologies']}, loss={stats['loss']:.4f}")


def test_grpo_advantage_normalization():
    """Verify group-relative advantage z-scoring."""
    import numpy as np
    # Simulate a group of rewards and check advantage computation
    rewards = [1.0, 3.0, 2.0, 4.0]
    grp = np.array(rewards)
    grp_mean = grp.mean()  # 2.5
    grp_std = grp.std()    # ~1.118
    eps = 1e-4

    advantages = [(r - grp_mean) / (grp_std + eps) for r in rewards]
    # Should be z-scored: mean ≈ 0, std ≈ 1
    assert abs(np.mean(advantages)) < 0.01
    assert abs(np.std(advantages) - 1.0) < 0.01

    # Identical rewards → all advantages ≈ 0 (protected by eps)
    uniform = [2.0, 2.0, 2.0, 2.0]
    grp_u = np.array(uniform)
    advs_u = [(r - grp_u.mean()) / (grp_u.std() + eps) for r in uniform]
    assert all(abs(a) < 0.01 for a in advs_u)


def test_grpo_train_dispatches():
    """train() dispatches to GRPO when config.grpo is True."""
    tokenizer = CircuitTokenizer()
    device = torch.device("cpu")
    config = ARCSConfig.small()

    model = ARCSModel(config).to(device)
    ref_model = ARCSModel(config).to(device)
    ref_model.load_state_dict(model.state_dict())

    rl_config = RLConfig(
        grpo=True,
        group_size=1,
        n_topos_per_step=1,
        n_steps=1,
        log_interval=1,
        eval_interval=1,
        n_eval_samples=1,
        save_interval=9999,
    )
    trainer = ARCSRLTrainer(
        model=model, ref_model=ref_model, tokenizer=tokenizer,
        config=rl_config, device=device, output_dir="/tmp/arcs_grpo_dispatch",
    )

    results = trainer.train()
    assert "best_reward" in results
    print(f"  GRPO train dispatch OK: best_reward={results['best_reward']:.3f}")


if __name__ == "__main__":
    test_components_to_params()
    test_decode_and_simulate()
    test_sample_specs()
    test_model_generation()
    test_rl_trainer_init()
    test_sample_specs_for_topology()
    test_grpo_config()
    test_grpo_trainer_step()
    test_grpo_advantage_normalization()
    test_grpo_train_dispatches()
    print("\n=== ALL TESTS PASSED ===")
