"""Smoke test for arcs.rl Phase 3 module."""

import torch
from arcs.rl import (
    COMPONENT_TO_PARAM,
    components_to_params,
    simulate_decoded_circuit,
    compute_reward,
    sample_training_specs,
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


if __name__ == "__main__":
    test_components_to_params()
    test_decode_and_simulate()
    test_sample_specs()
    test_model_generation()
    test_rl_trainer_init()
    print("\n=== ALL TESTS PASSED ===")
