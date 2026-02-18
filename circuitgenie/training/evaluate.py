"""
Comprehensive evaluation suite for CircuitGenie.

Measures:
1. Structural validity (decode success, correct params)
2. Spec-param consistency (V_in match, duty-voltage constraint)
3. SPICE simulation accuracy (V_out error vs spec, ripple)
4. Diversity (unique topologies, param spread)
"""

from typing import Dict, List, Optional

import numpy as np
import torch

from ..data.spice_templates import Topology, PARAM_NAMES, calculate_expected_vout
from ..data.simulator import run_simulation, check_ngspice
from ..model.transformer import CircuitGenieModel
from ..tokenizer.tokenizer import CircuitTokenizer
from ..training.generate import generate_circuit, validate_circuit


# Standard test specs covering all 7 topologies
TEST_SPECS = [
    # Buck (step-down)
    {'name': 'Buck 12V→5V 2A', 'specs': {'v_in': 12.0, 'v_out': 5.0, 'i_out': 2.0, 'ripple_pct': 2.0, 'efficiency': 0.92}, 'topology': Topology.BUCK},
    {'name': 'Buck 24V→12V 1A', 'specs': {'v_in': 24.0, 'v_out': 12.0, 'i_out': 1.0, 'ripple_pct': 1.5, 'efficiency': 0.93}, 'topology': Topology.BUCK},
    {'name': 'Buck 48V→3.3V 3A', 'specs': {'v_in': 48.0, 'v_out': 3.3, 'i_out': 3.0, 'ripple_pct': 3.0, 'efficiency': 0.90}, 'topology': Topology.BUCK},
    # Boost (step-up)
    {'name': 'Boost 5V→12V 0.5A', 'specs': {'v_in': 5.0, 'v_out': 12.0, 'i_out': 0.5, 'ripple_pct': 3.0, 'efficiency': 0.90}, 'topology': Topology.BOOST},
    {'name': 'Boost 12V→24V 0.3A', 'specs': {'v_in': 12.0, 'v_out': 24.0, 'i_out': 0.3, 'ripple_pct': 2.0, 'efficiency': 0.91}, 'topology': Topology.BOOST},
    # Buck-Boost (inverting)
    {'name': 'BuckBoost 12V→8V', 'specs': {'v_in': 12.0, 'v_out': 8.0, 'i_out': 1.0, 'ripple_pct': 3.0, 'efficiency': 0.88}, 'topology': Topology.BUCK_BOOST},
    # SEPIC (non-inverting buck-boost)
    {'name': 'SEPIC 12V→15V 0.5A', 'specs': {'v_in': 12.0, 'v_out': 15.0, 'i_out': 0.5, 'ripple_pct': 4.0, 'efficiency': 0.87}, 'topology': Topology.SEPIC},
    # Cuk (inverting, low ripple)
    {'name': 'Cuk 12V→10V 0.8A', 'specs': {'v_in': 12.0, 'v_out': 10.0, 'i_out': 0.8, 'ripple_pct': 2.0, 'efficiency': 0.86}, 'topology': Topology.CUK},
    # Flyback (isolated)
    {'name': 'Flyback 48V→5V 1A', 'specs': {'v_in': 48.0, 'v_out': 5.0, 'i_out': 1.0, 'ripple_pct': 5.0, 'efficiency': 0.85}, 'topology': Topology.FLYBACK},
    # QR Flyback
    {'name': 'QRFlyback 100V→12V 0.5A', 'specs': {'v_in': 100.0, 'v_out': 12.0, 'i_out': 0.5, 'ripple_pct': 3.0, 'efficiency': 0.87}, 'topology': Topology.QR_FLYBACK},
]


def evaluate_model(
    model: CircuitGenieModel,
    tokenizer: CircuitTokenizer,
    n_samples_per_spec: int = 10,
    temperature: float = 0.7,
    top_k: int = 15,
    device: str = "cpu",
    run_spice: bool = True,
) -> Dict:
    """
    Run comprehensive evaluation on a trained model.

    Args:
        model: Trained CircuitGenieModel
        tokenizer: CircuitTokenizer
        n_samples_per_spec: How many circuits to generate per test spec
        temperature: Sampling temperature
        top_k: Top-k sampling
        device: torch device
        run_spice: Whether to run ngspice validation (slower)

    Returns:
        Dict with all metrics
    """
    model.eval()
    has_spice = check_ngspice() if run_spice else False

    all_results = []
    per_topology = {}

    for test_case in TEST_SPECS:
        tc_name = test_case['name']
        specs = test_case['specs']
        topo = test_case['topology']

        tc_results = {
            'name': tc_name,
            'topology': topo.name,
            'target_v_out': specs['v_out'],
            'target_v_in': specs['v_in'],
            'decode_success': 0,
            'structural_valid': 0,
            'v_in_match': 0,
            'duty_reasonable': 0,
            'spice_success': 0,
            'v_out_errors': [],
            'generated_duties': [],
            'n_samples': n_samples_per_spec,
        }

        for _ in range(n_samples_per_spec):
            result = generate_circuit(
                model, tokenizer, specs, topo, temperature, top_k, device
            )

            if result is None:
                continue
            tc_results['decode_success'] += 1

            # Structural validation
            checks = validate_circuit(result)
            if all(checks.values()):
                tc_results['structural_valid'] += 1

            params = result['params']

            # V_in consistency: spec V_in ≈ generated V_in
            gen_v_in = params.get('V_in', 0)
            if gen_v_in > 0:
                vin_ratio = specs['v_in'] / gen_v_in
                if 0.85 < vin_ratio < 1.18:
                    tc_results['v_in_match'] += 1

            # Duty cycle reasonableness
            duty = params.get('duty', 0)
            tc_results['generated_duties'].append(duty)

            # Check if duty gives reasonable V_out
            expected_vout = calculate_expected_vout(topo, params)
            if specs['v_out'] > 0:
                vout_ratio = expected_vout / specs['v_out']
                if 0.5 < vout_ratio < 2.0:
                    tc_results['duty_reasonable'] += 1

            # SPICE validation
            if has_spice:
                try:
                    waveform = run_simulation(topo, params, timeout=10)
                    if waveform is not None:
                        tc_results['spice_success'] += 1
                        v_out_sim = float(np.mean(np.abs(waveform[-100:])))
                        v_out_error = abs(v_out_sim - specs['v_out']) / max(specs['v_out'], 0.01)
                        tc_results['v_out_errors'].append(v_out_error)
                except (KeyError, ValueError, TypeError):
                    pass  # Template formatting failed — missing params

        all_results.append(tc_results)

        if topo.name not in per_topology:
            per_topology[topo.name] = []
        per_topology[topo.name].append(tc_results)

    # Aggregate metrics
    total_gen = sum(r['n_samples'] for r in all_results)
    total_decoded = sum(r['decode_success'] for r in all_results)
    total_valid = sum(r['structural_valid'] for r in all_results)
    total_vin = sum(r['v_in_match'] for r in all_results)
    total_duty = sum(r['duty_reasonable'] for r in all_results)
    total_spice = sum(r['spice_success'] for r in all_results)
    all_vout_errors = [e for r in all_results for e in r['v_out_errors']]

    metrics = {
        'n_test_cases': len(TEST_SPECS),
        'n_samples_per_spec': n_samples_per_spec,
        'total_generated': total_gen,
        'decode_rate': total_decoded / total_gen,
        'structural_validity': total_valid / max(1, total_decoded),
        'v_in_match_rate': total_vin / max(1, total_decoded),
        'duty_reasonable_rate': total_duty / max(1, total_decoded),
        'spice_success_rate': total_spice / max(1, total_decoded) if has_spice else None,
        'mean_v_out_error': float(np.mean(all_vout_errors)) if all_vout_errors else None,
        'median_v_out_error': float(np.median(all_vout_errors)) if all_vout_errors else None,
        'per_test_case': all_results,
    }

    return metrics


def print_evaluation_report(metrics: Dict) -> None:
    """Pretty-print evaluation results."""
    print("\n" + "=" * 70)
    print("CircuitGenie Evaluation Report")
    print("=" * 70)

    print(f"\nTest cases: {metrics['n_test_cases']}, "
          f"Samples/spec: {metrics['n_samples_per_spec']}, "
          f"Total: {metrics['total_generated']}")

    print(f"\n--- Aggregate Metrics ---")
    print(f"  Decode rate:           {metrics['decode_rate']:.1%}")
    print(f"  Structural validity:   {metrics['structural_validity']:.1%}")
    print(f"  V_in match rate:       {metrics['v_in_match_rate']:.1%}")
    print(f"  Duty reasonable rate:  {metrics['duty_reasonable_rate']:.1%}")

    if metrics['spice_success_rate'] is not None:
        print(f"  SPICE success rate:    {metrics['spice_success_rate']:.1%}")
    if metrics['mean_v_out_error'] is not None:
        print(f"  Mean V_out error:      {metrics['mean_v_out_error']:.1%}")
        print(f"  Median V_out error:    {metrics['median_v_out_error']:.1%}")

    print(f"\n--- Per Test Case ---")
    for tc in metrics['per_test_case']:
        n = tc['n_samples']
        decoded = tc['decode_success']
        print(f"\n  {tc['name']}:")
        print(f"    Decoded: {decoded}/{n} | "
              f"Valid: {tc['structural_valid']}/{decoded} | "
              f"V_in OK: {tc['v_in_match']}/{decoded} | "
              f"Duty OK: {tc['duty_reasonable']}/{decoded}")

        if tc['v_out_errors']:
            mean_err = np.mean(tc['v_out_errors'])
            print(f"    SPICE: {tc['spice_success']}/{decoded} sims | "
                  f"V_out error: {mean_err:.1%}")

        if tc['generated_duties']:
            duties = tc['generated_duties']
            print(f"    Duties: mean={np.mean(duties):.3f} "
                  f"std={np.std(duties):.3f} "
                  f"range=[{np.min(duties):.3f}, {np.max(duties):.3f}]")

    print("\n" + "=" * 70)
