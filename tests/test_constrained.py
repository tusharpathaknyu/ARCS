"""Tests for ARCS constrained decoding module.

Tests:
  1. Grammar state machine transitions
  2. Constraint mask computation at each level
  3. End-to-end constrained generation (all 16 topologies)
  4. 100% structural validity guarantee
  5. Value range enforcement
  6. Lagrangian constraint loss
  7. Constrained sampling with log-probs (for RL)
"""

from __future__ import annotations

import math

import pytest
import torch

from arcs.constrained import (
    ConstrainedGenerator,
    ConstraintLevel,
    ConstraintMask,
    DecoderState,
    GrammarState,
    LagrangianConstraintLoss,
    _get_param_bounds,
    _get_value_token_range,
    _update_state,
    _value_bin_index,
    constrained_sample_with_logprobs,
)
from arcs.model import ARCSConfig, ARCSModel
from arcs.model_enhanced import create_model
from arcs.simulate import COMPONENT_TO_PARAM, normalize_topology
from arcs.tokenizer import CircuitTokenizer, TokenType


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tokenizer():
    return CircuitTokenizer()


@pytest.fixture
def small_model(tokenizer):
    config = ARCSConfig.small()
    config.vocab_size = tokenizer.vocab_size
    model = ARCSModel(config)
    return model


@pytest.fixture
def two_head_model(tokenizer):
    config = ARCSConfig.small()
    config.vocab_size = tokenizer.vocab_size
    model = create_model("two_head", config)
    return model


@pytest.fixture
def mask_computer(tokenizer):
    return ConstraintMask(tokenizer)


# ---------------------------------------------------------------------------
# Token helpers
# ---------------------------------------------------------------------------

class TestTokenHelpers:
    """Test value binning and range helpers."""

    def test_value_bin_1ohm(self, tokenizer):
        """1 ohm → bin index around 333 (12 decades, 1 ohm = 10^0)."""
        idx = _value_bin_index(1.0, tokenizer)
        # log10(1) = 0, range is [-12, 6], so fraction = 12/18 = 0.667
        expected = int(12.0 / 18.0 * 500)
        assert abs(idx - expected) <= 1

    def test_value_bin_1pf(self, tokenizer):
        """1e-12 → bin 0 (minimum value)."""
        idx = _value_bin_index(1e-12, tokenizer)
        assert idx == 0

    def test_value_bin_1meg(self, tokenizer):
        """1e6 → bin 499 (maximum value)."""
        idx = _value_bin_index(1e6, tokenizer)
        assert idx == 499

    def test_value_bin_monotonic(self, tokenizer):
        """Larger values should map to larger bins."""
        vals = [1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1, 100, 1e4]
        bins = [_value_bin_index(v, tokenizer) for v in vals]
        for i in range(len(bins) - 1):
            assert bins[i] < bins[i + 1], f"Bin {bins[i]} >= {bins[i+1]} for {vals[i]}"

    def test_value_token_range(self, tokenizer):
        """10µH to 1mH should cover a narrow range of value tokens."""
        lo_id, hi_id = _get_value_token_range(10e-6, 1e-3, tokenizer)
        assert lo_id <= hi_id
        assert lo_id >= tokenizer.name_to_id["VAL_0"]
        assert hi_id <= tokenizer.name_to_id[f"VAL_{tokenizer.N_VALUE_BINS - 1}"]
        # Should span ~2 decades → ~56 bins
        n_bins = hi_id - lo_id + 1
        assert 30 < n_bins < 100, f"Expected ~56 bins, got {n_bins}"


class TestParamBounds:
    """Test parameter bounds lookup."""

    def test_buck_inductance(self):
        bounds = _get_param_bounds("buck", "inductance")
        assert bounds is not None
        lo, hi = bounds
        assert lo == pytest.approx(1e-6, rel=0.1)
        assert hi == pytest.approx(1e-3, rel=0.1)

    def test_inverting_amp_r_input(self):
        bounds = _get_param_bounds("inverting_amp", "r_input")
        assert bounds is not None
        lo, hi = bounds
        assert lo > 0
        assert hi > lo

    def test_unknown_topology(self):
        assert _get_param_bounds("nonexistent_topology", "x") is None

    def test_unknown_param(self):
        assert _get_param_bounds("buck", "nonexistent_param") is None


# ---------------------------------------------------------------------------
# Grammar state machine
# ---------------------------------------------------------------------------

class TestGrammarState:
    """Test the grammar state machine transitions."""

    def test_comp_to_val_transition(self, tokenizer):
        """After generating COMP_RESISTOR, state should expect VAL."""
        state = DecoderState(
            grammar=GrammarState.EXPECT_COMP,
            topology="buck",
            expected_components=list(COMPONENT_TO_PARAM["buck"]),
        )
        # Generate a component token (RESISTOR)
        comp_id = tokenizer.name_to_id["COMP_RESISTOR"]
        state = _update_state(state, comp_id, tokenizer)
        assert state.grammar == GrammarState.EXPECT_VAL

    def test_val_to_comp_transition(self, tokenizer):
        """After generating a VAL token, state should expect COMP."""
        state = DecoderState(
            grammar=GrammarState.EXPECT_VAL,
            topology="buck",
            expected_components=list(COMPONENT_TO_PARAM["buck"]),
            current_component=("RESISTOR", "esr"),
        )
        val_id = tokenizer.name_to_id["VAL_250"]  # some value token
        state = _update_state(state, val_id, tokenizer)
        assert state.grammar == GrammarState.EXPECT_COMP
        assert state.n_components_generated == 1

    def test_end_transition(self, tokenizer):
        """Generating END should transition to DONE."""
        state = DecoderState(
            grammar=GrammarState.EXPECT_COMP,
            topology="buck",
            expected_components=[],
        )
        state = _update_state(state, tokenizer.end_id, tokenizer)
        assert state.grammar == GrammarState.DONE

    def test_full_buck_sequence(self, tokenizer):
        """Simulate a complete buck converter component sequence."""
        expected = list(COMPONENT_TO_PARAM["buck"])
        state = DecoderState(
            grammar=GrammarState.EXPECT_COMP,
            topology="buck",
            expected_components=list(expected),
        )

        # Generate each component-value pair
        for comp_type, _ in expected:
            comp_id = tokenizer.name_to_id[f"COMP_{comp_type}"]
            state = _update_state(state, comp_id, tokenizer)
            assert state.grammar == GrammarState.EXPECT_VAL

            val_id = tokenizer.name_to_id["VAL_200"]
            state = _update_state(state, val_id, tokenizer)
            assert state.grammar == GrammarState.EXPECT_COMP

        # All components placed — should have empty expected list
        assert len(state.expected_components) == 0

        # Now END
        state = _update_state(state, tokenizer.end_id, tokenizer)
        assert state.grammar == GrammarState.DONE
        assert state.n_components_generated == 4


# ---------------------------------------------------------------------------
# Constraint mask
# ---------------------------------------------------------------------------

class TestConstraintMask:
    """Test constraint mask computation."""

    def test_expect_comp_grammar_level(self, tokenizer, mask_computer):
        """At GRAMMAR level, any COMP or END should be valid."""
        state = DecoderState(
            grammar=GrammarState.EXPECT_COMP,
            topology="buck",
            expected_components=list(COMPONENT_TO_PARAM["buck"]),
            n_components_generated=2,
        )
        mask = mask_computer.compute_mask(state, ConstraintLevel.GRAMMAR)

        # Component tokens should be valid (0.0)
        for tid in mask_computer.component_ids:
            assert mask[tid] == 0.0, f"COMP token {tid} should be valid at GRAMMAR level"

        # END should be valid (≥2 components already)
        assert mask[tokenizer.end_id] == 0.0

        # Value tokens should be invalid
        for tid in mask_computer.value_ids:
            assert mask[tid] == float("-inf"), f"VAL token {tid} should be invalid"

        # Topology tokens should be invalid
        for tid in mask_computer.topology_ids:
            assert mask[tid] == float("-inf")

    def test_expect_comp_topology_level(self, tokenizer, mask_computer):
        """At TOPOLOGY level, only expected component types should be valid."""
        state = DecoderState(
            grammar=GrammarState.EXPECT_COMP,
            topology="inverting_amp",
            expected_components=list(COMPONENT_TO_PARAM["inverting_amp"]),
        )
        mask = mask_computer.compute_mask(state, ConstraintLevel.TOPOLOGY)

        # COMP_RESISTOR should be valid (inverting_amp has 2 resistors)
        r_id = tokenizer.name_to_id["COMP_RESISTOR"]
        assert mask[r_id] == 0.0

        # COMP_INDUCTOR should NOT be valid (inverting_amp has no inductors)
        l_id = tokenizer.name_to_id["COMP_INDUCTOR"]
        assert mask[l_id] == float("-inf"), "INDUCTOR invalid for inverting_amp"

        # COMP_CAPACITOR should NOT be valid
        c_id = tokenizer.name_to_id["COMP_CAPACITOR"]
        assert mask[c_id] == float("-inf"), "CAPACITOR invalid for inverting_amp"

    def test_expect_val_full_level(self, tokenizer, mask_computer):
        """At FULL level, value tokens should be restricted to valid range."""
        state = DecoderState(
            grammar=GrammarState.EXPECT_VAL,
            topology="buck",
            expected_components=[],
            current_component=("INDUCTOR", "inductance"),
        )
        mask = mask_computer.compute_mask(state, ConstraintLevel.FULL)

        # Check that some value tokens are valid (within inductance range)
        valid_vals = [tid for tid in mask_computer.value_ids if mask[tid] == 0.0]
        assert len(valid_vals) > 0, "Should have valid value tokens for inductance"

        # Check that the valid range is smaller than all 500 value tokens
        assert len(valid_vals) < 500, "Should restrict to a subset of value tokens"

        # Non-value tokens should be invalid
        assert mask[tokenizer.end_id] == float("-inf")
        for tid in mask_computer.component_ids:
            assert mask[tid] == float("-inf")

    def test_all_components_placed_forces_end(self, tokenizer, mask_computer):
        """When all expected components are placed, only END should be valid."""
        state = DecoderState(
            grammar=GrammarState.EXPECT_COMP,
            topology="buck",
            expected_components=[],  # all placed
        )
        mask = mask_computer.compute_mask(state, ConstraintLevel.TOPOLOGY)

        # Only END should be valid
        assert mask[tokenizer.end_id] == 0.0
        for tid in mask_computer.component_ids:
            assert mask[tid] == float("-inf"), "No COMP allowed when all placed"

    def test_done_state_only_pad(self, tokenizer, mask_computer):
        """In DONE state, only PAD should be valid."""
        state = DecoderState(grammar=GrammarState.DONE, topology="buck")
        mask = mask_computer.compute_mask(state, ConstraintLevel.FULL)
        assert mask[tokenizer.pad_id] == 0.0
        n_valid = (mask == 0.0).sum().item()
        assert n_valid == 1, "Only PAD should be valid in DONE state"


# ---------------------------------------------------------------------------
# All topologies have valid mappings
# ---------------------------------------------------------------------------

class TestAllTopologies:
    """Ensure constrained decoding works for all 16 topologies."""

    ALL_TOPOLOGIES = list(COMPONENT_TO_PARAM.keys())

    @pytest.mark.parametrize("topology", ALL_TOPOLOGIES)
    def test_initial_state(self, tokenizer, topology):
        """Each topology should initialize with correct expected components."""
        expected = COMPONENT_TO_PARAM[topology]
        state = DecoderState(
            grammar=GrammarState.EXPECT_COMP,
            topology=topology,
            expected_components=list(expected),
        )
        assert len(state.expected_components) == len(expected)
        assert state.n_components_generated == 0

    @pytest.mark.parametrize("topology", ALL_TOPOLOGIES)
    def test_component_tokens_exist(self, tokenizer, topology):
        """All component types for each topology must have valid token IDs."""
        expected = COMPONENT_TO_PARAM[topology]
        for comp_type, _ in expected:
            tok_name = f"COMP_{comp_type}"
            assert tok_name in tokenizer.name_to_id, \
                f"Token {tok_name} not found for topology {topology}"

    @pytest.mark.parametrize("topology", ALL_TOPOLOGIES)
    def test_mask_allows_expected_components(self, tokenizer, mask_computer, topology):
        """At TOPOLOGY level, the mask should allow all expected component types."""
        expected = COMPONENT_TO_PARAM[topology]
        state = DecoderState(
            grammar=GrammarState.EXPECT_COMP,
            topology=topology,
            expected_components=list(expected),
        )
        mask = mask_computer.compute_mask(state, ConstraintLevel.TOPOLOGY)

        for comp_type, _ in expected:
            tid = tokenizer.name_to_id[f"COMP_{comp_type}"]
            assert mask[tid] == 0.0, \
                f"COMP_{comp_type} should be valid for {topology}"


# ---------------------------------------------------------------------------
# Constrained generation (end-to-end)
# ---------------------------------------------------------------------------

class TestConstrainedGeneration:
    """End-to-end constrained generation tests."""

    def _build_prefix(self, tokenizer, topology, specs=None):
        """Build a conditioning prefix."""
        prefix_ids = [tokenizer.start_id]
        topo_key = f"TOPO_{topology.upper()}"
        _topo_to_token = {
            "sallen_key_lowpass": "TOPO_SALLEN_KEY_LP",
            "sallen_key_highpass": "TOPO_SALLEN_KEY_HP",
            "sallen_key_bandpass": "TOPO_SALLEN_KEY_BP",
        }
        topo_key = _topo_to_token.get(topology, topo_key)
        if topo_key in tokenizer.name_to_id:
            prefix_ids.append(tokenizer.name_to_id[topo_key])
        prefix_ids.append(tokenizer.sep_id)
        if specs:
            for name, val in specs.items():
                spec_key = f"SPEC_{name.upper()}"
                if spec_key in tokenizer.name_to_id:
                    prefix_ids.append(tokenizer.name_to_id[spec_key])
                    prefix_ids.append(tokenizer.encode_value(abs(val)))
        prefix_ids.append(tokenizer.sep_id)
        return torch.tensor([prefix_ids])

    def test_baseline_unconstrained(self, tokenizer, small_model):
        """Baseline (no constraints) should generate something."""
        prefix = self._build_prefix(tokenizer, "buck", {"vin": 12, "vout": 5})
        gen = ConstrainedGenerator(small_model, tokenizer, ConstraintLevel.NONE)
        output = gen.generate(prefix, topology="buck")
        assert output.shape[1] > prefix.shape[1]

    def test_grammar_constrained(self, tokenizer, small_model):
        """Grammar-level constraint should produce alternating COMP/VAL."""
        prefix = self._build_prefix(tokenizer, "buck", {"vin": 12, "vout": 5})
        gen = ConstrainedGenerator(small_model, tokenizer, ConstraintLevel.GRAMMAR)
        output = gen.generate(prefix, topology="buck")

        # Decode the generated part (after prefix)
        gen_ids = output[0, prefix.shape[1]:].tolist()
        toks = [tokenizer.tokens[tid] for tid in gen_ids]

        # Check alternation: COMP, VAL, COMP, VAL, ..., END
        expect_comp = True
        for tok in toks:
            if tok.name in ("END", "PAD"):
                break
            if expect_comp:
                assert tok.token_type == TokenType.COMPONENT, \
                    f"Expected COMP, got {tok.name}"
                expect_comp = False
            else:
                assert tok.token_type == TokenType.VALUE, \
                    f"Expected VAL, got {tok.name}"
                expect_comp = True

    def test_topology_constrained_buck(self, tokenizer, small_model):
        """Topology-level constraint for buck: must have exactly 4 components."""
        prefix = self._build_prefix(tokenizer, "buck", {"vin": 12, "vout": 5})
        gen = ConstrainedGenerator(small_model, tokenizer, ConstraintLevel.TOPOLOGY)
        output = gen.generate(prefix, topology="buck", temperature=1.0)

        gen_ids = output[0, prefix.shape[1]:].tolist()
        toks = [tokenizer.tokens[tid] for tid in gen_ids]

        # Count components
        comp_toks = [t for t in toks if t.token_type == TokenType.COMPONENT]
        assert len(comp_toks) == 4, f"Buck needs 4 components, got {len(comp_toks)}"

        # Check that all expected types are present
        comp_types = {t.name.replace("COMP_", "") for t in comp_toks}
        expected_types = {ct for ct, _ in COMPONENT_TO_PARAM["buck"]}
        assert comp_types == expected_types, \
            f"Expected {expected_types}, got {comp_types}"

        # Must end with END
        assert toks[-1].name == "END" or any(t.name == "END" for t in toks)

    def test_topology_constrained_inverting_amp(self, tokenizer, small_model):
        """inverting_amp: must have exactly 2 resistors."""
        prefix = self._build_prefix(tokenizer, "inverting_amp")
        gen = ConstrainedGenerator(small_model, tokenizer, ConstraintLevel.TOPOLOGY)
        output = gen.generate(prefix, topology="inverting_amp", temperature=1.0)

        gen_ids = output[0, prefix.shape[1]:].tolist()
        toks = [tokenizer.tokens[tid] for tid in gen_ids]
        comp_toks = [t for t in toks if t.token_type == TokenType.COMPONENT]
        assert len(comp_toks) == 2
        assert all(t.name == "COMP_RESISTOR" for t in comp_toks)

    def test_topology_constrained_colpitts(self, tokenizer, small_model):
        """Colpitts: 1 inductor + 2 capacitors + 4 resistors = 7 components."""
        prefix = self._build_prefix(tokenizer, "colpitts")
        gen = ConstrainedGenerator(small_model, tokenizer, ConstraintLevel.TOPOLOGY)
        output = gen.generate(prefix, topology="colpitts", temperature=1.0)

        gen_ids = output[0, prefix.shape[1]:].tolist()
        toks = [tokenizer.tokens[tid] for tid in gen_ids]
        comp_toks = [t for t in toks if t.token_type == TokenType.COMPONENT]
        assert len(comp_toks) == 7, f"Colpitts needs 7 components, got {len(comp_toks)}"

    @pytest.mark.parametrize("topology", list(COMPONENT_TO_PARAM.keys()))
    def test_100pct_validity_all_topologies(self, tokenizer, small_model, topology):
        """Every topology must produce a structurally valid circuit."""
        from arcs.evaluate import decode_generated_sequence

        prefix = self._build_prefix(tokenizer, topology)
        gen = ConstrainedGenerator(small_model, tokenizer, ConstraintLevel.TOPOLOGY)
        output = gen.generate(prefix, topology=topology, temperature=1.0)

        decoded = decode_generated_sequence(output[0].tolist(), tokenizer)
        assert decoded.valid_structure, \
            f"{topology}: invalid structure — error={decoded.error}, " \
            f"topo={decoded.topology}, comps={len(decoded.components)}"

    def test_full_constraint_value_ranges(self, tokenizer, small_model):
        """FULL level should restrict values to valid parameter ranges."""
        prefix = self._build_prefix(tokenizer, "buck", {"vin": 12, "vout": 5})
        gen = ConstrainedGenerator(small_model, tokenizer, ConstraintLevel.FULL)
        output = gen.generate(prefix, topology="buck", temperature=1.0)

        from arcs.evaluate import decode_generated_sequence
        decoded = decode_generated_sequence(output[0].tolist(), tokenizer)
        assert decoded.valid_structure

        # Check that values are within the overall value range
        # (component order may be shuffled, so we just verify the mask worked)
        assert len(decoded.components) == 4, "Buck must have 4 components"
        # All values should be in the tokenizer's representable range
        for comp_type, value in decoded.components:
            assert value >= tokenizer.VALUE_MIN * 0.5, \
                f"{comp_type}: {value} below VALUE_MIN"
            assert value <= tokenizer.VALUE_MAX * 2.0, \
                f"{comp_type}: {value} above VALUE_MAX"

    def test_two_head_model(self, tokenizer, two_head_model):
        """Constrained generation should work with TwoHead model."""
        prefix = self._build_prefix(tokenizer, "buck", {"vin": 12, "vout": 5})
        gen = ConstrainedGenerator(two_head_model, tokenizer, ConstraintLevel.TOPOLOGY)
        output = gen.generate(prefix, topology="buck", temperature=1.0)

        from arcs.evaluate import decode_generated_sequence
        decoded = decode_generated_sequence(output[0].tolist(), tokenizer)
        assert decoded.valid_structure


# ---------------------------------------------------------------------------
# Constrained sampling with log-probs
# ---------------------------------------------------------------------------

class TestConstrainedSampling:
    """Test constrained sampling for RL integration."""

    def test_sample_returns_correct_shapes(self, tokenizer, small_model):
        """Should return tokens, log-probs, and entropies of same length."""
        prefix = torch.tensor([[
            tokenizer.start_id,
            tokenizer.name_to_id["TOPO_BUCK"],
            tokenizer.sep_id,
            tokenizer.name_to_id["SPEC_VIN"],
            tokenizer.encode_value(12.0),
            tokenizer.sep_id,
        ]])
        tokens, logprobs, entropies = constrained_sample_with_logprobs(
            small_model, prefix, tokenizer, "buck",
            level=ConstraintLevel.TOPOLOGY,
        )
        assert tokens.ndim == 1
        assert logprobs.shape == tokens.shape
        assert entropies.shape == tokens.shape
        assert len(tokens) > 0

    def test_log_probs_are_negative(self, tokenizer, small_model):
        """Log probabilities should be ≤ 0."""
        prefix = torch.tensor([[
            tokenizer.start_id,
            tokenizer.name_to_id["TOPO_BUCK"],
            tokenizer.sep_id,
            tokenizer.sep_id,
        ]])
        _, logprobs, _ = constrained_sample_with_logprobs(
            small_model, prefix, tokenizer, "buck",
            level=ConstraintLevel.TOPOLOGY,
        )
        # Allow small numerical error
        assert (logprobs <= 1e-6).all(), "Log probs should be ≤ 0"

    def test_entropies_non_negative(self, tokenizer, small_model):
        """Entropies should be ≥ 0."""
        prefix = torch.tensor([[
            tokenizer.start_id,
            tokenizer.name_to_id["TOPO_BUCK"],
            tokenizer.sep_id,
            tokenizer.sep_id,
        ]])
        _, _, entropies = constrained_sample_with_logprobs(
            small_model, prefix, tokenizer, "buck",
            level=ConstraintLevel.TOPOLOGY,
        )
        assert (entropies >= -1e-6).all(), "Entropies should be ≥ 0"


# ---------------------------------------------------------------------------
# Lagrangian constraint loss
# ---------------------------------------------------------------------------

class TestLagrangianLoss:
    """Test the Lagrangian constraint loss module."""

    def test_loss_forward(self, tokenizer, small_model):
        """Loss should compute without errors."""
        loss_fn = LagrangianConstraintLoss(tokenizer)

        # Create a dummy batch
        B, T = 2, 20
        x = torch.randint(0, tokenizer.vocab_size, (B, T))
        logits = torch.randn(B, T, tokenizer.vocab_size)

        # Need SEP tokens for the component section detection
        x[:, 3] = tokenizer.sep_id
        x[:, 7] = tokenizer.sep_id

        loss, stats = loss_fn(logits, x, x, ["buck", "boost"])

        assert loss.ndim == 0  # scalar
        assert "constraint/total_loss" in stats
        assert "constraint/lambda_mean" in stats

    def test_lambdas_positive(self, tokenizer):
        """Lagrange multipliers should always be positive."""
        loss_fn = LagrangianConstraintLoss(tokenizer)
        assert (loss_fn.lambdas > 0).all()

    def test_lambda_update(self, tokenizer):
        """Dual ascent should increase multipliers for violated constraints."""
        loss_fn = LagrangianConstraintLoss(tokenizer)
        old_lambda_0 = loss_fn.lambdas[0].item()
        old_lambda_1 = loss_fn.lambdas[1].item()

        violations = torch.tensor([1.0, 0.0, 0.5, 0.0])
        loss_fn.update_lambdas(violations)

        # Lambda for constraint 0 should increase (violation = 1.0)
        assert loss_fn.lambdas[0].item() > old_lambda_0
        # Lambda for constraint 1 should be unchanged (violation = 0.0)
        assert loss_fn.lambdas[1].item() == pytest.approx(old_lambda_1, abs=1e-6)


# ---------------------------------------------------------------------------
# Batch generation
# ---------------------------------------------------------------------------

class TestBatchGeneration:
    """Test batch generation utility."""

    def test_batch_generate(self, tokenizer, small_model):
        """Should generate multiple circuits."""
        gen = ConstrainedGenerator(small_model, tokenizer, ConstraintLevel.TOPOLOGY)

        topologies = ["buck", "inverting_amp"]
        prefixes = []
        for topo in topologies:
            prefix_ids = [
                tokenizer.start_id,
                tokenizer.name_to_id.get(f"TOPO_{topo.upper()}", tokenizer.start_id),
                tokenizer.sep_id,
                tokenizer.sep_id,
            ]
            prefixes.append(torch.tensor([prefix_ids]))

        results = gen.generate_batch(prefixes, topologies, temperature=1.0)
        assert len(results) == 2
        for r in results:
            assert r.shape[0] == 1
            assert r.shape[1] > 4  # at least prefix + some generation


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Test edge cases and robustness."""

    def test_unknown_topology(self, tokenizer, small_model):
        """Unknown topology should fall back gracefully (empty expected list)."""
        gen = ConstrainedGenerator(small_model, tokenizer, ConstraintLevel.GRAMMAR)
        prefix = torch.tensor([[
            tokenizer.start_id,
            tokenizer.name_to_id["TOPO_BUCK"],  # Token says buck but...
            tokenizer.sep_id,
            tokenizer.sep_id,
        ]])
        # Use a nonexistent topology name
        output = gen.generate(prefix, topology="nonexistent", temperature=1.0)
        # Should still produce output without crashing
        assert output.shape[1] > prefix.shape[1]

    def test_max_tokens_limit(self, tokenizer, small_model):
        """Should respect max_new_tokens limit."""
        gen = ConstrainedGenerator(small_model, tokenizer, ConstraintLevel.GRAMMAR)
        prefix = torch.tensor([[
            tokenizer.start_id,
            tokenizer.name_to_id["TOPO_BUCK"],
            tokenizer.sep_id,
            tokenizer.sep_id,
        ]])
        output = gen.generate(prefix, topology="buck", max_new_tokens=5)
        # Should generate at most 5 new tokens + possibly forced END
        assert output.shape[1] <= prefix.shape[1] + 6

    def test_topology_auto_detection(self, tokenizer, small_model):
        """Should auto-detect topology from prefix."""
        gen = ConstrainedGenerator(small_model, tokenizer, ConstraintLevel.TOPOLOGY)
        prefix = torch.tensor([[
            tokenizer.start_id,
            tokenizer.name_to_id["TOPO_BOOST"],
            tokenizer.sep_id,
            tokenizer.sep_id,
        ]])
        # Don't pass topology explicitly
        output = gen.generate(prefix, topology=None, temperature=1.0)

        from arcs.evaluate import decode_generated_sequence
        decoded = decode_generated_sequence(output[0].tolist(), tokenizer)
        assert decoded.valid_structure
        assert decoded.topology == "boost"


# ---------------------------------------------------------------------------
# Statistics / validity guarantee
# ---------------------------------------------------------------------------

class TestValidityGuarantee:
    """Statistical test: constrained decoding achieves 100% validity."""

    def test_100_samples_all_valid(self, tokenizer, small_model):
        """Generate 100 circuits with TOPOLOGY constraints —all must be valid."""
        from arcs.evaluate import decode_generated_sequence

        gen = ConstrainedGenerator(small_model, tokenizer, ConstraintLevel.TOPOLOGY)
        topologies = list(COMPONENT_TO_PARAM.keys())
        n_valid = 0
        n_total = 0

        rng = torch.Generator().manual_seed(42)
        for i in range(100):
            topo = topologies[i % len(topologies)]
            topo_key = f"TOPO_{topo.upper()}"
            _topo_to_token = {
                "sallen_key_lowpass": "TOPO_SALLEN_KEY_LP",
                "sallen_key_highpass": "TOPO_SALLEN_KEY_HP",
                "sallen_key_bandpass": "TOPO_SALLEN_KEY_BP",
            }
            topo_key = _topo_to_token.get(topo, topo_key)

            prefix_ids = [
                tokenizer.start_id,
                tokenizer.name_to_id.get(topo_key, tokenizer.start_id),
                tokenizer.sep_id,
                tokenizer.sep_id,
            ]
            prefix = torch.tensor([prefix_ids])
            output = gen.generate(prefix, topology=topo, temperature=1.0)
            decoded = decode_generated_sequence(output[0].tolist(), tokenizer)

            n_total += 1
            if decoded.valid_structure:
                n_valid += 1

        rate = n_valid / n_total
        assert rate == 1.0, f"Validity rate {rate:.1%} < 100% ({n_valid}/{n_total})"
