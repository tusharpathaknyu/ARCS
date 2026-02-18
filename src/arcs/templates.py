"""Parameterized SPICE netlist templates for power converter topologies.

Each template is a function that takes component values as arguments
and returns a complete SPICE netlist string with .measure statements
for extracting performance metrics.

Tier 1: Power Electronics
- Buck converter
- Boost converter
- Buck-Boost converter
- Flyback converter
- Forward converter
- Ćuk converter
- SEPIC converter
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable
import numpy as np


@dataclass
class ComponentBounds:
    """Physical bounds for a component parameter."""

    name: str
    unit: str
    min_val: float
    max_val: float
    log_scale: bool = True  # Sample in log space (natural for R, L, C)
    description: str = ""

    def sample(self, rng: np.random.Generator | None = None) -> float:
        """Sample a random value within bounds."""
        rng = rng or np.random.default_rng()
        if self.log_scale:
            return float(np.exp(rng.uniform(np.log(self.min_val), np.log(self.max_val))))
        else:
            return float(rng.uniform(self.min_val, self.max_val))

    def snap_to_e_series(self, value: float, series: int = 24) -> float:
        """Snap a value to the nearest E-series standard value."""
        # E-series: E12, E24, E48, E96
        e_series = {
            12: [1.0, 1.2, 1.5, 1.8, 2.2, 2.7, 3.3, 3.9, 4.7, 5.6, 6.8, 8.2],
            24: [1.0, 1.1, 1.2, 1.3, 1.5, 1.6, 1.8, 2.0, 2.2, 2.4, 2.7, 3.0,
                 3.3, 3.6, 3.9, 4.3, 4.7, 5.1, 5.6, 6.2, 6.8, 7.5, 8.2, 9.1],
        }
        if series not in e_series:
            return value

        decade = np.floor(np.log10(value))
        mantissa = value / (10 ** decade)

        # Find closest E-series value
        vals = e_series[series]
        closest = min(vals, key=lambda x: abs(x - mantissa))

        return closest * (10 ** decade)


@dataclass
class TopologyTemplate:
    """A parameterized circuit topology with bounds and netlist generator."""

    name: str
    description: str
    component_bounds: list[ComponentBounds]
    netlist_fn: Callable[..., str]  # Function that generates netlist from values
    metric_names: list[str]  # Expected .measure output names
    operating_conditions: dict[str, float] = field(default_factory=dict)

    def sample_parameters(self, rng: np.random.Generator | None = None) -> dict[str, float]:
        """Sample random component values within bounds."""
        rng = rng or np.random.default_rng()
        return {b.name: b.sample(rng) for b in self.component_bounds}

    def generate_netlist(self, params: dict[str, float]) -> str:
        """Generate a SPICE netlist from parameter values."""
        return self.netlist_fn(params, self.operating_conditions)


# =============================================================================
# Buck Converter
# =============================================================================

def _buck_netlist(params: dict[str, float], conditions: dict[str, float]) -> str:
    """Synchronous buck converter with voltage-mode PWM control.

    Uses a behavioral PWM source for the switch to avoid convergence issues.
    Simplified model suitable for steady-state performance extraction.
    """
    vin = conditions.get("vin", 12.0)
    vout_target = conditions.get("vout", 5.0)
    iout = conditions.get("iout", 1.0)
    fsw = conditions.get("fsw", 100e3)

    L = params["inductance"]
    C = params["capacitance"]
    R_load = params.get("r_load", vout_target / iout)
    R_esr = params.get("esr", 0.01)
    R_dson = params.get("r_dson", 0.05)
    duty = vout_target / vin

    period = 1.0 / fsw
    ton = duty * period
    sim_time = 200 * period
    meas_start = 150 * period
    tstep = period / 100

    return f"""\
* ARCS Buck Converter
* Vin={vin}V, Vout_target={vout_target}V, Iout={iout}A, fsw={fsw/1e3:.0f}kHz

Vin input 0 DC {vin}

* PWM control signal (0–5V swing for switch threshold)
Vpwm pwm_ctrl 0 PULSE(0 5 0 1n 1n {ton:.10e} {period:.10e})

* Main MOSFET as voltage-controlled switch
S1 input sw_node pwm_ctrl 0 SMOD
.model SMOD SW(RON={R_dson} ROFF=1e6 VT=2.5 VH=0.1)

* Freewheeling diode
Dfw 0 sw_node DSCHOTTKY
.model DSCHOTTKY D(IS=1e-6 RS=0.03 N=1.05 BV=40 CJO=100p)

* Output LC filter
L1 sw_node vout {L:.6e} IC=0
Resr vout cap_node {R_esr}
C1 cap_node 0 {C:.6e} IC={vout_target}

* Load with current sense
Rload vout load_mid {R_load:.4f}
Vsense load_mid 0 DC 0

* === Analysis ===
.tran {tstep:.10e} {sim_time:.10e} {meas_start:.10e} UIC

* === Measurements ===
.measure TRAN vout_avg AVG V(vout) FROM={meas_start:.10e} TO={sim_time:.10e}
.measure TRAN vout_ripple PP V(vout) FROM={meas_start:.10e} TO={sim_time:.10e}
.measure TRAN iout_avg AVG I(Vsense) FROM={meas_start:.10e} TO={sim_time:.10e}
.measure TRAN iin_avg AVG I(Vin) FROM={meas_start:.10e} TO={sim_time:.10e}
.measure TRAN il_ripple PP I(L1) FROM={meas_start:.10e} TO={sim_time:.10e}

.end
"""


def _boost_netlist(params: dict[str, float], conditions: dict[str, float]) -> str:
    """Boost converter template. Steps up voltage: Vout > Vin."""
    vin = conditions.get("vin", 5.0)
    vout_target = conditions.get("vout", 12.0)
    iout = conditions.get("iout", 0.5)
    fsw = conditions.get("fsw", 100e3)

    L = params["inductance"]
    C = params["capacitance"]
    R_load = params.get("r_load", vout_target / iout)
    R_esr = params.get("esr", 0.02)
    R_dson = params.get("r_dson", 0.05)
    duty = max(0.05, min(0.95, 1.0 - (vin / vout_target)))

    period = 1.0 / fsw
    ton = duty * period
    sim_time = 200 * period
    meas_start = 150 * period
    tstep = period / 100

    return f"""\
* ARCS Boost Converter
* Vin={vin}V, Vout_target={vout_target}V, Iout={iout}A, fsw={fsw/1e3:.0f}kHz

Vin input 0 DC {vin}

L1 input sw_node {L:.6e} IC=0

Vpwm pwm_ctrl 0 PULSE(0 1 0 1n 1n {ton:.10e} {period:.10e})
Bsw sw_node 0 I = V(sw_node) / {R_dson} * V(pwm_ctrl)
Rsw_damp sw_node 0 1e6

Dboost sw_node vout DSCHOTTKY
.model DSCHOTTKY D(IS=1e-6 RS=0.03 N=1.05 BV=100 CJO=100p)

Resr vout cap_node {R_esr}
C1 cap_node 0 {C:.6e} IC={vout_target}

Rload vout load_mid {R_load:.4f}
Vsense load_mid 0 DC 0

.tran {tstep:.10e} {sim_time:.10e} {meas_start:.10e} UIC

.measure TRAN vout_avg AVG V(vout) FROM={meas_start:.10e} TO={sim_time:.10e}
.measure TRAN vout_ripple PP V(vout) FROM={meas_start:.10e} TO={sim_time:.10e}
.measure TRAN iout_avg AVG I(Vsense) FROM={meas_start:.10e} TO={sim_time:.10e}
.measure TRAN iin_avg AVG I(Vin) FROM={meas_start:.10e} TO={sim_time:.10e}
.measure TRAN il_ripple PP I(L1) FROM={meas_start:.10e} TO={sim_time:.10e}

.end
"""


def _buck_boost_netlist(params: dict[str, float], conditions: dict[str, float]) -> str:
    """Inverting buck-boost converter template."""
    vin = conditions.get("vin", 12.0)
    vout_target = conditions.get("vout", -9.0)
    iout = conditions.get("iout", 0.5)
    fsw = conditions.get("fsw", 100e3)

    L = params["inductance"]
    C = params["capacitance"]
    R_load = params.get("r_load", abs(vout_target) / iout)
    R_esr = params.get("esr", 0.02)
    R_dson = params.get("r_dson", 0.05)
    duty = abs(vout_target) / (vin + abs(vout_target))

    period = 1.0 / fsw
    ton = duty * period
    sim_time = 200 * period
    meas_start = 150 * period
    tstep = period / 100

    return f"""\
* ARCS Inverting Buck-Boost Converter
* Vin={vin}V, Vout_target={vout_target}V, fsw={fsw/1e3:.0f}kHz

Vin input 0 DC {vin}

Vpwm pwm_ctrl 0 PULSE(0 5 0 1n 1n {ton:.10e} {period:.10e})

* Main MOSFET as voltage-controlled switch
S1 input sw_node pwm_ctrl 0 SMOD
.model SMOD SW(RON={R_dson} ROFF=1e6 VT=2.5 VH=0.1)

L1 sw_node 0 {L:.6e} IC=0

Dbb sw_node vout_neg DSCHOTTKY
.model DSCHOTTKY D(IS=1e-6 RS=0.03 N=1.05 BV=60 CJO=100p)

Resr vout_neg cap_node {R_esr}
C1 cap_node 0 {C:.6e} IC={vout_target}

Rload vout_neg load_mid {R_load:.4f}
Vsense load_mid 0 DC 0

.tran {tstep:.10e} {sim_time:.10e} {meas_start:.10e} UIC

.measure TRAN vout_avg AVG V(vout_neg) FROM={meas_start:.10e} TO={sim_time:.10e}
.measure TRAN vout_ripple PP V(vout_neg) FROM={meas_start:.10e} TO={sim_time:.10e}
.measure TRAN iout_avg AVG I(Vsense) FROM={meas_start:.10e} TO={sim_time:.10e}
.measure TRAN iin_avg AVG I(Vin) FROM={meas_start:.10e} TO={sim_time:.10e}

.end
"""


def _cuk_netlist(params: dict[str, float], conditions: dict[str, float]) -> str:
    """Ćuk converter template. Two inductors + coupling cap."""
    vin = conditions.get("vin", 12.0)
    vout_target = conditions.get("vout", 5.0)
    iout = conditions.get("iout", 1.0)
    fsw = conditions.get("fsw", 100e3)

    L1 = params["inductance_1"]
    L2 = params["inductance_2"]
    C_couple = params["cap_coupling"]
    C_out = params["capacitance"]
    R_load = params.get("r_load", vout_target / iout)
    R_esr = params.get("esr", 0.02)
    R_dson = params.get("r_dson", 0.05)
    duty = max(0.05, min(0.95, vout_target / (vin + vout_target)))

    period = 1.0 / fsw
    ton = duty * period
    sim_time = 200 * period
    meas_start = 150 * period
    tstep = period / 100

    return f"""\
* ARCS Cuk Converter
* Vin={vin}V, Vout_target={vout_target}V, fsw={fsw/1e3:.0f}kHz

Vin input 0 DC {vin}

L1 input sw_node {L1:.6e} IC=0

Vpwm pwm_ctrl 0 PULSE(0 1 0 1n 1n {ton:.10e} {period:.10e})
Bsw sw_node 0 I = V(sw_node) / {R_dson} * V(pwm_ctrl)
Rsw_damp sw_node 0 1e6

Cc sw_node mid {C_couple:.6e} IC={vin}

Dcuk mid vout_node DSCHOTTKY
.model DSCHOTTKY D(IS=1e-6 RS=0.03 N=1.05 BV=60 CJO=100p)

L2 vout_node vout {L2:.6e} IC=0

Resr vout cap_node {R_esr}
C1 cap_node 0 {C_out:.6e} IC={vout_target}

Rload vout load_mid {R_load:.4f}
Vsense load_mid 0 DC 0

.tran {tstep:.10e} {sim_time:.10e} {meas_start:.10e} UIC

.measure TRAN vout_avg AVG V(vout) FROM={meas_start:.10e} TO={sim_time:.10e}
.measure TRAN vout_ripple PP V(vout) FROM={meas_start:.10e} TO={sim_time:.10e}
.measure TRAN iout_avg AVG I(Vsense) FROM={meas_start:.10e} TO={sim_time:.10e}
.measure TRAN iin_avg AVG I(Vin) FROM={meas_start:.10e} TO={sim_time:.10e}

.end
"""


def _sepic_netlist(params: dict[str, float], conditions: dict[str, float]) -> str:
    """SEPIC converter template. Non-inverting, can step up or down."""
    vin = conditions.get("vin", 12.0)
    vout_target = conditions.get("vout", 5.0)
    iout = conditions.get("iout", 1.0)
    fsw = conditions.get("fsw", 100e3)

    L1 = params["inductance_1"]
    L2 = params["inductance_2"]
    C_couple = params["cap_coupling"]
    C_out = params["capacitance"]
    R_load = params.get("r_load", vout_target / iout)
    R_esr = params.get("esr", 0.02)
    R_dson = params.get("r_dson", 0.05)
    duty = max(0.05, min(0.95, vout_target / (vin + vout_target)))

    period = 1.0 / fsw
    ton = duty * period
    sim_time = 200 * period
    meas_start = 150 * period
    tstep = period / 100

    return f"""\
* ARCS SEPIC Converter
* Vin={vin}V, Vout_target={vout_target}V, fsw={fsw/1e3:.0f}kHz

Vin input 0 DC {vin}

L1 input sw_node {L1:.6e} IC=0

Vpwm pwm_ctrl 0 PULSE(0 1 0 1n 1n {ton:.10e} {period:.10e})
Bsw sw_node 0 I = V(sw_node) / {R_dson} * V(pwm_ctrl)
Rsw_damp sw_node 0 1e6

Cc sw_node l2_in {C_couple:.6e} IC={vin}

L2 l2_in diode_a {L2:.6e} IC=0

Dsepic diode_a vout DSCHOTTKY
.model DSCHOTTKY D(IS=1e-6 RS=0.03 N=1.05 BV=60 CJO=100p)

Resr vout cap_node {R_esr}
C1 cap_node 0 {C_out:.6e} IC={vout_target}

Rload vout load_mid {R_load:.4f}
Vsense load_mid 0 DC 0

.tran {tstep:.10e} {sim_time:.10e} {meas_start:.10e} UIC

.measure TRAN vout_avg AVG V(vout) FROM={meas_start:.10e} TO={sim_time:.10e}
.measure TRAN vout_ripple PP V(vout) FROM={meas_start:.10e} TO={sim_time:.10e}
.measure TRAN iout_avg AVG I(Vsense) FROM={meas_start:.10e} TO={sim_time:.10e}
.measure TRAN iin_avg AVG I(Vin) FROM={meas_start:.10e} TO={sim_time:.10e}

.end
"""


def _flyback_netlist(params: dict[str, float], conditions: dict[str, float]) -> str:
    """Flyback converter with coupled inductor (transformer) model."""
    vin = conditions.get("vin", 12.0)
    vout_target = conditions.get("vout", 5.0)
    iout = conditions.get("iout", 1.0)
    fsw = conditions.get("fsw", 100e3)

    Lp = params["inductance_primary"]
    turns_ratio = params["turns_ratio"]
    C_out = params["capacitance"]
    R_load = params.get("r_load", vout_target / iout)
    R_esr = params.get("esr", 0.02)
    R_dson = params.get("r_dson", 0.05)
    duty = max(0.05, min(0.85, vout_target / (vout_target + vin / turns_ratio)))

    Ls = Lp / (turns_ratio ** 2)
    k = 0.98

    period = 1.0 / fsw
    ton = duty * period
    sim_time = 200 * period
    meas_start = 150 * period
    tstep = period / 100

    return f"""\
* ARCS Flyback Converter
* Vin={vin}V, Vout_target={vout_target}V, N={turns_ratio:.2f}, fsw={fsw/1e3:.0f}kHz

Vin input 0 DC {vin}

L_pri input sw_drain {Lp:.6e} IC=0
L_sec sec_dot 0 {Ls:.6e} IC=0
K1 L_pri L_sec {k}

Vpwm pwm_ctrl 0 PULSE(0 1 0 1n 1n {ton:.10e} {period:.10e})
Bsw sw_drain 0 I = V(sw_drain) / {R_dson} * V(pwm_ctrl)
Rsw_damp sw_drain 0 1e6

Dfb sec_dot vout DSCHOTTKY
.model DSCHOTTKY D(IS=1e-6 RS=0.03 N=1.05 BV=100 CJO=100p)

Resr vout cap_node {R_esr}
C1 cap_node 0 {C_out:.6e} IC={vout_target}

Rload vout load_mid {R_load:.4f}
Vsense load_mid 0 DC 0

.tran {tstep:.10e} {sim_time:.10e} {meas_start:.10e} UIC

.measure TRAN vout_avg AVG V(vout) FROM={meas_start:.10e} TO={sim_time:.10e}
.measure TRAN vout_ripple PP V(vout) FROM={meas_start:.10e} TO={sim_time:.10e}
.measure TRAN iout_avg AVG I(Vsense) FROM={meas_start:.10e} TO={sim_time:.10e}
.measure TRAN iin_avg AVG I(Vin) FROM={meas_start:.10e} TO={sim_time:.10e}

.end
"""


def _forward_netlist(params: dict[str, float], conditions: dict[str, float]) -> str:
    """Forward converter template. Isolated buck derivative."""
    vin = conditions.get("vin", 48.0)
    vout_target = conditions.get("vout", 12.0)
    iout = conditions.get("iout", 2.0)
    fsw = conditions.get("fsw", 100e3)

    Lp = params["inductance_primary"]
    turns_ratio = params["turns_ratio"]
    L_out = params["inductance_output"]
    C_out = params["capacitance"]
    R_load = params.get("r_load", vout_target / iout)
    R_esr = params.get("esr", 0.02)
    R_dson = params.get("r_dson", 0.05)
    duty = max(0.05, min(0.85, vout_target / (vin / turns_ratio)))

    Ls = Lp / (turns_ratio ** 2)
    k = 0.98

    period = 1.0 / fsw
    ton = duty * period
    sim_time = 200 * period
    meas_start = 150 * period
    tstep = period / 100

    return f"""\
* ARCS Forward Converter
* Vin={vin}V, Vout_target={vout_target}V, N={turns_ratio:.2f}, fsw={fsw/1e3:.0f}kHz

Vin input 0 DC {vin}

L_pri input sw_drain {Lp:.6e} IC=0
L_sec 0 sec_out {Ls:.6e} IC=0
K1 L_pri L_sec {k}

Vpwm pwm_ctrl 0 PULSE(0 1 0 1n 1n {ton:.10e} {period:.10e})
Bsw sw_drain 0 I = V(sw_drain) / {R_dson} * V(pwm_ctrl)
Rsw_damp sw_drain 0 1e6

Dfwd sec_out rect_out DSCHOTTKY
.model DSCHOTTKY D(IS=1e-6 RS=0.03 N=1.05 BV=100 CJO=100p)

Dfw 0 rect_out DSCHOTTKY2
.model DSCHOTTKY2 D(IS=1e-6 RS=0.03 N=1.05 BV=100 CJO=100p)

Lout rect_out vout {L_out:.6e} IC=0
Resr vout cap_node {R_esr}
C1 cap_node 0 {C_out:.6e} IC={vout_target}

Rload vout load_mid {R_load:.4f}
Vsense load_mid 0 DC 0

.tran {tstep:.10e} {sim_time:.10e} {meas_start:.10e} UIC

.measure TRAN vout_avg AVG V(vout) FROM={meas_start:.10e} TO={sim_time:.10e}
.measure TRAN vout_ripple PP V(vout) FROM={meas_start:.10e} TO={sim_time:.10e}
.measure TRAN iout_avg AVG I(Vsense) FROM={meas_start:.10e} TO={sim_time:.10e}
.measure TRAN iin_avg AVG I(Vin) FROM={meas_start:.10e} TO={sim_time:.10e}

.end
"""


# =============================================================================
# Registry of all topologies
# =============================================================================

POWER_CONVERTER_BOUNDS = {
    "buck": [
        ComponentBounds("inductance", "H", 1e-6, 1e-3, log_scale=True, description="Output inductor"),
        ComponentBounds("capacitance", "F", 1e-6, 1e-2, log_scale=True, description="Output capacitor"),
        ComponentBounds("esr", "Ω", 0.001, 0.5, log_scale=True, description="Cap ESR"),
        ComponentBounds("r_dson", "Ω", 0.01, 0.5, log_scale=True, description="MOSFET Rds(on)"),
    ],
    "boost": [
        ComponentBounds("inductance", "H", 1e-6, 1e-3, log_scale=True, description="Boost inductor"),
        ComponentBounds("capacitance", "F", 1e-6, 1e-2, log_scale=True, description="Output capacitor"),
        ComponentBounds("esr", "Ω", 0.001, 0.5, log_scale=True, description="Cap ESR"),
        ComponentBounds("r_dson", "Ω", 0.01, 0.5, log_scale=True, description="MOSFET Rds(on)"),
    ],
    "buck_boost": [
        ComponentBounds("inductance", "H", 1e-6, 1e-3, log_scale=True),
        ComponentBounds("capacitance", "F", 1e-6, 1e-2, log_scale=True),
        ComponentBounds("esr", "Ω", 0.001, 0.5, log_scale=True),
        ComponentBounds("r_dson", "Ω", 0.01, 0.5, log_scale=True),
    ],
    "cuk": [
        ComponentBounds("inductance_1", "H", 1e-6, 1e-3, log_scale=True, description="Input inductor"),
        ComponentBounds("inductance_2", "H", 1e-6, 1e-3, log_scale=True, description="Output inductor"),
        ComponentBounds("cap_coupling", "F", 0.1e-6, 100e-6, log_scale=True, description="Coupling cap"),
        ComponentBounds("capacitance", "F", 1e-6, 1e-2, log_scale=True, description="Output cap"),
        ComponentBounds("esr", "Ω", 0.001, 0.5, log_scale=True),
        ComponentBounds("r_dson", "Ω", 0.01, 0.5, log_scale=True),
    ],
    "sepic": [
        ComponentBounds("inductance_1", "H", 1e-6, 1e-3, log_scale=True),
        ComponentBounds("inductance_2", "H", 1e-6, 1e-3, log_scale=True),
        ComponentBounds("cap_coupling", "F", 0.1e-6, 100e-6, log_scale=True),
        ComponentBounds("capacitance", "F", 1e-6, 1e-2, log_scale=True),
        ComponentBounds("esr", "Ω", 0.001, 0.5, log_scale=True),
        ComponentBounds("r_dson", "Ω", 0.01, 0.5, log_scale=True),
    ],
    "flyback": [
        ComponentBounds("inductance_primary", "H", 10e-6, 5e-3, log_scale=True),
        ComponentBounds("turns_ratio", "", 0.1, 10.0, log_scale=True, description="Np/Ns"),
        ComponentBounds("capacitance", "F", 1e-6, 1e-2, log_scale=True),
        ComponentBounds("esr", "Ω", 0.001, 0.5, log_scale=True),
        ComponentBounds("r_dson", "Ω", 0.01, 0.5, log_scale=True),
    ],
    "forward": [
        ComponentBounds("inductance_primary", "H", 10e-6, 5e-3, log_scale=True),
        ComponentBounds("turns_ratio", "", 0.1, 10.0, log_scale=True),
        ComponentBounds("inductance_output", "H", 1e-6, 1e-3, log_scale=True),
        ComponentBounds("capacitance", "F", 1e-6, 1e-2, log_scale=True),
        ComponentBounds("esr", "Ω", 0.001, 0.5, log_scale=True),
        ComponentBounds("r_dson", "Ω", 0.01, 0.5, log_scale=True),
    ],
}


OPERATING_CONDITIONS = {
    "buck": {"vin": 12.0, "vout": 5.0, "iout": 1.0, "fsw": 100e3},
    "boost": {"vin": 5.0, "vout": 12.0, "iout": 0.5, "fsw": 100e3},
    "buck_boost": {"vin": 12.0, "vout": -9.0, "iout": 0.5, "fsw": 100e3},
    "cuk": {"vin": 12.0, "vout": 5.0, "iout": 1.0, "fsw": 100e3},
    "sepic": {"vin": 12.0, "vout": 5.0, "iout": 1.0, "fsw": 100e3},
    "flyback": {"vin": 12.0, "vout": 5.0, "iout": 1.0, "fsw": 100e3},
    "forward": {"vin": 48.0, "vout": 12.0, "iout": 2.0, "fsw": 100e3},
}


def get_topology(name: str) -> TopologyTemplate:
    """Get a topology template by name."""
    netlist_fns = {
        "buck": _buck_netlist,
        "boost": _boost_netlist,
        "buck_boost": _buck_boost_netlist,
        "cuk": _cuk_netlist,
        "sepic": _sepic_netlist,
        "flyback": _flyback_netlist,
        "forward": _forward_netlist,
    }

    descriptions = {
        "buck": "Step-down DC-DC converter",
        "boost": "Step-up DC-DC converter",
        "buck_boost": "Inverting buck-boost DC-DC converter",
        "cuk": "Ćuk DC-DC converter (non-inverting, two inductors)",
        "sepic": "SEPIC DC-DC converter (non-inverting, can step up or down)",
        "flyback": "Isolated flyback DC-DC converter",
        "forward": "Isolated forward DC-DC converter",
    }

    metric_names = ["vout_avg", "vout_ripple", "iout_avg", "iin_avg", "il_ripple", "pout", "pin"]

    if name not in netlist_fns:
        raise ValueError(f"Unknown topology: {name}. Available: {list(netlist_fns.keys())}")

    return TopologyTemplate(
        name=name,
        description=descriptions[name],
        component_bounds=POWER_CONVERTER_BOUNDS[name],
        netlist_fn=netlist_fns[name],
        metric_names=metric_names,
        operating_conditions=OPERATING_CONDITIONS[name],
    )


def get_all_topologies() -> list[TopologyTemplate]:
    """Get all available topology templates."""
    names = ["buck", "boost", "buck_boost", "cuk", "sepic", "flyback", "forward"]
    return [get_topology(name) for name in names]
