"""Parameterized SPICE netlist templates for analog circuit topologies.

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

Tier 2: Analog Signal Processing
- Inverting amplifier (op-amp)
- Non-inverting amplifier (op-amp)
- Sallen-Key low-pass filter
- Sallen-Key high-pass filter
- Sallen-Key band-pass filter
- Wien bridge oscillator
- Colpitts oscillator
- Instrumentation amplifier (3 op-amp)
- Differential amplifier (single op-amp)
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
    sim_time = 500 * period
    meas_start = 400 * period
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
    sim_time = 500 * period
    meas_start = 400 * period
    tstep = period / 100

    return f"""\
* ARCS Boost Converter
* Vin={vin}V, Vout_target={vout_target}V, Iout={iout}A, fsw={fsw/1e3:.0f}kHz

Vin input 0 DC {vin}

L1 input sw_node {L:.6e} IC=0

* Shunt MOSFET switch to ground
Vpwm pwm_ctrl 0 PULSE(0 5 0 1n 1n {ton:.10e} {period:.10e})
S1 sw_node 0 pwm_ctrl 0 SMOD
.model SMOD SW(RON={R_dson} ROFF=1e6 VT=2.5 VH=0.1)

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
    sim_time = 500 * period
    meas_start = 400 * period
    tstep = period / 100

    return f"""\
* ARCS Inverting Buck-Boost Converter
* Vin={vin}V, Vout_target={vout_target}V, fsw={fsw/1e3:.0f}kHz

Vin input 0 DC {vin}

Vpwm pwm_ctrl 0 PULSE(0 5 0 1n 1n {ton:.10e} {period:.10e})

* Main MOSFET as voltage-controlled switch
S1 input sw_node pwm_ctrl 0 SMOD
.model SMOD SW(RON={R_dson} ROFF=1e6 VT=2.5 VH=0.1)

* Inductor stores energy when switch is on
L1 sw_node 0 {L:.6e} IC=0

* Diode conducts inductor current to negative output when switch is off
* Current flows: 0 -> vout_neg (charging it negative) -> Dbb -> sw_node
Dbb vout_neg sw_node DSCHOTTKY
.model DSCHOTTKY D(IS=1e-6 RS=0.03 N=1.05 BV=60 CJO=100p)

* Output cap and load (output is negative)
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
    """Ćuk converter template (inverting). Two inductors + coupling cap.
    
    Standard Ćuk: Vout is NEGATIVE.
    Switch ON: L1 charges from Vin, D off, L2 freewheels through output.
    Switch OFF: L1 charges Cc, D conducts, energy transfers to output.
    """
    vin = conditions.get("vin", 12.0)
    vout_target = conditions.get("vout", -5.0)
    iout = conditions.get("iout", 1.0)
    fsw = conditions.get("fsw", 100e3)

    L1 = params["inductance_1"]
    L2 = params["inductance_2"]
    C_couple = params["cap_coupling"]
    C_out = params["capacitance"]
    R_load = params.get("r_load", abs(vout_target) / iout)
    R_esr = params.get("esr", 0.02)
    R_dson = params.get("r_dson", 0.05)
    duty = max(0.05, min(0.95, abs(vout_target) / (vin + abs(vout_target))))

    period = 1.0 / fsw
    ton = duty * period
    sim_time = 500 * period
    meas_start = 400 * period
    tstep = period / 100

    return f"""\
* ARCS Cuk Converter (Inverting)
* Vin={vin}V, Vout_target={vout_target}V, fsw={fsw/1e3:.0f}kHz

Vin input 0 DC {vin}

* Input inductor
L1 input sw_node {L1:.6e} IC=0

* Shunt MOSFET switch to ground
Vpwm pwm_ctrl 0 PULSE(0 5 0 1n 1n {ton:.10e} {period:.10e})
S1 sw_node 0 pwm_ctrl 0 SMOD
.model SMOD SW(RON={R_dson} ROFF=1e6 VT=2.5 VH=0.1)

* Coupling capacitor
Cc sw_node cb {C_couple:.6e} IC={vin}

* Diode from cb to GND: conducts when cb > 0 (switch OFF, L1 charges Cc)
Dcuk cb 0 DSCHOTTKY
.model DSCHOTTKY D(IS=1e-6 RS=0.03 N=1.05 BV=60 CJO=100p)

* Output inductor from cb to negative output
L2 cb vout {L2:.6e} IC=0

* Output cap and load (output voltage is negative)
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
    """SEPIC converter template. Non-inverting, can step up or down.
    
    Switch ON: L1 charges from Vin, L2 charges from Cc, D off.
    Switch OFF: L1+L2 energy transfers to output through D.
    """
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
    sim_time = 500 * period
    meas_start = 400 * period
    tstep = period / 100

    return f"""\
* ARCS SEPIC Converter
* Vin={vin}V, Vout_target={vout_target}V, fsw={fsw/1e3:.0f}kHz

Vin input 0 DC {vin}

* Input inductor
L1 input sw_node {L1:.6e} IC=0

* Shunt MOSFET switch to ground
Vpwm pwm_ctrl 0 PULSE(0 5 0 1n 1n {ton:.10e} {period:.10e})
S1 sw_node 0 pwm_ctrl 0 SMOD
.model SMOD SW(RON={R_dson} ROFF=1e6 VT=2.5 VH=0.1)

* Coupling capacitor
Cc sw_node cb {C_couple:.6e} IC={vin}

* Second inductor from cb to ground
L2 cb 0 {L2:.6e} IC=0

* Output diode from cb junction to output (non-inverting)
Dsepic cb vout DSCHOTTKY
.model DSCHOTTKY D(IS=1e-6 RS=0.03 N=1.05 BV=60 CJO=100p)

* Output cap and load
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
    sim_time = 500 * period
    meas_start = 400 * period
    tstep = period / 100

    return f"""\
* ARCS Flyback Converter
* Vin={vin}V, Vout_target={vout_target}V, N={turns_ratio:.2f}, fsw={fsw/1e3:.0f}kHz

Vin input 0 DC {vin}

* Primary winding
L_pri input sw_drain {Lp:.6e} IC=0
* Secondary winding (dot convention: when sw_drain goes low, sec_dot goes positive)
L_sec sec_dot 0 {Ls:.6e} IC=0
K1 L_pri L_sec {k}

* Primary-side MOSFET switch
Vpwm pwm_ctrl 0 PULSE(0 5 0 1n 1n {ton:.10e} {period:.10e})
S1 sw_drain 0 pwm_ctrl 0 SMOD
.model SMOD SW(RON={R_dson} ROFF=1e6 VT=2.5 VH=0.1)

* Secondary-side diode (conducts when switch is off)
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
    sim_time = 500 * period
    meas_start = 400 * period
    tstep = period / 100

    return f"""\
* ARCS Forward Converter
* Vin={vin}V, Vout_target={vout_target}V, N={turns_ratio:.2f}, fsw={fsw/1e3:.0f}kHz

Vin input 0 DC {vin}

* Primary winding
L_pri input sw_drain {Lp:.6e} IC=0
* Secondary winding
L_sec 0 sec_out {Ls:.6e} IC=0
K1 L_pri L_sec {k}

* Primary-side MOSFET switch
Vpwm pwm_ctrl 0 PULSE(0 5 0 1n 1n {ton:.10e} {period:.10e})
S1 sw_drain 0 pwm_ctrl 0 SMOD
.model SMOD SW(RON={R_dson} ROFF=1e6 VT=2.5 VH=0.1)

* Secondary-side rectifier + freewheeling diode
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
# Tier 2: Analog Signal Processing — Op-Amp Circuits
# =============================================================================

# Ideal op-amp subcircuit used by all op-amp based templates.
# E1 gives a voltage-controlled voltage source with large gain.
_OPAMP_SUBCKT = """\
* Ideal op-amp subcircuit
.subckt IDEAL_OPAMP inp inn out
Rin inp inn 1e12
E1 out 0 inp inn 1e6
Rout out 0 1
.ends IDEAL_OPAMP
"""


def _inverting_amp_netlist(params: dict[str, float], conditions: dict[str, float]) -> str:
    """Inverting amplifier: Vout = -(Rf/Rin)*Vin."""
    vin_amp = conditions.get("vin_amp", 0.1)   # AC signal amplitude (V)
    freq_test = conditions.get("freq_test", 1e3)  # Test frequency (Hz)

    Rin = params["r_input"]
    Rf = params["r_feedback"]

    return f"""\
* ARCS Inverting Amplifier
* Gain target = -{Rf/Rin:.2f}, Vin_amp={vin_amp}V, f_test={freq_test/1e3:.1f}kHz

{_OPAMP_SUBCKT}

* AC input source
Vin inp 0 DC 0 AC {vin_amp}

* Input resistor
Rin inp vminus {Rin:.6e}

* Feedback resistor
Rf vminus vout {Rf:.6e}

* Op-amp: non-inv=GND (0), inv=vminus, out=vout
XU1 0 vminus vout IDEAL_OPAMP

* Sense node (for measurement consistency)
Rload vout 0 1e6

* === Analysis ===
.ac dec 100 1 100e6

* === Measurements ===
.measure AC gain_db FIND VDB(vout) AT={freq_test:.6e}
.measure AC gain_mag FIND VM(vout) AT={freq_test:.6e}
.measure AC phase_deg FIND VP(vout) AT={freq_test:.6e}
.measure AC bw_3db WHEN VDB(vout)=(gain_db-3) FALL=1
.measure AC gain_dc FIND VDB(vout) AT=1

.end
"""


def _noninverting_amp_netlist(params: dict[str, float], conditions: dict[str, float]) -> str:
    """Non-inverting amplifier: Vout = (1 + Rf/Rg)*Vin."""
    vin_amp = conditions.get("vin_amp", 0.1)
    freq_test = conditions.get("freq_test", 1e3)

    Rg = params["r_ground"]
    Rf = params["r_feedback"]

    return f"""\
* ARCS Non-Inverting Amplifier
* Gain target = {1 + Rf/Rg:.2f}, Vin_amp={vin_amp}V

{_OPAMP_SUBCKT}

Vin inp 0 DC 0 AC {vin_amp}

* Feedback network
Rf vout vminus {Rf:.6e}
Rg vminus 0 {Rg:.6e}

* Op-amp: non-inv=inp, inv=vminus, out=vout
XU1 inp vminus vout IDEAL_OPAMP

Rload vout 0 1e6

.ac dec 100 1 100e6

.measure AC gain_db FIND VDB(vout) AT={freq_test:.6e}
.measure AC gain_mag FIND VM(vout) AT={freq_test:.6e}
.measure AC phase_deg FIND VP(vout) AT={freq_test:.6e}
.measure AC bw_3db WHEN VDB(vout)=(gain_db-3) FALL=1
.measure AC gain_dc FIND VDB(vout) AT=1

.end
"""


def _instrumentation_amp_netlist(params: dict[str, float], conditions: dict[str, float]) -> str:
    """3 op-amp instrumentation amplifier. Gain = (1 + 2*R1/Rgain) * (R3/R2)."""
    vin_amp = conditions.get("vin_amp", 0.01)
    freq_test = conditions.get("freq_test", 1e3)

    R1 = params["r1"]
    Rgain = params["r_gain"]
    R2 = params["r2"]
    R3 = params["r3"]

    return f"""\
* ARCS Instrumentation Amplifier (3 Op-Amp)
* Gain = (1 + 2*{R1:.0f}/{Rgain:.0f}) * ({R3:.0f}/{R2:.0f})

{_OPAMP_SUBCKT}

* Differential input
Vip vinp 0 DC 0 AC {vin_amp}
Vin vinn 0 DC 0 AC -{vin_amp}

* === First stage: two non-inverting buffers with shared gain resistor ===
* U1: buffer for positive input
XU1 vinp u1inv u1out IDEAL_OPAMP
R1a u1out u1inv {R1:.6e}
Rgain u1inv u2inv {Rgain:.6e}
R1b u2inv u2out {R1:.6e}

* U2: buffer for negative input
XU2 vinn u2inv u2out IDEAL_OPAMP

* === Second stage: difference amplifier ===
R2a u1out diff_inv {R2:.6e}
R3a diff_inv vout {R3:.6e}
R2b u2out diff_noninv {R2:.6e}
R3b diff_noninv 0 {R3:.6e}

XU3 diff_noninv diff_inv vout IDEAL_OPAMP

Rload vout 0 1e6

.ac dec 100 1 100e6

.measure AC gain_db FIND VDB(vout) AT={freq_test:.6e}
.measure AC gain_mag FIND VM(vout) AT={freq_test:.6e}
.measure AC phase_deg FIND VP(vout) AT={freq_test:.6e}
.measure AC bw_3db WHEN VDB(vout)=(gain_db-3) FALL=1
.measure AC gain_dc FIND VDB(vout) AT=1

.end
"""


def _differential_amp_netlist(params: dict[str, float], conditions: dict[str, float]) -> str:
    """Single op-amp differential amplifier. Gain = R2/R1 (when matched)."""
    vin_amp = conditions.get("vin_amp", 0.1)
    freq_test = conditions.get("freq_test", 1e3)

    R1 = params["r1"]
    R2 = params["r2"]

    return f"""\
* ARCS Differential Amplifier (Single Op-Amp)
* Gain = {R2/R1:.2f}

{_OPAMP_SUBCKT}

* Differential input
Vip vinp 0 DC 0 AC {vin_amp}
Vin vinn 0 DC 0 AC -{vin_amp}

R1a vinp noninv {R1:.6e}
R2a noninv 0 {R2:.6e}
R1b vinn inv {R1:.6e}
R2b inv vout {R2:.6e}

XU1 noninv inv vout IDEAL_OPAMP

Rload vout 0 1e6

.ac dec 100 1 100e6

.measure AC gain_db FIND VDB(vout) AT={freq_test:.6e}
.measure AC gain_mag FIND VM(vout) AT={freq_test:.6e}
.measure AC phase_deg FIND VP(vout) AT={freq_test:.6e}
.measure AC bw_3db WHEN VDB(vout)=(gain_db-3) FALL=1
.measure AC gain_dc FIND VDB(vout) AT=1

.end
"""


# =============================================================================
# Tier 2: Analog Signal Processing — Active Filters
# =============================================================================

def _sallen_key_lowpass_netlist(params: dict[str, float], conditions: dict[str, float]) -> str:
    """Sallen-Key 2nd-order low-pass filter (unity gain).
    
    fc = 1/(2*pi*sqrt(R1*R2*C1*C2))
    Q  = sqrt(R1*R2*C1*C2) / (R1*C2 + R2*C2)  (for unity gain)
    """
    freq_test = conditions.get("freq_test", 1e3)

    R1 = params["r1"]
    R2 = params["r2"]
    C1 = params["c1"]
    C2 = params["c2"]

    return f"""\
* ARCS Sallen-Key Low-Pass Filter (2nd Order)
* fc_target ~ {1/(2*3.14159*((R1*R2*C1*C2)**0.5)):.1f} Hz

{_OPAMP_SUBCKT}

Vin inp 0 DC 0 AC 1

R1 inp n1 {R1:.6e}
R2 n1 noninv {R2:.6e}
C1 n1 vout {C1:.6e}
C2 noninv 0 {C2:.6e}

* Unity-gain buffer (output tied to inv input)
XU1 noninv vout vout IDEAL_OPAMP

Rload vout 0 1e6

.ac dec 100 1 100e6

.measure AC gain_dc FIND VDB(vout) AT=1
.measure AC gain_at_test FIND VDB(vout) AT={freq_test:.6e}
.measure AC phase_at_test FIND VP(vout) AT={freq_test:.6e}
.measure AC fc_3db WHEN VDB(vout)=(gain_dc-3) FALL=1

.end
"""


def _sallen_key_highpass_netlist(params: dict[str, float], conditions: dict[str, float]) -> str:
    """Sallen-Key 2nd-order high-pass filter (unity gain).
    
    fc = 1/(2*pi*sqrt(R1*R2*C1*C2))
    Swap R and C positions relative to low-pass.
    """
    freq_test = conditions.get("freq_test", 1e3)

    R1 = params["r1"]
    R2 = params["r2"]
    C1 = params["c1"]
    C2 = params["c2"]

    return f"""\
* ARCS Sallen-Key High-Pass Filter (2nd Order)

{_OPAMP_SUBCKT}

Vin inp 0 DC 0 AC 1

C1 inp n1 {C1:.6e}
C2 n1 noninv {C2:.6e}
R1 n1 0 {R1:.6e}
R2 noninv 0 {R2:.6e}
* Feed-forward from input to output through R2 path accounted for by topology
* Additional R from noninv to vout for proper Q
Rfb noninv vout 0.001

XU1 noninv vout vout IDEAL_OPAMP

Rload vout 0 1e6

.ac dec 100 1 100e6

.measure AC gain_passband FIND VDB(vout) AT=100e6
.measure AC gain_at_test FIND VDB(vout) AT={freq_test:.6e}
.measure AC phase_at_test FIND VP(vout) AT={freq_test:.6e}
.measure AC fc_3db WHEN VDB(vout)=(gain_passband-3) RISE=1

.end
"""


def _sallen_key_bandpass_netlist(params: dict[str, float], conditions: dict[str, float]) -> str:
    """Multiple feedback (MFB) band-pass filter (2nd order).
    
    A single op-amp band-pass with R1, R2, R3, C1, C2.
    Center freq: f0 = (1/2pi) * sqrt((1/R1 + 1/R3)/(R2*C1*C2))
    """
    freq_test = conditions.get("freq_test", 1e3)

    R1 = params["r1"]
    R2 = params["r2"]
    R3 = params["r3"]
    C1 = params["c1"]
    C2 = params["c2"]

    return f"""\
* ARCS MFB Band-Pass Filter (2nd Order)

{_OPAMP_SUBCKT}

Vin inp 0 DC 0 AC 1

R1 inp inv {R1:.6e}
C1 inv vout {C1:.6e}
R2 inv 0 {R2:.6e}
C2 inv n2 {C2:.6e}
R3 n2 vout {R3:.6e}

* Op-amp: non-inv=GND, inv=inv, out=vout
XU1 0 inv vout IDEAL_OPAMP

Rload vout 0 1e6

.ac dec 100 1 100e6

.measure AC gain_peak MAX VDB(vout)
.measure AC f_peak WHEN VDB(vout)=gain_peak RISE=1
.measure AC gain_at_test FIND VDB(vout) AT={freq_test:.6e}
.measure AC phase_at_test FIND VP(vout) AT={freq_test:.6e}
.measure AC bw_lo WHEN VDB(vout)=(gain_peak-3) RISE=1
.measure AC bw_hi WHEN VDB(vout)=(gain_peak-3) FALL=1

.end
"""


# =============================================================================
# Tier 2: Analog Signal Processing — Oscillators
# =============================================================================

def _wien_bridge_netlist(params: dict[str, float], conditions: dict[str, float]) -> str:
    """Wien bridge oscillator. f_osc = 1/(2*pi*R*C) (when R1=R2, C1=C2).
    
    Uses a non-inverting amplifier with gain = 3 (Rf = 2*Rg) at resonance.
    We use slightly higher gain for reliable startup.
    """
    R = params["r_freq"]
    C = params["c_freq"]
    Rf = params["r_feedback"]
    Rg = params["r_ground"]

    f_osc = 1.0 / (2 * np.pi * R * C)
    sim_time = max(50 / f_osc, 1e-3)  # At least 50 cycles
    meas_start = max(30 / f_osc, 0.5e-3)  # Skip startup transient

    return f"""\
* ARCS Wien Bridge Oscillator
* f_osc ~ {f_osc:.1f} Hz, Gain = {1 + Rf/Rg:.2f}

{_OPAMP_SUBCKT}

* Feedback network (Wien bridge): series RC from output to non-inv
R1 vout n1 {R:.6e}
C1 n1 noninv {C:.6e}
* Parallel RC from non-inv to ground
R2 noninv 0 {R:.6e}
C2 noninv 0 {C:.6e}

* Gain-setting resistors
Rf vout inv {Rf:.6e}
Rg inv 0 {Rg:.6e}

XU1 noninv inv vout IDEAL_OPAMP

* Small initial kick to start oscillation
Vkick noninv 0 PULSE(0.01 0 0 1n 1n 1n 1)

.tran {sim_time/5000:.10e} {sim_time:.10e} UIC

.measure TRAN vosc_pp PP V(vout) FROM={meas_start:.10e} TO={sim_time:.10e}
.measure TRAN vosc_avg AVG V(vout) FROM={meas_start:.10e} TO={sim_time:.10e}

.end
"""


def _colpitts_netlist(params: dict[str, float], conditions: dict[str, float]) -> str:
    """Colpitts oscillator using BJT common-emitter configuration.
    
    f_osc = 1 / (2*pi*sqrt(L * C1*C2/(C1+C2)))
    Uses capacitive voltage divider (C1, C2) + inductor in feedback.
    """
    Vcc = conditions.get("vcc", 12.0)

    L = params["inductance"]
    C1 = params["c1"]
    C2 = params["c2"]
    Rb1 = params["r_bias_1"]
    Rb2 = params["r_bias_2"]
    Re = params["r_emitter"]
    Rc = params["r_collector"]

    Cseries = C1 * C2 / (C1 + C2)
    f_osc = 1.0 / (2 * np.pi * (L * Cseries) ** 0.5)
    sim_time = max(100 / f_osc, 1e-3)
    meas_start = max(60 / f_osc, 0.5e-3)

    return f"""\
* ARCS Colpitts Oscillator (BJT)
* f_osc ~ {f_osc:.1f} Hz, Vcc={Vcc}V

Vcc vcc 0 DC {Vcc}

.model QNPN NPN(IS=1e-14 BF=200 VAF=100 CJC=5p CJE=10p TF=0.3n)

* Bias network
Rb1 vcc base {Rb1:.6e}
Rb2 base 0 {Rb2:.6e}

* BJT
Q1 collector base emitter QNPN

Rc vcc collector {Rc:.6e}
Re emitter 0 {Re:.6e}

* Tank circuit on collector: L + C1/C2 capacitive divider
L1 collector tank_mid {L:.6e} IC=0
C1 tank_mid base {C1:.6e}
C2 tank_mid 0 {C2:.6e}

* Bypass cap on emitter for AC ground
Ce emitter 0 100e-6

.tran {sim_time/5000:.10e} {sim_time:.10e} UIC

.measure TRAN vosc_pp PP V(collector) FROM={meas_start:.10e} TO={sim_time:.10e}
.measure TRAN vosc_avg AVG V(collector) FROM={meas_start:.10e} TO={sim_time:.10e}

.end
"""


# =============================================================================
# Component bounds for Tier 2 topologies
# =============================================================================

SIGNAL_CIRCUIT_BOUNDS = {
    "inverting_amp": [
        ComponentBounds("r_input", "Ω", 100, 1e6, log_scale=True, description="Input resistor"),
        ComponentBounds("r_feedback", "Ω", 100, 10e6, log_scale=True, description="Feedback resistor"),
    ],
    "noninverting_amp": [
        ComponentBounds("r_ground", "Ω", 100, 1e6, log_scale=True, description="Ground resistor"),
        ComponentBounds("r_feedback", "Ω", 100, 10e6, log_scale=True, description="Feedback resistor"),
    ],
    "instrumentation_amp": [
        ComponentBounds("r1", "Ω", 1e3, 100e3, log_scale=True, description="R1 (buffers)"),
        ComponentBounds("r_gain", "Ω", 10, 100e3, log_scale=True, description="Gain resistor"),
        ComponentBounds("r2", "Ω", 1e3, 100e3, log_scale=True, description="R2 (diff amp)"),
        ComponentBounds("r3", "Ω", 1e3, 100e3, log_scale=True, description="R3 (diff amp)"),
    ],
    "differential_amp": [
        ComponentBounds("r1", "Ω", 100, 1e6, log_scale=True, description="Input resistor"),
        ComponentBounds("r2", "Ω", 100, 10e6, log_scale=True, description="Feedback resistor"),
    ],
    "sallen_key_lowpass": [
        ComponentBounds("r1", "Ω", 100, 1e6, log_scale=True),
        ComponentBounds("r2", "Ω", 100, 1e6, log_scale=True),
        ComponentBounds("c1", "F", 10e-12, 10e-6, log_scale=True),
        ComponentBounds("c2", "F", 10e-12, 10e-6, log_scale=True),
    ],
    "sallen_key_highpass": [
        ComponentBounds("r1", "Ω", 100, 1e6, log_scale=True),
        ComponentBounds("r2", "Ω", 100, 1e6, log_scale=True),
        ComponentBounds("c1", "F", 10e-12, 10e-6, log_scale=True),
        ComponentBounds("c2", "F", 10e-12, 10e-6, log_scale=True),
    ],
    "sallen_key_bandpass": [
        ComponentBounds("r1", "Ω", 100, 1e6, log_scale=True),
        ComponentBounds("r2", "Ω", 100, 1e6, log_scale=True),
        ComponentBounds("r3", "Ω", 100, 1e6, log_scale=True),
        ComponentBounds("c1", "F", 10e-12, 10e-6, log_scale=True),
        ComponentBounds("c2", "F", 10e-12, 10e-6, log_scale=True),
    ],
    "wien_bridge": [
        ComponentBounds("r_freq", "Ω", 100, 100e3, log_scale=True, description="Frequency-setting R"),
        ComponentBounds("c_freq", "F", 100e-12, 10e-6, log_scale=True, description="Frequency-setting C"),
        ComponentBounds("r_feedback", "Ω", 1e3, 100e3, log_scale=True, description="Gain Rf"),
        ComponentBounds("r_ground", "Ω", 1e3, 100e3, log_scale=True, description="Gain Rg"),
    ],
    "colpitts": [
        ComponentBounds("inductance", "H", 1e-6, 10e-3, log_scale=True, description="Tank inductor"),
        ComponentBounds("c1", "F", 10e-12, 1e-6, log_scale=True, description="Tank C1"),
        ComponentBounds("c2", "F", 10e-12, 1e-6, log_scale=True, description="Tank C2"),
        ComponentBounds("r_bias_1", "Ω", 1e3, 1e6, log_scale=True, description="Bias Rb1"),
        ComponentBounds("r_bias_2", "Ω", 1e3, 1e6, log_scale=True, description="Bias Rb2"),
        ComponentBounds("r_emitter", "Ω", 10, 10e3, log_scale=True, description="Emitter R"),
        ComponentBounds("r_collector", "Ω", 100, 100e3, log_scale=True, description="Collector R"),
    ],
}


SIGNAL_OPERATING_CONDITIONS = {
    "inverting_amp": {"vin_amp": 0.1, "freq_test": 1e3},
    "noninverting_amp": {"vin_amp": 0.1, "freq_test": 1e3},
    "instrumentation_amp": {"vin_amp": 0.01, "freq_test": 1e3},
    "differential_amp": {"vin_amp": 0.1, "freq_test": 1e3},
    "sallen_key_lowpass": {"freq_test": 1e3},
    "sallen_key_highpass": {"freq_test": 1e3},
    "sallen_key_bandpass": {"freq_test": 1e3},
    "wien_bridge": {},
    "colpitts": {"vcc": 12.0},
}


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
    "cuk": {"vin": 12.0, "vout": -5.0, "iout": 1.0, "fsw": 100e3},
    "sepic": {"vin": 12.0, "vout": 5.0, "iout": 1.0, "fsw": 100e3},
    "flyback": {"vin": 12.0, "vout": 5.0, "iout": 1.0, "fsw": 100e3},
    "forward": {"vin": 48.0, "vout": 12.0, "iout": 2.0, "fsw": 100e3},
}


# ---------------------------------------------------------------------------
# Tier classification helpers
# ---------------------------------------------------------------------------

_TIER1_NAMES = ["buck", "boost", "buck_boost", "cuk", "sepic", "flyback", "forward"]

_TIER2_NAMES = [
    "inverting_amp", "noninverting_amp", "instrumentation_amp", "differential_amp",
    "sallen_key_lowpass", "sallen_key_highpass", "sallen_key_bandpass",
    "wien_bridge", "colpitts",
]

_TIER1_NETLIST_FNS = {
    "buck": _buck_netlist,
    "boost": _boost_netlist,
    "buck_boost": _buck_boost_netlist,
    "cuk": _cuk_netlist,
    "sepic": _sepic_netlist,
    "flyback": _flyback_netlist,
    "forward": _forward_netlist,
}

_TIER2_NETLIST_FNS = {
    "inverting_amp": _inverting_amp_netlist,
    "noninverting_amp": _noninverting_amp_netlist,
    "instrumentation_amp": _instrumentation_amp_netlist,
    "differential_amp": _differential_amp_netlist,
    "sallen_key_lowpass": _sallen_key_lowpass_netlist,
    "sallen_key_highpass": _sallen_key_highpass_netlist,
    "sallen_key_bandpass": _sallen_key_bandpass_netlist,
    "wien_bridge": _wien_bridge_netlist,
    "colpitts": _colpitts_netlist,
}

_ALL_NETLIST_FNS = {**_TIER1_NETLIST_FNS, **_TIER2_NETLIST_FNS}

_ALL_DESCRIPTIONS = {
    # Tier 1 — power converters
    "buck": "Step-down DC-DC converter",
    "boost": "Step-up DC-DC converter",
    "buck_boost": "Inverting buck-boost DC-DC converter",
    "cuk": "Ćuk DC-DC converter (non-inverting, two inductors)",
    "sepic": "SEPIC DC-DC converter (non-inverting, can step up or down)",
    "flyback": "Isolated flyback DC-DC converter",
    "forward": "Isolated forward DC-DC converter",
    # Tier 2 — signal processing
    "inverting_amp": "Inverting op-amp amplifier",
    "noninverting_amp": "Non-inverting op-amp amplifier",
    "instrumentation_amp": "3 op-amp instrumentation amplifier",
    "differential_amp": "Single op-amp differential amplifier",
    "sallen_key_lowpass": "Sallen-Key 2nd-order low-pass filter",
    "sallen_key_highpass": "Sallen-Key 2nd-order high-pass filter",
    "sallen_key_bandpass": "MFB 2nd-order band-pass filter",
    "wien_bridge": "Wien bridge oscillator",
    "colpitts": "Colpitts BJT oscillator",
}

# Metric names per domain
_POWER_METRIC_NAMES = ["vout_avg", "vout_ripple", "iout_avg", "iin_avg", "il_ripple", "pout", "pin"]
_AMP_METRIC_NAMES = ["gain_db", "gain_mag", "phase_deg", "bw_3db", "gain_dc"]
_FILTER_METRIC_NAMES = ["gain_dc", "gain_at_test", "phase_at_test", "fc_3db"]
_BANDPASS_METRIC_NAMES = ["gain_peak", "f_peak", "gain_at_test", "phase_at_test", "bw_lo", "bw_hi"]
_OSC_METRIC_NAMES = ["vosc_pp", "vosc_avg"]

_METRIC_MAP = {
    **{n: _POWER_METRIC_NAMES for n in _TIER1_NAMES},
    "inverting_amp": _AMP_METRIC_NAMES,
    "noninverting_amp": _AMP_METRIC_NAMES,
    "instrumentation_amp": _AMP_METRIC_NAMES,
    "differential_amp": _AMP_METRIC_NAMES,
    "sallen_key_lowpass": _FILTER_METRIC_NAMES,
    "sallen_key_highpass": ["gain_passband", "gain_at_test", "phase_at_test", "fc_3db"],
    "sallen_key_bandpass": _BANDPASS_METRIC_NAMES,
    "wien_bridge": _OSC_METRIC_NAMES,
    "colpitts": _OSC_METRIC_NAMES,
}

_ALL_BOUNDS = {**POWER_CONVERTER_BOUNDS, **SIGNAL_CIRCUIT_BOUNDS}
_ALL_CONDITIONS = {**OPERATING_CONDITIONS, **SIGNAL_OPERATING_CONDITIONS}


def get_topology(name: str) -> TopologyTemplate:
    """Get a topology template by name (supports all tiers)."""
    if name not in _ALL_NETLIST_FNS:
        raise ValueError(
            f"Unknown topology: {name}. Available: {sorted(_ALL_NETLIST_FNS.keys())}"
        )

    return TopologyTemplate(
        name=name,
        description=_ALL_DESCRIPTIONS[name],
        component_bounds=_ALL_BOUNDS[name],
        netlist_fn=_ALL_NETLIST_FNS[name],
        metric_names=_METRIC_MAP[name],
        operating_conditions=_ALL_CONDITIONS[name],
    )


def get_all_topologies(tier: int | None = None) -> list[TopologyTemplate]:
    """Get topology templates. tier=1 for power converters, tier=2 for signal, None for all."""
    if tier == 1:
        names = _TIER1_NAMES
    elif tier == 2:
        names = _TIER2_NAMES
    else:
        names = _TIER1_NAMES + _TIER2_NAMES
    return [get_topology(name) for name in names]
