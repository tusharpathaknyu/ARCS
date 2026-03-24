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

# Probe frequencies for AC bandwidth estimation (Hz).
# Used in _ac_measure_block() and must match datagen.BANDWIDTH_PROBE_FREQS.
BANDWIDTH_PROBE_FREQS: list[float] = [10, 100, 1e3, 10e3, 100e3, 1e6, 10e6, 50e6]


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
.measure TRAN iout_avg AVG par('-I(Vsense)') FROM={meas_start:.10e} TO={sim_time:.10e}
.measure TRAN iin_avg AVG par('-I(Vin)') FROM={meas_start:.10e} TO={sim_time:.10e}
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
.measure TRAN iout_avg AVG par('-I(Vsense)') FROM={meas_start:.10e} TO={sim_time:.10e}
.measure TRAN iin_avg AVG par('-I(Vin)') FROM={meas_start:.10e} TO={sim_time:.10e}
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
.measure TRAN iout_avg AVG par('-I(Vsense)') FROM={meas_start:.10e} TO={sim_time:.10e}
.measure TRAN iin_avg AVG par('-I(Vin)') FROM={meas_start:.10e} TO={sim_time:.10e}

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
.measure TRAN iout_avg AVG par('-I(Vsense)') FROM={meas_start:.10e} TO={sim_time:.10e}
.measure TRAN iin_avg AVG par('-I(Vin)') FROM={meas_start:.10e} TO={sim_time:.10e}

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
    sim_time = 1000 * period      # SEPIC needs more cycles: two coupled inductors slow transients
    meas_start = 800 * period
    tstep = period / 100

    return f"""\
* ARCS SEPIC Converter
* Vin={vin}V, Vout_target={vout_target}V, D={duty:.3f}, fsw={fsw/1e3:.0f}kHz

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
.measure TRAN iout_avg AVG par('-I(Vsense)') FROM={meas_start:.10e} TO={sim_time:.10e}
.measure TRAN iin_avg AVG par('-I(Vin)') FROM={meas_start:.10e} TO={sim_time:.10e}

.end
"""


def _flyback_netlist(params: dict[str, float], conditions: dict[str, float]) -> str:
    """Flyback converter with coupled inductor (transformer) model.

    Topology (N = turns_ratio = Np/Ns):
      - Switch ON:  Primary L_pri stores energy from Vin; secondary diode Dfb reverse-biased.
      - Switch OFF: Primary field collapses; reflected voltage Vin/N + Vout forward-biases Dfb;
                    energy transfers to Cout via secondary.
      - Clamp diode Dclamp (input → sw_drain) suppresses the primary spike when S1 opens,
        clamping sw_drain to Vin and recycling energy back to source.
    Vout = Vin * (Ns/Np) * D/(1-D) = Vin/N * D/(1-D)
    D = (Vout/Vin*N) / (1 + Vout/Vin*N)
    """
    vin = conditions.get("vin", 12.0)
    vout_target = conditions.get("vout", 5.0)
    iout = conditions.get("iout", 1.0)
    fsw = conditions.get("fsw", 100e3)

    Lp = params["inductance_primary"]
    turns_ratio = params["turns_ratio"]   # Np/Ns
    C_out = params["capacitance"]
    R_load = params.get("r_load", vout_target / iout)
    R_esr = params.get("esr", 0.02)
    R_dson = params.get("r_dson", 0.05)

    # Flyback duty: Vout = (Vin/N) * D/(1-D)  =>  D = (Vout*N) / (Vin + Vout*N)
    Vout_N = vout_target * turns_ratio   # effective secondary reflected voltage
    duty = max(0.05, min(0.85, Vout_N / (vin + Vout_N)))

    Ls = Lp / (turns_ratio ** 2)
    k = 0.98

    period = 1.0 / fsw
    ton = duty * period
    sim_time = 300 * period        # transformer with IC set reaches steady state by cycle 200
    meas_start = 200 * period
    tstep = period / 200           # finer step for spike capture

    return f"""\
* ARCS Flyback Converter
* Vin={vin}V, Vout_target={vout_target}V, N(Np/Ns)={turns_ratio:.2f}, D={duty:.3f}, fsw={fsw/1e3:.0f}kHz

Vin input 0 DC {vin}

* Primary winding
L_pri input sw_drain {Lp:.6e} IC=0
* Secondary winding (dot at sec_dot; conducts when S1 opens)
L_sec sec_dot 0 {Ls:.6e} IC=0
K1 L_pri L_sec {k}

* Primary-side MOSFET switch
Vpwm pwm_ctrl 0 PULSE(0 5 0 1n 1n {ton:.10e} {period:.10e})
S1 sw_drain 0 pwm_ctrl 0 SMOD
.model SMOD SW(RON={R_dson:.6e} ROFF=1e6 VT=2.5 VH=0.1)

* Primary clamp diode: recycles leakage energy, clamps sw_drain spike to Vin
* Conducts when sw_drain > Vin (S1 opens and primary field tries to fly high)
Dclamp sw_drain input DCLAMP
.model DCLAMP D(IS=1e-8 RS=0.05 N=1.02 BV=200 CJO=50p)

* Secondary-side rectifier (conducts when S1 is off and sec_dot > Vout)
Dfb sec_dot vout DSCHOTTKY
.model DSCHOTTKY D(IS=1e-6 RS=0.03 N=1.05 BV=100 CJO=100p)

Resr vout cap_node {R_esr:.6e}
C1 cap_node 0 {C_out:.6e} IC={vout_target}

Rload vout load_mid {R_load:.4f}
Vsense load_mid 0 DC 0

.tran {tstep:.10e} {sim_time:.10e} {meas_start:.10e} UIC

.measure TRAN vout_avg AVG V(vout) FROM={meas_start:.10e} TO={sim_time:.10e}
.measure TRAN vout_ripple PP V(vout) FROM={meas_start:.10e} TO={sim_time:.10e}
.measure TRAN iout_avg AVG par('-I(Vsense)') FROM={meas_start:.10e} TO={sim_time:.10e}
.measure TRAN iin_avg AVG par('-I(Vin)') FROM={meas_start:.10e} TO={sim_time:.10e}

.end
"""


def _forward_netlist(params: dict[str, float], conditions: dict[str, float]) -> str:
    """Forward converter template. Isolated buck derivative.

    Topology (N = turns_ratio = Np/Ns):
      - Switch ON:  L_pri magnetizes; L_sec delivers Vin/N to output through Dfwd and Lout.
      - Switch OFF: L_sec current freewheels through Dfw; L_pri MUST demagnetize.
      - Core reset: Tertiary winding L_ter (1:1 with primary) + Dreset clamp.
        When S1 opens, Dreset conducts and returns primary magnetizing energy to Vin,
        resetting core flux to zero within (1-D)/fsw seconds.
      - Duty limited to 50% for 1:1 reset winding to guarantee full demagnetization.
    Vout = Vin * (Ns/Np) * D = (Vin/N) * D
    D = Vout * N / Vin  (capped at 0.45 for reset margin)
    """
    vin = conditions.get("vin", 48.0)
    vout_target = conditions.get("vout", 12.0)
    iout = conditions.get("iout", 2.0)
    fsw = conditions.get("fsw", 100e3)

    Lp = params["inductance_primary"]
    turns_ratio = params["turns_ratio"]   # Np/Ns
    L_out = params["inductance_output"]
    C_out = params["capacitance"]
    R_load = params.get("r_load", vout_target / iout)
    R_esr = params.get("esr", 0.02)
    R_dson = params.get("r_dson", 0.05)

    # Forward: Vout = (Vin/N) * D  =>  D = Vout*N/Vin  (cap at 0.45 for reset winding margin)
    duty = max(0.05, min(0.45, vout_target * turns_ratio / vin))

    Ls = Lp / (turns_ratio ** 2)   # secondary inductance
    Lt = Lp                         # tertiary = primary (1:1 reset winding)
    k_ps = 0.98                     # primary-secondary coupling
    k_pt = 0.98                     # primary-tertiary coupling

    period = 1.0 / fsw
    ton = duty * period
    sim_time = 100 * period      # 100 cycles (3-winding transformer is expensive to simulate)
    meas_start = 60 * period     # steady-state from cycle 60 (IC on C1 = vout_target)
    tstep = period / 50          # coarser timestep for 3-winding mutual inductance

    return f"""\
* ARCS Forward Converter (with tertiary reset winding)
* Vin={vin}V, Vout_target={vout_target}V, N(Np/Ns)={turns_ratio:.2f}, D={duty:.3f}, fsw={fsw/1e3:.0f}kHz

Vin input 0 DC {vin}

* Primary winding
L_pri input sw_drain {Lp:.6e} IC=0
* Secondary winding (dot at sec_out; delivers power during ON time)
L_sec 0 sec_out {Ls:.6e} IC=0
* Tertiary reset winding (1:1 with primary; dot at reset_node same polarity as primary dot)
L_ter reset_node input {Lt:.6e} IC=0
K1 L_pri L_sec {k_ps}
K2 L_pri L_ter {k_pt}

* Primary-side MOSFET switch
Vpwm pwm_ctrl 0 PULSE(0 5 0 1n 1n {ton:.10e} {period:.10e})
S1 sw_drain 0 pwm_ctrl 0 SMOD
.model SMOD SW(RON={R_dson:.6e} ROFF=1e6 VT=2.5 VH=0.1)

* Tertiary reset diode: conducts when S1 opens, clamps primary and resets core
Dreset 0 reset_node DRESET
.model DRESET D(IS=1e-8 RS=0.05 N=1.02 BV=200 CJO=50p)

* Secondary-side rectifier
Dfwd sec_out rect_out DSCHOTTKY
.model DSCHOTTKY D(IS=1e-6 RS=0.03 N=1.05 BV=100 CJO=100p)

* Output freewheeling diode (allows Lout current to continue when Dfwd is off)
Dfw 0 rect_out DSCHOTTKY2
.model DSCHOTTKY2 D(IS=1e-6 RS=0.03 N=1.05 BV=100 CJO=100p)

* Output filter
Lout rect_out vout {L_out:.6e} IC=0
Resr vout cap_node {R_esr:.6e}
C1 cap_node 0 {C_out:.6e} IC={vout_target}

Rload vout load_mid {R_load:.4f}
Vsense load_mid 0 DC 0

.tran {tstep:.10e} {sim_time:.10e} {meas_start:.10e} UIC

.measure TRAN vout_avg AVG V(vout) FROM={meas_start:.10e} TO={sim_time:.10e}
.measure TRAN vout_ripple PP V(vout) FROM={meas_start:.10e} TO={sim_time:.10e}
.measure TRAN iout_avg AVG par('-I(Vsense)') FROM={meas_start:.10e} TO={sim_time:.10e}
.measure TRAN iin_avg AVG par('-I(Vin)') FROM={meas_start:.10e} TO={sim_time:.10e}

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


def _ac_measure_block(freq_test: float) -> str:
    """Common .measure block for AC analysis.

    Probes gain (VDB) at the test frequency, at DC (1 Hz), and at logarithmically
    spaced frequencies for bandwidth estimation.  No self-referencing measures —
    bandwidth / fc is computed in Python post-processing.

    NOTE: `.save v(vout)` is required so ngspice actually runs the AC analysis
    when subcircuits are present (without it, ngspice skips AC analysis with
    "no data saved" error).  VP() returns phase in **radians**.
    """
    probes = BANDWIDTH_PROBE_FREQS
    lines = [
        "* Force ngspice to save AC data (required for subcircuit nodes)",
        ".save v(vout)",
        "",
        f".measure AC gain_db FIND VDB(vout) AT={freq_test:.6e}",
        f".measure AC gain_mag FIND VM(vout) AT={freq_test:.6e}",
        f".measure AC phase_rad FIND VP(vout) AT={freq_test:.6e}",
        ".measure AC gain_dc FIND VDB(vout) AT=1",
    ]
    for i, f in enumerate(probes):
        lines.append(f".measure AC vdb_{i} FIND VDB(vout) AT={f:.6e}")
    return "\n".join(lines)


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
{_ac_measure_block(freq_test)}

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

{_ac_measure_block(freq_test)}

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

{_ac_measure_block(freq_test)}

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

{_ac_measure_block(freq_test)}

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

{_ac_measure_block(freq_test)}

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

{_ac_measure_block(freq_test)}

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

{_ac_measure_block(freq_test)}

.end
"""


# =============================================================================
# Tier 2: Analog Signal Processing — Oscillators
# =============================================================================

def _wien_bridge_netlist(params: dict[str, float], conditions: dict[str, float]) -> str:
    """Wien bridge oscillator. f_osc = 1/(2*pi*R*C) (when R1=R2, C1=C2).

    Uses a non-inverting amplifier with gain = 3 (Rf = 2*Rg) at resonance.
    Slightly higher gain for reliable startup. Uses a rail-limited opamp
    (behavioral source clipping at ±15V) to provide amplitude limiting,
    since oscillation requires nonlinear amplitude control.

    Initial kick: capacitor IC on C2 (parallel cap in Wien network) instead
    of a voltage source that would short the feedback path.
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

* Rail-limited ideal op-amp (clips at ±15V like real supply rails)
.subckt OPAMP_RAILED inp inn out
Rin inp inn 1e12
B1 out 0 V = min(15, max(-15, 1e5*V(inp,inn)))
.ends OPAMP_RAILED

* Feedback network (Wien bridge): series RC from output to non-inv
R1 vout n1 {R:.6e}
C1 n1 noninv {C:.6e}
* Parallel RC from non-inv to ground (IC on C2 provides startup kick)
R2 noninv 0 {R:.6e}
C2 noninv 0 {C:.6e} IC=0.1

* Gain-setting resistors
Rf vout inv {Rf:.6e}
Rg inv 0 {Rg:.6e}

XU1 noninv inv vout OPAMP_RAILED

.tran {sim_time/5000:.10e} {sim_time:.10e} UIC

.measure TRAN vosc_pp PP V(vout) FROM={meas_start:.10e} TO={sim_time:.10e}
.measure TRAN vosc_avg AVG V(vout) FROM={meas_start:.10e} TO={sim_time:.10e}

.end
"""


def _colpitts_netlist(params: dict[str, float], conditions: dict[str, float]) -> str:
    """Colpitts oscillator using BJT common-emitter configuration.

    Topology:
      - Tank: L from collector to tank_mid; C1 from tank_mid to base (feedback tap);
        C2 from tank_mid to ground (return path).
      - Feedback: AC voltage at tank_mid feeds base through C1 (capacitive divider).
      - Startup: IC=0.1V on C2 provides initial perturbation to kick oscillation.
      - Ce bypasses Re at AC, making emitter an AC ground for common-emitter gain.

    f_osc = 1/(2*pi*sqrt(L * Ceq))  where Ceq = C1*C2/(C1+C2)
    Oscillation condition: Rc/Re >= C2/C1 (loop gain >= 1)
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

    # Need enough cycles to confirm oscillation; cap total steps at ~50k
    sim_time = min(max(200 / f_osc, 5e-3), 50e-3)
    # Measure last 30% of sim (after transient startup)
    meas_start = 0.7 * sim_time
    # Step size: ≥50 pts/cycle but cap total steps at ~50k for speed
    tstep = max(1.0 / (f_osc * 50), sim_time / 50000)

    return f"""\
* ARCS Colpitts Oscillator (BJT)
* f_osc ~ {f_osc:.1f} Hz, Vcc={Vcc}V

Vcc vcc 0 DC {Vcc}

.model QNPN NPN(IS=1e-14 BF=200 VAF=100 CJC=5p CJE=10p TF=0.3n)

* Bias network: sets quiescent base voltage ~ Vcc*Rb2/(Rb1+Rb2)
Rb1 vcc base {Rb1:.6e}
Rb2 base 0 {Rb2:.6e}

* BJT common-emitter stage
Q1 collector base emitter QNPN

Rc vcc collector {Rc:.6e}
Re emitter 0 {Re:.6e}

* Tank circuit: L from collector to tap; C1 taps feedback to base; C2 returns to GND
L1 collector tank_mid {L:.6e} IC=0
C1 tank_mid base {C1:.6e} IC=0
C2 tank_mid 0 {C2:.6e} IC=0.1

* Emitter bypass: AC-grounds emitter for common-emitter gain
Ce emitter 0 10e-6

.tran {tstep:.10e} {sim_time:.10e} UIC

.measure TRAN vosc_pp PP V(collector) FROM={meas_start:.10e} TO={sim_time:.10e}
.measure TRAN vosc_avg AVG V(collector) FROM={meas_start:.10e} TO={sim_time:.10e}

.end
"""


# =============================================================================
# Tier 3: BJT Amplifier Topologies
# =============================================================================

def _common_emitter_netlist(params: dict[str, float], conditions: dict[str, float]) -> str:
    """Common emitter amplifier. Gain ≈ -R_collector / R_emitter (bypassed)."""
    vin_amp = conditions.get("vin_amp", 0.1)
    freq_test = conditions.get("freq_test", 1e3)
    vcc = conditions.get("vcc", 12.0)

    R_collector = params["r_collector"]
    R_base = params["r_base"]
    R_emitter = params["r_emitter"]
    C_bypass = params["c_bypass"]

    return f"""\
* ARCS Common Emitter Amplifier
* Vcc={vcc}V, Vin_amp={vin_amp}V, f_test={freq_test/1e3:.1f}kHz

Vcc vcc 0 DC {vcc}
Vin inp 0 DC 0 AC {vin_amp}
Cin inp base 1u
Rb vcc base {R_base:.6e}

.model QNPN NPN(IS=1e-15 BF=200 VAF=100 RB=100 CJE=20p CJC=10p TF=0.5n)
Q1 collector base emitter QNPN

Rc vcc collector {R_collector:.6e}
Re emitter 0 {R_emitter:.6e}
Ce emitter 0 {C_bypass:.6e}

Cout collector vout 1u
Rload vout 0 1e6

* === Analysis ===
.ac dec 100 1 100e6

* === Measurements ===
{_ac_measure_block(freq_test)}

.end
"""


def _common_collector_netlist(params: dict[str, float], conditions: dict[str, float]) -> str:
    """Common collector (emitter follower). Gain ≈ 1."""
    vin_amp = conditions.get("vin_amp", 0.1)
    freq_test = conditions.get("freq_test", 1e3)
    vcc = conditions.get("vcc", 12.0)

    R_base = params["r_base"]
    R_emitter = params["r_emitter"]

    return f"""\
* ARCS Common Collector (Emitter Follower)
* Vcc={vcc}V, Vin_amp={vin_amp}V, f_test={freq_test/1e3:.1f}kHz

Vcc vcc 0 DC {vcc}
Vin inp 0 DC 0 AC {vin_amp}
Cin inp base 1u
Rb vcc base {R_base:.6e}

.model QNPN NPN(IS=1e-15 BF=200 VAF=100 RB=100 CJE=20p CJC=10p TF=0.5n)
Q1 vcc base emitter QNPN

Re emitter 0 {R_emitter:.6e}

Cout emitter vout 1u
Rload vout 0 1e6

* === Analysis ===
.ac dec 100 1 100e6

* === Measurements ===
{_ac_measure_block(freq_test)}

.end
"""


def _common_base_netlist(params: dict[str, float], conditions: dict[str, float]) -> str:
    """Common base amplifier. Non-inverting, high-frequency performance."""
    vin_amp = conditions.get("vin_amp", 0.1)
    freq_test = conditions.get("freq_test", 1e3)
    vcc = conditions.get("vcc", 12.0)

    R_collector = params["r_collector"]
    R_emitter = params["r_emitter"]

    return f"""\
* ARCS Common Base Amplifier
* Vcc={vcc}V, Vin_amp={vin_amp}V, f_test={freq_test/1e3:.1f}kHz

Vcc vcc 0 DC {vcc}

* Bias: base DC-grounded through large resistor, AC-grounded via cap
Rbias vcc base 100e3
Rbias2 base 0 100e3
Cbase base 0 100u

.model QNPN NPN(IS=1e-15 BF=200 VAF=100 RB=100 CJE=20p CJC=10p TF=0.5n)
Q1 collector base emitter QNPN

Rc vcc collector {R_collector:.6e}
Re emitter 0 {R_emitter:.6e}

* Input at emitter through coupling cap
Vin inp 0 DC 0 AC {vin_amp}
Cin inp emitter 1u

Cout collector vout 1u
Rload vout 0 1e6

* === Analysis ===
.ac dec 100 1 100e6

* === Measurements ===
{_ac_measure_block(freq_test)}

.end
"""


def _cascode_netlist(params: dict[str, float], conditions: dict[str, float]) -> str:
    """Cascode amplifier. Q1 common-emitter, Q2 common-base stacked.

    Both transistors' base bias networks are parameterized so the data
    generator can explore a variety of operating points:
      - Q1 bias (Rb1_q1/Rb2_q1) sets V_base1 = Vcc * Rb2_q1/(Rb1_q1+Rb2_q1)
      - Q2 bias (R_bias1/R_bias2) sets V_base2 to keep Q2 in active region
        above Q1's collector (mid node).
    Re provides DC stability; Cbias2 AC-grounds Q2 base for common-base operation.
    """
    vin_amp = conditions.get("vin_amp", 0.1)
    freq_test = conditions.get("freq_test", 1e3)
    vcc = conditions.get("vcc", 12.0)

    R_collector = params["r_collector"]
    R_bias1 = params["r_bias1"]      # Q2 base bias upper
    R_bias2 = params["r_bias2"]      # Q2 base bias lower
    R_emitter = params["r_emitter"]
    # Q1 bias resistors — parameterized so the generator explores different operating points
    Rb1_q1 = params.get("r_bias_q1_1", 200e3)
    Rb2_q1 = params.get("r_bias_q1_2", 100e3)

    return f"""\
* ARCS Cascode Amplifier
* Vcc={vcc}V, Vin_amp={vin_amp}V, f_test={freq_test/1e3:.1f}kHz

Vcc vcc 0 DC {vcc}
Vin inp 0 DC 0 AC {vin_amp}
Cin inp base1 1u

* Q1 base bias (parameterized for operating-point sweep)
Rb1 vcc base1 {Rb1_q1:.6e}
Rb2 base1 0 {Rb2_q1:.6e}

.model QNPN NPN(IS=1e-15 BF=200 VAF=100 RB=100 CJE=20p CJC=10p TF=0.5n)

* Q1: common-emitter stage
Q1 mid base1 emitter QNPN
Re emitter 0 {R_emitter:.6e}

* Q2: common-base stage (base AC-grounded by Cbias2; sets V_mid operating point)
Rbias1 vcc base2 {R_bias1:.6e}
Rbias2 base2 0 {R_bias2:.6e}
Cbias2 base2 0 10u
Q2 collector base2 mid QNPN

Rc vcc collector {R_collector:.6e}

Cout collector vout 1u
Rload vout 0 100e3

* === Analysis ===
.ac dec 100 1 100e6

* === Measurements ===
{_ac_measure_block(freq_test)}

.end
"""


def _current_mirror_netlist(params: dict[str, float], conditions: dict[str, float]) -> str:
    """Current mirror. Two matched BJTs with emitter degeneration, R_ref sets reference current."""
    vcc = conditions.get("vcc", 12.0)

    R_ref = params["r_ref"]
    R_e = params["r_emitter"]

    return f"""\
* ARCS Current Mirror
* Vcc={vcc}V, R_ref={R_ref:.1f}Ω, R_e={R_e:.1f}Ω

Vcc vcc 0 DC {vcc}

.model QNPN NPN(IS=1e-15 BF=200 VAF=100 RB=100 CJE=20p CJC=10p TF=0.5n)

* Reference branch: R_ref → Q1 (diode-connected) → R_e
Rref vcc collector1 {R_ref:.6e}
Q1 collector1 collector1 emitter1 QNPN
Re1 emitter1 0 {R_e:.6e}

* Mirror branch: Q2 mirrors Q1 current → R_e
Q2 collector2 collector1 emitter2 QNPN
Re2 emitter2 0 {R_e:.6e}
Rload vcc collector2 {R_ref:.6e}

* Sense resistors for current measurement
Vsense_ref collector1 collector1_sense DC 0
Rsense1 collector1_sense 0 0.001
Vsense_out collector2 collector2_sense DC 0
Rsense2 collector2_sense 0 0.001

* === Analysis === (transient — consistent with all other topologies)
.tran 1u 1m 0.5m

* === Measurements ===
.measure TRAN iref AVG par('-I(Vsense_ref)') FROM=0.5m TO=1m
.measure TRAN iout AVG par('-I(Vsense_out)') FROM=0.5m TO=1m

.end
"""


# =============================================================================
# Tier 2: Filters — Additional Topologies
# =============================================================================

def _twin_t_notch_netlist(params: dict[str, float], conditions: dict[str, float]) -> str:
    """Twin-T notch filter.

    Two T-networks in parallel provide a deep null at f_notch = 1/(2*pi*R*C).
    Low-pass T: R1-C3-R2 with C1 to ground from midpoint.
    High-pass T: C1-R3-C2 with R3 to ground from midpoint.
    Unity-gain buffer (op-amp follower) at output.
    """
    vin_amp = conditions.get("vin_amp", 0.5)
    freq_test = conditions.get("freq_test", 1e3)

    R1 = params["r1"]
    R2 = params["r2"]
    R3 = params["r3"]
    C1 = params["c1"]
    C2 = params["c2"]
    C3 = params["c3"]

    return f"""\
* ARCS Twin-T Notch Filter
* Vin_amp={vin_amp}V, f_test={freq_test/1e3:.1f}kHz

{_OPAMP_SUBCKT}

Vin inp 0 DC 0 AC {vin_amp}

* Low-pass T path: inp -> R1 -> n1 -> R2 -> vout; C3 from n1 to ground
R1 inp n1 {R1:.6e}
R2 n1 n_opamp {R2:.6e}
C3 n1 0 {C3:.6e}

* High-pass T path: inp -> C1 -> n2 -> C2 -> vout; R3 from n2 to ground
C1 inp n2 {C1:.6e}
C2 n2 n_opamp {C2:.6e}
R3 n2 0 {R3:.6e}

* Unity-gain op-amp buffer
XU1 n_opamp vout vout IDEAL_OPAMP

Rload vout 0 1e6

* === Analysis ===
.ac dec 100 1 100e6

* === Measurements ===
{_ac_measure_block(freq_test)}

.end
"""


def _state_variable_filter_netlist(params: dict[str, float], conditions: dict[str, float]) -> str:
    """State variable filter (2nd order).

    Uses 3 op-amps to produce simultaneous LP, HP, and BP outputs.
    We measure at the LP output node (vout).
    f0 = 1/(2*pi*R3*C1) when R3=R4, C1=C2.
    Q = (1 + R2/R1) / 3.
    """
    freq_test = conditions.get("freq_test", 1e3)

    R1 = params["r1"]
    R2 = params["r2"]
    R3 = params["r3"]
    R4 = params["r4"]
    C1 = params["c1"]
    C2 = params["c2"]

    return f"""\
* ARCS State Variable Filter (2nd Order)
* f_test={freq_test/1e3:.1f}kHz

{_OPAMP_SUBCKT}

Vin inp 0 DC 0 AC 1

* Summing amplifier (op-amp 1): produces HP output
R1 inp sum_inv {R1:.6e}
R2 vout sum_inv {R2:.6e}
Rbp bp_out sum_inv {R1:.6e}
XU1 0 sum_inv hp_out IDEAL_OPAMP

* First integrator (op-amp 2): HP -> BP
R3 hp_out int1_inv {R3:.6e}
C1 int1_inv bp_out {C1:.6e}
XU2 0 int1_inv bp_out IDEAL_OPAMP

* Second integrator (op-amp 3): BP -> LP (vout)
R4 bp_out int2_inv {R4:.6e}
C2 int2_inv vout {C2:.6e}
XU3 0 int2_inv vout IDEAL_OPAMP

Rload vout 0 1e6

* === Analysis ===
.ac dec 100 1 100e6

* === Measurements ===
{_ac_measure_block(freq_test)}

.end
"""


# =============================================================================
# Tier 2: Oscillators — Additional Topologies
# =============================================================================

def _hartley_netlist(params: dict[str, float], conditions: dict[str, float]) -> str:
    """Hartley oscillator using BJT.

    f_osc = 1 / (2*pi*sqrt((L1+L2)*C1))
    Uses inductive voltage divider (L1, L2) + capacitor in tank.
    """
    Vcc = conditions.get("vcc", 12.0)

    L1 = params["inductance_1"]
    L2 = params["inductance_2"]
    C1 = params["c1"]
    Rb1 = params["r_bias_1"]
    Rb2 = params["r_bias_2"]

    L_total = L1 + L2
    f_osc = 1.0 / (2 * np.pi * (L_total * C1) ** 0.5)
    sim_time = max(100 / f_osc, 1e-3)
    meas_start = max(60 / f_osc, 0.5e-3)

    return f"""\
* ARCS Hartley Oscillator (BJT)
* f_osc ~ {f_osc:.1f} Hz, Vcc={Vcc}V

Vcc vcc 0 DC {Vcc}

.model QNPN NPN(IS=1e-14 BF=200 VAF=100 CJC=5p CJE=10p TF=0.3n)

* Bias network
Rb1 vcc base {Rb1:.6e}
Rb2 base 0 {Rb2:.6e}

* BJT
Q1 collector base emitter QNPN

* Collector load (RFC - radio frequency choke)
Lrfc vcc collector 10e-3

* Emitter resistor for DC bias
Re emitter 0 1e3

* Bypass cap on emitter
Ce emitter 0 100e-6

* Tank circuit: L1 + L2 inductive divider with capacitor
L1 collector tank_tap {L1:.6e} IC=0
L2 tank_tap 0 {L2:.6e} IC=0
C1 collector 0 {C1:.6e}

* Feedback tap from inductor midpoint to base via coupling cap
Cc tank_tap base 1e-6

.tran {sim_time/5000:.10e} {sim_time:.10e} UIC

.measure TRAN vosc_pp PP V(collector) FROM={meas_start:.10e} TO={sim_time:.10e}
.measure TRAN vosc_avg AVG V(collector) FROM={meas_start:.10e} TO={sim_time:.10e}

.end
"""


# =============================================================================
# Tier 2: Regulators
# =============================================================================

_REGULATOR_METRIC_NAMES = ["vout_avg", "vout_ripple", "iout_avg", "iin_avg"]


def _shunt_regulator_netlist(params: dict[str, float], conditions: dict[str, float]) -> str:
    """Zener-based shunt regulator.

    R_series drops voltage; zener diode clamps output to V_zener.
    """
    vcc = conditions.get("vcc", 12.0)
    v_zener = conditions.get("v_zener", 5.1)

    R_series = params["r_series"]
    R_load = params["r_load"]

    sim_time = 10e-3
    meas_start = 5e-3
    tstep = sim_time / 1000

    return f"""\
* ARCS Shunt Regulator (Zener-based)
* Vcc={vcc}V, Vz={v_zener}V

Vin input 0 DC {vcc}

* Series resistor
Rs input vout {R_series:.6e}

* Zener diode (modeled as ideal zener)
Dz 0 vout ZMOD
.model ZMOD D(IS=1e-14 BV={v_zener} IBV=5e-3 RS=1)

* Load resistor with current sense
Rload vout load_mid {R_load:.6e}
Vsense load_mid 0 DC 0

.tran {tstep:.10e} {sim_time:.10e} {meas_start:.10e}

.measure TRAN vout_avg AVG V(vout) FROM={meas_start:.10e} TO={sim_time:.10e}
.measure TRAN vout_ripple PP V(vout) FROM={meas_start:.10e} TO={sim_time:.10e}
.measure TRAN iout_avg AVG par('-I(Vsense)') FROM={meas_start:.10e} TO={sim_time:.10e}
.measure TRAN iin_avg AVG par('-I(Vin)') FROM={meas_start:.10e} TO={sim_time:.10e}

.end
"""


def _series_regulator_netlist(params: dict[str, float], conditions: dict[str, float]) -> str:
    """Series pass transistor regulator.

    Uses a BJT as series pass element with resistor divider feedback.
    R1 and R2 set the output voltage: Vout ~ Vref * (1 + R1/R2).
    """
    vcc = conditions.get("vcc", 12.0)
    v_ref = conditions.get("v_ref", 2.5)

    R1 = params["r1"]
    R2 = params["r2"]
    R_load = params["r_load"]

    sim_time = 10e-3
    meas_start = 5e-3
    tstep = sim_time / 1000

    return f"""\
* ARCS Series Pass Transistor Regulator
* Vcc={vcc}V, Vref={v_ref}V

Vin input 0 DC {vcc}

.model QNPN NPN(IS=1e-14 BF=200 VAF=100)

* Series pass transistor
Q1 input base vout QNPN

* Reference voltage
Vref ref 0 DC {v_ref}

* Error amplifier (op-amp compares feedback to reference)
* Feedback divider: Vfb = Vout * R2/(R1+R2)
R1 vout fb_node {R1:.6e}
R2 fb_node 0 {R2:.6e}

* Simple error amp driving base
Eamp base 0 ref fb_node 1000

* Load with current sense
Rload vout load_mid {R_load:.6e}
Vsense load_mid 0 DC 0

.tran {tstep:.10e} {sim_time:.10e} {meas_start:.10e}

.measure TRAN vout_avg AVG V(vout) FROM={meas_start:.10e} TO={sim_time:.10e}
.measure TRAN vout_ripple PP V(vout) FROM={meas_start:.10e} TO={sim_time:.10e}
.measure TRAN iout_avg AVG par('-I(Vsense)') FROM={meas_start:.10e} TO={sim_time:.10e}
.measure TRAN iin_avg AVG par('-I(Vin)') FROM={meas_start:.10e} TO={sim_time:.10e}

.end
"""


# =============================================================================
# Tier 2: Analog Signal Processing — Additional Op-Amp Circuits
# =============================================================================

def _phase_shift_netlist(params: dict[str, float], conditions: dict[str, float]) -> str:
    """3-stage RC phase shift oscillator with op-amp.

    Each RC stage shifts 60° at the oscillation frequency.
    Gain must be >= 29 for sustained oscillation (Rf = 29*R).
    f_osc = 1 / (2*pi*R*C*sqrt(6)) when R1=R2=R3, C1=C2=C3.
    """
    R1 = params["r1"]
    R2 = params["r2"]
    R3 = params["r3"]
    C1 = params["c1"]
    C2 = params["c2"]
    C3 = params["c3"]

    # Estimate oscillation frequency for sim timing
    R_avg = (R1 + R2 + R3) / 3
    C_avg = (C1 + C2 + C3) / 3
    f_osc = 1.0 / (2 * np.pi * R_avg * C_avg * np.sqrt(6))
    sim_time = max(50 / f_osc, 1e-3)
    meas_start = max(30 / f_osc, 0.5e-3)

    # Gain resistor: Rf = 29*R1 for oscillation condition
    Rf = 29 * R1

    return f"""\
* ARCS Phase Shift Oscillator (3-stage RC)
* f_osc ~ {f_osc:.1f} Hz, Gain = 29

* Rail-limited ideal op-amp (clips at +-15V like real supply rails)
.subckt OPAMP_RAILED inp inn out
Rin inp inn 1e12
B1 out 0 V = min(15, max(-15, 1e5*V(inp,inn)))
.ends OPAMP_RAILED

* Feedback from output through 3 RC stages
R1 vout n1 {R1:.6e}
C1 n1 0 {C1:.6e}
R2 n1 n2 {R2:.6e}
C2 n2 0 {C2:.6e}
R3 n2 vminus {R3:.6e}
C3 vminus 0 {C3:.6e}

* Op-amp with gain (Rf/Rin = 29 for oscillation)
Rf vout vminus {Rf:.6e}

* Op-amp: non-inv=GND (0), inv=vminus, out=vout
XU1 0 vminus vout OPAMP_RAILED

Rload vout 0 1e6

* Initial kick through high-impedance to avoid shorting vminus
Rkick kick_node vminus 100e3
Vkick kick_node 0 PULSE(1 0 0 1n 1n 1u 1)

.tran {sim_time/5000:.10e} {sim_time:.10e} UIC

.measure TRAN vosc_pp PP V(vout) FROM={meas_start:.10e} TO={sim_time:.10e}
.measure TRAN vosc_avg AVG V(vout) FROM={meas_start:.10e} TO={sim_time:.10e}

.end
"""


def _inverting_summing_amp_netlist(params: dict[str, float], conditions: dict[str, float]) -> str:
    """Inverting summing amplifier with 2 inputs.

    Vout = -(Rf/R1)*V1 - (Rf/R2)*V2
    """
    vin_amp = conditions.get("vin_amp", 0.1)
    freq_test = conditions.get("freq_test", 1e3)

    R1 = params["r_input1"]
    R2 = params["r_input2"]
    Rf = params["r_feedback"]

    return f"""\
* ARCS Inverting Summing Amplifier (2-input)
* Gain1 = -{Rf/R1:.2f}, Gain2 = -{Rf/R2:.2f}

{_OPAMP_SUBCKT}

* AC input sources
Vin1 inp1 0 DC 0 AC {vin_amp}
Vin2 inp2 0 DC 0 AC {vin_amp}

* Input resistors
R1 inp1 vminus {R1:.6e}
R2 inp2 vminus {R2:.6e}

* Feedback resistor
Rf vminus vout {Rf:.6e}

* Op-amp: non-inv=GND (0), inv=vminus, out=vout
XU1 0 vminus vout IDEAL_OPAMP

Rload vout 0 1e6

* === Analysis ===
.ac dec 100 1 100e6

* === Measurements ===
{_ac_measure_block(freq_test)}

.end
"""


def _transimpedance_amp_netlist(params: dict[str, float], conditions: dict[str, float]) -> str:
    """Transimpedance amplifier. Current input -> voltage output.

    Rf and Cf in parallel as feedback. Vout = -Iin * Zf
    where Zf = Rf || (1/jwCf).
    """
    freq_test = conditions.get("freq_test", 1e3)

    Rf = params["r_feedback"]
    Cf = params["c_feedback"]

    return f"""\
* ARCS Transimpedance Amplifier
* Rf={Rf:.1f} Ohm, Cf={Cf:.2e} F

{_OPAMP_SUBCKT}

* Current source input (1uA AC)
Iin 0 vminus DC 0 AC 1e-6

* Feedback network: Rf || Cf
Rf vminus vout {Rf:.6e}
Cf vminus vout {Cf:.6e}

* Op-amp: non-inv=GND (0), inv=vminus, out=vout
XU1 0 vminus vout IDEAL_OPAMP

Rload vout 0 1e6

* === Analysis ===
.ac dec 100 1 100e6

* === Measurements ===
{_ac_measure_block(freq_test)}

.end
"""


# =============================================================================
# Tier 1: Power Electronics — Additional Converters
# =============================================================================

def _half_bridge_netlist(params: dict[str, float], conditions: dict[str, float]) -> str:
    """Half-bridge converter. Vout ~ D * Vin.

    High-side and low-side MOSFETs with complementary PWM drive,
    output LC filter for smoothing.
    """
    vin = conditions.get("vin", 48.0)
    vout_target = conditions.get("vout", 24.0)
    iout = conditions.get("iout", 2.0)
    fsw = conditions.get("fsw", 100e3)

    R_dson_hi = params["r_dson_high"]
    R_dson_lo = params["r_dson_low"]
    L = params["inductance"]
    C = params["capacitance"]
    R_load = params.get("r_load", vout_target / iout)
    duty = max(0.05, min(0.95, vout_target / vin))

    period = 1.0 / fsw
    dead = period * 0.02  # 2% dead time
    ton_hi = duty * period - dead
    ton_lo = (1 - duty) * period - dead

    sim_time = 500 * period
    meas_start = 400 * period
    tstep = period / 100

    return f"""\
* ARCS Half-Bridge Converter
* Vin={vin}V, Vout_target={vout_target}V, Iout={iout}A, fsw={fsw/1e3:.0f}kHz

Vin input 0 DC {vin}

* High-side switch
S1 input sw_mid phase_hi 0 SMOD_HI
.model SMOD_HI SW(RON={R_dson_hi} ROFF=1e6 VT=2.5 VH=0.1)

* Low-side switch
S2 sw_mid 0 phase_lo 0 SMOD_LO
.model SMOD_LO SW(RON={R_dson_lo} ROFF=1e6 VT=2.5 VH=0.1)

* Dead-time complementary drive signals
Vphi_hi phase_hi 0 PULSE(0 5 {dead:.10e} 1n 1n {ton_hi:.10e} {period:.10e})
Vphi_lo phase_lo 0 PULSE(0 5 {ton_hi + dead:.10e} 1n 1n {ton_lo:.10e} {period:.10e})

* Output LC filter
L1 sw_mid vout {L:.6e} IC=0
C1 vout 0 {C:.6e} IC={vout_target}

* Load with current sense
Rload vout load_mid {R_load:.4f}
Vsense load_mid 0 DC 0

* === Analysis ===
.tran {tstep:.10e} {sim_time:.10e} {meas_start:.10e} UIC

* === Measurements ===
.measure TRAN vout_avg AVG V(vout) FROM={meas_start:.10e} TO={sim_time:.10e}
.measure TRAN vout_ripple PP V(vout) FROM={meas_start:.10e} TO={sim_time:.10e}
.measure TRAN iout_avg AVG par('-I(Vsense)') FROM={meas_start:.10e} TO={sim_time:.10e}
.measure TRAN iin_avg AVG par('-I(Vin)') FROM={meas_start:.10e} TO={sim_time:.10e}
.measure TRAN il_ripple PP I(L1) FROM={meas_start:.10e} TO={sim_time:.10e}

.end
"""


def _push_pull_netlist(params: dict[str, float], conditions: dict[str, float]) -> str:
    """Push-pull converter. Two switches alternate through a center-tapped transformer.

    Vout ~ 2 * D * Vin / turns_ratio. Each switch conducts for up to 50% duty.
    """
    vin = conditions.get("vin", 48.0)
    vout_target = conditions.get("vout", 12.0)
    iout = conditions.get("iout", 2.0)
    fsw = conditions.get("fsw", 100e3)

    N = params["turns_ratio"]
    R_dson = params["r_dson"]
    C = params["capacitance"]
    L = params["inductance"]
    R_load = params.get("r_load", vout_target / iout)

    # Each switch conducts up to 50%; effective duty per half-cycle
    duty = max(0.05, min(0.48, vout_target * N / (2.0 * vin)))

    period = 1.0 / fsw
    ton = duty * period
    sim_time = 500 * period
    meas_start = 400 * period
    tstep = period / 100

    return f"""\
* ARCS Push-Pull Converter
* Vin={vin}V, Vout_target={vout_target}V, N={N:.2f}, fsw={fsw/1e3:.0f}kHz

Vin input 0 DC {vin}

* Switch 1 (first half-cycle)
S1 input sw1 phase1 0 SMOD
.model SMOD SW(RON={R_dson} ROFF=1e6 VT=2.5 VH=0.1)

* Switch 2 (second half-cycle, 180 degree phase shift)
S2 input sw2 phase2 0 SMOD

Vphi1 phase1 0 PULSE(0 5 0 1n 1n {ton:.10e} {period:.10e})
Vphi2 phase2 0 PULSE(0 5 {period/2:.10e} 1n 1n {ton:.10e} {period:.10e})

* Transformer modeled as ideal turns ratio with R_dson losses
* Simplified: rectified secondary = Vin * duty / N
* Use behavioral source for secondary voltage
Rpri1 sw1 0 1e3
Rpri2 sw2 0 1e3
Bsec rect_out 0 V = (V(sw1) - V(sw2)) / {N:.4f}

* Full-wave rectification via diode
D1 rect_out rect_filt DSCHOTTKY
.model DSCHOTTKY D(IS=1e-6 RS=0.03 N=1.05 BV=100 CJO=100p)
Rfreewheel rect_filt 0 1e6

* Output LC filter
L1 rect_filt vout {L:.6e} IC=0
C1 vout 0 {C:.6e} IC={vout_target}

* Load with current sense
Rload vout load_mid {R_load:.4f}
Vsense load_mid 0 DC 0

* === Analysis ===
.tran {tstep:.10e} {sim_time:.10e} {meas_start:.10e} UIC

* === Measurements ===
.measure TRAN vout_avg AVG V(vout) FROM={meas_start:.10e} TO={sim_time:.10e}
.measure TRAN vout_ripple PP V(vout) FROM={meas_start:.10e} TO={sim_time:.10e}
.measure TRAN iout_avg AVG par('-I(Vsense)') FROM={meas_start:.10e} TO={sim_time:.10e}
.measure TRAN iin_avg AVG par('-I(Vin)') FROM={meas_start:.10e} TO={sim_time:.10e}
.measure TRAN il_ripple PP I(L1) FROM={meas_start:.10e} TO={sim_time:.10e}

.end
"""


def _charge_pump_netlist(params: dict[str, float], conditions: dict[str, float]) -> str:
    """Charge pump (switched-capacitor) voltage doubler.

    Flying capacitor alternately charges from Vin and dumps charge to output.
    Vout ~ 2 * Vin for ideal doubler; losses from ESR and switch resistance.
    """
    vin = conditions.get("vin", 5.0)
    iout = conditions.get("iout", 0.1)
    fsw = conditions.get("fsw", 100e3)

    C_fly = params["c_flying"]
    C_out = params["c_output"]
    R_load_param = params["r_load"]
    R_esr = params["r_esr"]

    R_load = params.get("r_load", vin * 2.0 / max(iout, 0.001))

    period = 1.0 / fsw
    ton = 0.45 * period  # slightly less than 50% for dead time
    sim_time = 500 * period
    meas_start = 400 * period
    tstep = period / 100

    return f"""\
* ARCS Charge Pump (Voltage Doubler)
* Vin={vin}V, fsw={fsw/1e3:.0f}kHz

Vin input 0 DC {vin}

* Phase 1: charge flying cap from Vin
* Phase 2: stack flying cap on top of Vin to output

* Switches modeled as voltage-controlled switches
S1 input fly_p phase1 0 SMOD
S2 0 fly_n phase1 0 SMOD
S3 input fly_n phase2 0 SMOD
S4 fly_p cap_node phase2 0 SMOD
.model SMOD SW(RON=0.5 ROFF=1e6 VT=2.5 VH=0.1)

Vphi1 phase1 0 PULSE(0 5 0 1n 1n {ton:.10e} {period:.10e})
Vphi2 phase2 0 PULSE(0 5 {period/2:.10e} 1n 1n {ton:.10e} {period:.10e})

* Flying capacitor with ESR
Resr_fly fly_p fly_p2 {R_esr:.6e}
Cfly fly_p2 fly_n {C_fly:.6e}

* Output capacitor
Cout cap_node 0 {C_out:.6e} IC={vin * 2}

* Load with current sense
Rload cap_node load_mid {R_load:.4f}
Vsense load_mid 0 DC 0

* === Analysis ===
.tran {tstep:.10e} {sim_time:.10e} {meas_start:.10e} UIC

* === Measurements ===
.measure TRAN vout_avg AVG V(cap_node) FROM={meas_start:.10e} TO={sim_time:.10e}
.measure TRAN vout_ripple PP V(cap_node) FROM={meas_start:.10e} TO={sim_time:.10e}
.measure TRAN iout_avg AVG par('-I(Vsense)') FROM={meas_start:.10e} TO={sim_time:.10e}
.measure TRAN iin_avg AVG par('-I(Vin)') FROM={meas_start:.10e} TO={sim_time:.10e}

.end
"""


def _voltage_doubler_netlist(params: dict[str, float], conditions: dict[str, float]) -> str:
    """Voltage doubler rectifier. Two diodes and two capacitors.

    Villard/Greinacher topology: each cap charges to Vin through its diode,
    series-stacked output ~ 2*Vin minus diode drops.
    """
    vin = conditions.get("vin", 12.0)
    iout = conditions.get("iout", 0.1)
    fsw = conditions.get("fsw", 50e3)

    C1 = params["c1"]
    C2 = params["c2"]
    R_d1 = params["r_diode1"]
    R_d2 = params["r_diode2"]

    R_load = params.get("r_load", vin * 2.0 / max(iout, 0.001))

    period = 1.0 / fsw
    sim_time = 500 * period
    meas_start = 400 * period
    tstep = period / 100

    return f"""\
* ARCS Voltage Doubler (Villard)
* Vin_pk={vin}V, fsw={fsw/1e3:.0f}kHz

* AC input (square wave approximating switched input)
Vin input 0 PULSE({-vin} {vin} 0 1n 1n {period/2:.10e} {period:.10e})

* D1 charges C1 on negative half-cycle
D1 0 mid DMOD
.model DMOD D(IS=1e-6 RS={R_d1} N=1.05 BV=100 CJO=100p)
C1 input mid {C1:.6e} IC={vin}

* D2 charges C2 on positive half-cycle (stacked)
D2 mid vout DMOD2
.model DMOD2 D(IS=1e-6 RS={R_d2} N=1.05 BV=100 CJO=100p)
C2 vout 0 {C2:.6e} IC={vin * 2}

* Load with current sense
Rload vout load_mid {R_load:.4f}
Vsense load_mid 0 DC 0

* === Analysis ===
.tran {tstep:.10e} {sim_time:.10e} {meas_start:.10e} UIC

* === Measurements ===
.measure TRAN vout_avg AVG V(vout) FROM={meas_start:.10e} TO={sim_time:.10e}
.measure TRAN vout_ripple PP V(vout) FROM={meas_start:.10e} TO={sim_time:.10e}
.measure TRAN iout_avg AVG par('-I(Vsense)') FROM={meas_start:.10e} TO={sim_time:.10e}
.measure TRAN iin_avg AVG par('-I(Vin)') FROM={meas_start:.10e} TO={sim_time:.10e}

.end
"""


def _zeta_converter_netlist(params: dict[str, float], conditions: dict[str, float]) -> str:
    """Zeta (inverse-SEPIC) converter.

    Non-inverting, can step up or down. Dual of the SEPIC:
    - Series switch (S1) from input to switch node
    - L1 from switch node to ground (stores energy when S1 on)
    - Coupling cap (Cc) from switch node to coupling node
    - Diode from ground to coupling node (freewheeling)
    - L2 from coupling node to output

    Vout/Vin = D/(1-D), same transfer function as SEPIC.
    Switch ON: Vin charges L1 through S1, Cc discharges through L2 to output.
    Switch OFF: L1 energy transfers through D to charge Cc and output.
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
* ARCS Zeta Converter (Inverse-SEPIC)
* Vin={vin}V, Vout_target={vout_target}V, fsw={fsw/1e3:.0f}kHz

Vin input 0 DC {vin}

* Series MOSFET switch from input
Vpwm pwm_ctrl 0 PULSE(0 5 0 1n 1n {ton:.10e} {period:.10e})
S1 input sw_node pwm_ctrl 0 SMOD
.model SMOD SW(RON={R_dson} ROFF=1e6 VT=2.5 VH=0.1)

* Input inductor from switch node to ground
L1 sw_node 0 {L1:.6e} IC=0

* Coupling capacitor from switch node to coupling node
Cc sw_node cb {C_couple:.6e} IC={vin}

* Diode from ground to coupling node (freewheeling)
Dzeta 0 cb DSCHOTTKY
.model DSCHOTTKY D(IS=1e-6 RS=0.03 N=1.05 BV=60 CJO=100p)

* Output inductor from coupling node to output
L2 cb vout {L2:.6e} IC=0

* Output cap and load
Resr vout cap_node {R_esr}
C1 cap_node 0 {C_out:.6e} IC={vout_target}

Rload vout load_mid {R_load:.4f}
Vsense load_mid 0 DC 0

.tran {tstep:.10e} {sim_time:.10e} {meas_start:.10e} UIC

.measure TRAN vout_avg AVG V(vout) FROM={meas_start:.10e} TO={sim_time:.10e}
.measure TRAN vout_ripple PP V(vout) FROM={meas_start:.10e} TO={sim_time:.10e}
.measure TRAN iout_avg AVG par('-I(Vsense)') FROM={meas_start:.10e} TO={sim_time:.10e}
.measure TRAN iin_avg AVG par('-I(Vin)') FROM={meas_start:.10e} TO={sim_time:.10e}

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
        ComponentBounds("r_freq", "Ω", 1e3, 100e3, log_scale=True, description="Frequency-setting R"),
        ComponentBounds("c_freq", "F", 1e-9, 1e-6, log_scale=True, description="Frequency-setting C"),
        ComponentBounds("r_feedback", "Ω", 4e3, 100e3, log_scale=True, description="Gain Rf (need Rf >= 2*Rg for oscillation)"),
        ComponentBounds("r_ground", "Ω", 1e3, 33e3, log_scale=True, description="Gain Rg (gain = 1 + Rf/Rg >= 3)"),
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
    # ---- Tier 3: BJT Amplifiers ----
    "common_emitter": [
        ComponentBounds("r_collector", "Ω", 100, 100e3, log_scale=True, description="Collector resistor"),
        ComponentBounds("r_base", "Ω", 1e3, 1e6, log_scale=True, description="Base bias resistor"),
        ComponentBounds("r_emitter", "Ω", 10, 10e3, log_scale=True, description="Emitter resistor"),
        ComponentBounds("c_bypass", "F", 1e-9, 100e-6, log_scale=True, description="Emitter bypass cap"),
    ],
    "common_collector": [
        ComponentBounds("r_base", "Ω", 1e3, 1e6, log_scale=True, description="Base bias resistor"),
        ComponentBounds("r_emitter", "Ω", 100, 100e3, log_scale=True, description="Emitter resistor"),
    ],
    "common_base": [
        ComponentBounds("r_collector", "Ω", 100, 100e3, log_scale=True, description="Collector resistor"),
        ComponentBounds("r_emitter", "Ω", 10, 10e3, log_scale=True, description="Emitter resistor"),
    ],
    "cascode": [
        ComponentBounds("r_collector", "Ω", 1e3, 100e3, log_scale=True, description="Collector resistor"),
        ComponentBounds("r_bias1", "Ω", 10e3, 500e3, log_scale=True, description="Q2 bias R1 (upper)"),
        ComponentBounds("r_bias2", "Ω", 5e3, 200e3, log_scale=True, description="Q2 bias R2 (lower)"),
        ComponentBounds("r_emitter", "Ω", 100, 5e3, log_scale=True, description="Emitter degeneration"),
        ComponentBounds("r_bias_q1_1", "Ω", 50e3, 500e3, log_scale=True, description="Q1 bias R1 (upper)"),
        ComponentBounds("r_bias_q1_2", "Ω", 20e3, 200e3, log_scale=True, description="Q1 bias R2 (lower)"),
    ],
    "current_mirror": [
        ComponentBounds("r_ref", "Ω", 100, 100e3, log_scale=True, description="Reference resistor"),
        ComponentBounds("r_emitter", "Ω", 1, 1000, log_scale=True, description="Emitter degeneration resistor"),
    ],
    # ---- Additional Filters ----
    "twin_t_notch": [
        ComponentBounds("r1", "Ω", 100, 1e6, log_scale=True, description="Low-pass T R1"),
        ComponentBounds("r2", "Ω", 100, 1e6, log_scale=True, description="Low-pass T R2"),
        ComponentBounds("r3", "Ω", 100, 1e6, log_scale=True, description="High-pass T R3"),
        ComponentBounds("c1", "F", 10e-12, 10e-6, log_scale=True, description="High-pass T C1"),
        ComponentBounds("c2", "F", 10e-12, 10e-6, log_scale=True, description="High-pass T C2"),
        ComponentBounds("c3", "F", 10e-12, 10e-6, log_scale=True, description="Low-pass T C3"),
    ],
    "state_variable_filter": [
        ComponentBounds("r1", "Ω", 100, 1e6, log_scale=True, description="Summing R1"),
        ComponentBounds("r2", "Ω", 100, 1e6, log_scale=True, description="Feedback R2"),
        ComponentBounds("r3", "Ω", 100, 1e6, log_scale=True, description="Integrator R3"),
        ComponentBounds("r4", "Ω", 100, 1e6, log_scale=True, description="Integrator R4"),
        ComponentBounds("c1", "F", 10e-12, 10e-6, log_scale=True, description="Integrator C1"),
        ComponentBounds("c2", "F", 10e-12, 10e-6, log_scale=True, description="Integrator C2"),
    ],
    # ---- Additional Oscillators ----
    "hartley": [
        ComponentBounds("inductance_1", "H", 1e-6, 10e-3, log_scale=True, description="Tank L1"),
        ComponentBounds("inductance_2", "H", 1e-6, 10e-3, log_scale=True, description="Tank L2"),
        ComponentBounds("c1", "F", 10e-12, 1e-6, log_scale=True, description="Tank capacitor"),
        ComponentBounds("r_bias_1", "Ω", 1e3, 1e6, log_scale=True, description="Bias Rb1"),
        ComponentBounds("r_bias_2", "Ω", 1e3, 1e6, log_scale=True, description="Bias Rb2"),
    ],
    "phase_shift": [
        ComponentBounds("r1", "Ω", 100, 1e6, log_scale=True, description="Phase R1"),
        ComponentBounds("r2", "Ω", 100, 1e6, log_scale=True, description="Phase R2"),
        ComponentBounds("r3", "Ω", 100, 1e6, log_scale=True, description="Phase R3"),
        ComponentBounds("c1", "F", 10e-12, 10e-6, log_scale=True, description="Phase C1"),
        ComponentBounds("c2", "F", 10e-12, 10e-6, log_scale=True, description="Phase C2"),
        ComponentBounds("c3", "F", 10e-12, 10e-6, log_scale=True, description="Phase C3"),
    ],
    # ---- Regulators ----
    "shunt_regulator": [
        ComponentBounds("r_series", "Ω", 1, 10e3, log_scale=True, description="Series resistor"),
        ComponentBounds("r_load", "Ω", 10, 100e3, log_scale=True, description="Load resistor"),
    ],
    "series_regulator": [
        ComponentBounds("r1", "Ω", 100, 1e6, log_scale=True, description="Feedback R1"),
        ComponentBounds("r2", "Ω", 100, 1e6, log_scale=True, description="Feedback R2"),
        ComponentBounds("r_load", "Ω", 10, 100e3, log_scale=True, description="Load resistor"),
    ],
    # ---- Additional Amplifiers ----
    "inverting_summing_amp": [
        ComponentBounds("r_input1", "Ω", 100, 1e6, log_scale=True, description="Input 1 resistor"),
        ComponentBounds("r_input2", "Ω", 100, 1e6, log_scale=True, description="Input 2 resistor"),
        ComponentBounds("r_feedback", "Ω", 100, 10e6, log_scale=True, description="Feedback resistor"),
    ],
    "transimpedance_amp": [
        ComponentBounds("r_feedback", "Ω", 100, 10e6, log_scale=True, description="Feedback resistor"),
        ComponentBounds("c_feedback", "F", 1e-12, 1e-6, log_scale=True, description="Feedback capacitor"),
    ],
    # ---- Power/Misc Topologies ----
    "half_bridge": [
        ComponentBounds("r_dson_high", "Ω", 0.001, 0.5, log_scale=True, description="High-side MOSFET Rdson"),
        ComponentBounds("r_dson_low", "Ω", 0.001, 0.5, log_scale=True, description="Low-side MOSFET Rdson"),
        ComponentBounds("inductance", "H", 1e-6, 1e-3, log_scale=True, description="Output inductor"),
        ComponentBounds("capacitance", "F", 1e-6, 1e-3, log_scale=True, description="Output capacitor"),
    ],
    "push_pull": [
        ComponentBounds("turns_ratio", "", 0.1, 10.0, log_scale=True, description="Transformer Np/Ns"),
        ComponentBounds("r_dson", "Ω", 0.01, 0.5, log_scale=True, description="MOSFET Rds(on)"),
        ComponentBounds("capacitance", "F", 1e-6, 1e-2, log_scale=True, description="Output capacitor"),
        ComponentBounds("inductance", "H", 1e-6, 1e-3, log_scale=True, description="Output inductor"),
    ],
    "charge_pump": [
        ComponentBounds("c_flying", "F", 1e-6, 100e-6, log_scale=True, description="Flying capacitor"),
        ComponentBounds("c_output", "F", 1e-6, 1e-3, log_scale=True, description="Output capacitor"),
        ComponentBounds("r_load", "Ω", 1, 1e4, log_scale=True, description="Load resistance"),
        ComponentBounds("r_esr", "Ω", 0.001, 1.0, log_scale=True, description="ESR"),
    ],
    "voltage_doubler": [
        ComponentBounds("c1", "F", 1e-6, 1e-3, log_scale=True, description="Coupling capacitor"),
        ComponentBounds("c2", "F", 1e-6, 1e-3, log_scale=True, description="Output capacitor"),
        ComponentBounds("r_diode1", "Ω", 0.01, 1.0, log_scale=True, description="Diode 1 forward resistance"),
        ComponentBounds("r_diode2", "Ω", 0.01, 1.0, log_scale=True, description="Diode 2 forward resistance"),
    ],
    "zeta_converter": [
        ComponentBounds("inductance_1", "H", 1e-6, 1e-3, log_scale=True, description="Input inductor"),
        ComponentBounds("inductance_2", "H", 1e-6, 1e-3, log_scale=True, description="Output inductor"),
        ComponentBounds("cap_coupling", "F", 0.1e-6, 100e-6, log_scale=True, description="Coupling capacitor"),
        ComponentBounds("capacitance", "F", 1e-6, 1e-2, log_scale=True, description="Output capacitor"),
        ComponentBounds("r_dson", "Ω", 0.01, 0.5, log_scale=True, description="MOSFET Rds(on)"),
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
    # ---- Tier 3: BJT Amplifiers ----
    "common_emitter": {"vin_amp": 0.1, "freq_test": 1e3, "vcc": 12.0},
    "common_collector": {"vin_amp": 0.1, "freq_test": 1e3, "vcc": 12.0},
    "common_base": {"vin_amp": 0.1, "freq_test": 1e3, "vcc": 12.0},
    "cascode": {"vin_amp": 0.1, "freq_test": 1e3, "vcc": 12.0},
    "current_mirror": {"vcc": 12.0},
    # ---- Additional Filters ----
    "twin_t_notch": {"vin_amp": 0.5, "freq_test": 1e3, "vcc": 12.0},
    "state_variable_filter": {"freq_test": 1e3},
    # ---- Additional Oscillators ----
    "hartley": {"vcc": 12.0},
    "phase_shift": {},
    # ---- Regulators ----
    "shunt_regulator": {"vcc": 12.0, "v_zener": 5.1},
    "series_regulator": {"vcc": 12.0, "v_ref": 2.5},
    # ---- Additional Amplifiers ----
    "inverting_summing_amp": {"vin_amp": 0.1, "freq_test": 1e3},
    "transimpedance_amp": {"freq_test": 1e3},
    # ---- Power/Misc Topologies ----
    "half_bridge": {"vin": 48.0, "vout": 24.0, "iout": 2.0, "fsw": 100e3},
    "push_pull": {"vin": 48.0, "vout": 12.0, "iout": 2.0, "fsw": 100e3},
    "charge_pump": {"vin": 5.0, "vout": 10.0, "iout": 0.1, "fsw": 100e3},
    "voltage_doubler": {"vin": 12.0, "vout": 24.0, "iout": 0.1, "fsw": 50e3},
    "zeta_converter": {"vin": 12.0, "vout": 5.0, "iout": 1.0, "fsw": 100e3},
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
        ComponentBounds("inductance_1", "H", 10e-6, 500e-6, log_scale=True),
        ComponentBounds("inductance_2", "H", 10e-6, 500e-6, log_scale=True),
        ComponentBounds("cap_coupling", "F", 1e-6, 22e-6, log_scale=True),  # tighter: 0.1-100uF was too wide
        ComponentBounds("capacitance", "F", 10e-6, 1e-3, log_scale=True),
        ComponentBounds("esr", "Ω", 0.005, 0.2, log_scale=True),
        ComponentBounds("r_dson", "Ω", 0.01, 0.2, log_scale=True),
    ],
    "flyback": [
        ComponentBounds("inductance_primary", "H", 50e-6, 2e-3, log_scale=True),
        ComponentBounds("turns_ratio", "", 0.5, 5.0, log_scale=True, description="Np/Ns"),  # tighter: avoids extreme duty
        ComponentBounds("capacitance", "F", 10e-6, 1e-3, log_scale=True),
        ComponentBounds("esr", "Ω", 0.005, 0.2, log_scale=True),
        ComponentBounds("r_dson", "Ω", 0.01, 0.2, log_scale=True),
    ],
    "forward": [
        ComponentBounds("inductance_primary", "H", 50e-6, 2e-3, log_scale=True),
        ComponentBounds("turns_ratio", "", 1.0, 8.0, log_scale=True),  # Np/Ns; Vout=Vin/N*D, D<=0.45
        ComponentBounds("inductance_output", "H", 5e-6, 500e-6, log_scale=True),
        ComponentBounds("capacitance", "F", 10e-6, 1e-3, log_scale=True),
        ComponentBounds("esr", "Ω", 0.005, 0.2, log_scale=True),
        ComponentBounds("r_dson", "Ω", 0.01, 0.2, log_scale=True),
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
    "common_emitter", "common_collector", "common_base", "cascode", "current_mirror",
    "twin_t_notch", "state_variable_filter",
    "hartley", "phase_shift",
    "shunt_regulator", "series_regulator",
    "inverting_summing_amp", "transimpedance_amp",
    "half_bridge", "push_pull", "charge_pump", "voltage_doubler", "zeta_converter",
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
    "common_emitter": _common_emitter_netlist,
    "common_collector": _common_collector_netlist,
    "common_base": _common_base_netlist,
    "cascode": _cascode_netlist,
    "current_mirror": _current_mirror_netlist,
    "twin_t_notch": _twin_t_notch_netlist,
    "state_variable_filter": _state_variable_filter_netlist,
    "hartley": _hartley_netlist,
    "phase_shift": _phase_shift_netlist,
    "shunt_regulator": _shunt_regulator_netlist,
    "series_regulator": _series_regulator_netlist,
    "inverting_summing_amp": _inverting_summing_amp_netlist,
    "transimpedance_amp": _transimpedance_amp_netlist,
    "half_bridge": _half_bridge_netlist,
    "push_pull": _push_pull_netlist,
    "charge_pump": _charge_pump_netlist,
    "voltage_doubler": _voltage_doubler_netlist,
    "zeta_converter": _zeta_converter_netlist,
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
    # Tier 3 — BJT amplifiers
    "common_emitter": "Common emitter BJT amplifier",
    "common_collector": "Common collector (emitter follower) BJT amplifier",
    "common_base": "Common base BJT amplifier",
    "cascode": "Cascode BJT amplifier (CE + CB stacked)",
    "current_mirror": "BJT current mirror",
    # Additional filters
    "twin_t_notch": "Twin-T notch filter",
    "state_variable_filter": "State variable filter (2nd order)",
    # Additional oscillators
    "hartley": "Hartley oscillator",
    "phase_shift": "Phase shift RC oscillator",
    # Regulators
    "shunt_regulator": "Zener-based shunt regulator",
    "series_regulator": "Series pass transistor regulator",
    # Additional amplifiers
    "inverting_summing_amp": "Inverting summing amplifier (2-input)",
    "transimpedance_amp": "Transimpedance amplifier",
    # Power/Misc
    "half_bridge": "Half-bridge DC-DC converter",
    "push_pull": "Push-pull isolated DC-DC converter",
    "charge_pump": "Switched-capacitor charge pump",
    "voltage_doubler": "Voltage doubler rectifier",
    "zeta_converter": "Zeta (inverse-SEPIC) DC-DC converter",
}

# Metric names per domain
_POWER_METRIC_NAMES = ["vout_avg", "vout_ripple", "iout_avg", "iin_avg", "il_ripple", "pout", "pin"]
# All AC circuits now share the same raw measurement names: gain_db, gain_mag,
# phase_rad, gain_dc, vdb_0..vdb_7 (probed at 10,100,1k,10k,100k,1M,10M,50M Hz).
# VP() returns phase in radians.  Derived metrics computed in datagen post-processing.
_AC_METRIC_NAMES = ["gain_db", "gain_mag", "phase_rad", "gain_dc",
                    "vdb_0", "vdb_1", "vdb_2", "vdb_3", "vdb_4", "vdb_5", "vdb_6", "vdb_7"]
_OSC_METRIC_NAMES = ["vosc_pp", "vosc_avg"]
_MIRROR_METRIC_NAMES = ["iref", "iout"]

_METRIC_MAP = {
    **{n: _POWER_METRIC_NAMES for n in _TIER1_NAMES},
    "inverting_amp": _AC_METRIC_NAMES,
    "noninverting_amp": _AC_METRIC_NAMES,
    "instrumentation_amp": _AC_METRIC_NAMES,
    "differential_amp": _AC_METRIC_NAMES,
    "sallen_key_lowpass": _AC_METRIC_NAMES,
    "sallen_key_highpass": _AC_METRIC_NAMES,
    "sallen_key_bandpass": _AC_METRIC_NAMES,
    "wien_bridge": _OSC_METRIC_NAMES,
    "colpitts": _OSC_METRIC_NAMES,
    "common_emitter": _AC_METRIC_NAMES,
    "common_collector": _AC_METRIC_NAMES,
    "common_base": _AC_METRIC_NAMES,
    "cascode": _AC_METRIC_NAMES,
    "current_mirror": _MIRROR_METRIC_NAMES,
    "twin_t_notch": _AC_METRIC_NAMES,
    "state_variable_filter": _AC_METRIC_NAMES,
    "hartley": _OSC_METRIC_NAMES,
    "phase_shift": _OSC_METRIC_NAMES,
    "shunt_regulator": _REGULATOR_METRIC_NAMES,
    "series_regulator": _REGULATOR_METRIC_NAMES,
    "inverting_summing_amp": _AC_METRIC_NAMES,
    "transimpedance_amp": _AC_METRIC_NAMES,
    "half_bridge": _POWER_METRIC_NAMES,
    "push_pull": _POWER_METRIC_NAMES,
    "charge_pump": _POWER_METRIC_NAMES,
    "voltage_doubler": _POWER_METRIC_NAMES,
    "zeta_converter": _POWER_METRIC_NAMES,
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
