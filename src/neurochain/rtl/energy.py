"""Energy per sample of the synthesised chain, from counts and switching activity.

No free tool gives the power of an ASIC that has not been designed, and the
power an FPGA would draw says nothing about an implant: an ECP5 burns tens of
milliwatts standing still, in configuration memory and routing that a custom
chip does not have. What the free tools do measure is structure and activity:

* how many gates and flip-flops the logic needs (Yosys, technology-neutral),
* how often every gate output switches on real data (Verilator on that
  gate-level netlist, written out as a SAIF file),
* how many memory bits one sample reads and writes (exact, from the RTL).

Each is multiplied by a stated energy. The energies are assumptions for a
65 nm low-power process at its nominal 1.2 V, kept in :class:`EnergyModel`
with where they come from; none is measured here. The dynamic power of CMOS is
``P = a * C * V**2 * f`` (ELEC2141): ``a * f`` is the measured toggle rate and
``C * V**2 / 2`` the assumed energy of one transition.

What the estimate leaves out: glitches (a zero-delay simulation sees each node
settle once per cycle), wire capacitance beyond the lumped node value, and the
clock tree above the flip-flop pins.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

FLIP_FLOP_GATES = 6
"""Two-input gate equivalents of one flip-flop, for the leakage count."""


@dataclass(frozen=True)
class EnergyModel:
    """Technology constants of the digital energy estimate. All are assumptions.

    Attributes:
        v_supply_v: Core supply. 1.2 V is the nominal voltage of a 65 nm
            low-power process; near-threshold operation would cut dynamic
            energy several times and is not assumed.
        c_node_f: Capacitance switched when one gate output toggles: the
            inputs it drives plus local wire. 2 fF is an order of magnitude
            for a small 65 nm cell with a fan-out of two or three.
        memory_j_per_bit: Energy per memory bit read or written. Meinerzhagen
            et al. (2012) measured 14 fJ per accessed bit on a 4 kbit
            latch-based standard-cell memory in 65 nm, built for implants, at
            its energy-minimum supply of 0.5 V. 20 fJ is that order; at the
            1.2 V assumed for the logic the same memory would cost several
            times more, and a compiled SRAM more again.
        gate_leakage_w: Leakage of one two-input gate equivalent, high-threshold
            cells at body temperature.
        bit_leakage_w: Leakage of one memory bit. The same paper reports 0.5 pW
            per bit at a 0.22 V retention supply; 5 pW allows for a supply that
            is not lowered between samples.
    """

    v_supply_v: float = 1.2
    c_node_f: float = 2.0e-15
    memory_j_per_bit: float = 20.0e-15
    gate_leakage_w: float = 5.0e-12
    bit_leakage_w: float = 5.0e-12

    @property
    def toggle_j(self) -> float:
        """Energy of one transition of one node: ``C * V**2 / 2``."""
        return 0.5 * self.c_node_f * self.v_supply_v**2


def summarise_activity(toggles: dict[str, int], samples: int) -> dict[str, Any]:
    """Reduce a net-by-net toggle count to what the energy model needs.

    Args:
        toggles: Transitions of every net (:func:`neurochain.rtl.tools.read_saif`).
        samples: Samples of one channel the count covers (channels times
            samples per channel).

    Returns:
        The number of nets, the data toggles per sample (clock and reset left
        out), the clock cycles per sample and the mean activity: toggles per
        net per clock cycle.
    """
    data = {name: count for name, count in toggles.items() if name not in {"clk", "rst"}}
    total = sum(data.values())
    cycles = toggles.get("clk", 0) / 2.0
    return {
        "nets": len(data),
        "samples": samples,
        "cycles_per_sample": cycles / samples,
        "toggles_per_sample": total / samples,
        "activity": total / max(cycles, 1.0) / max(len(data), 1),
    }


def chain_energy(
    model: EnergyModel,
    *,
    toggles_per_sample: float,
    flip_flops: int,
    gates: int,
    memory_bits_per_sample: float,
    memory_bits: int,
    n_channels: int,
    fs_hz: float,
    cycles_per_sample: float = 1.0,
) -> dict[str, float]:
    """Return the digital energy per sample and power per channel.

    Args:
        model: Technology constants.
        toggles_per_sample: Node transitions per sample of one channel,
            measured on the gate-level netlist.
        flip_flops: Flip-flops in the datapath; each sees two clock edges per
            cycle.
        gates: Two-input gates in the datapath.
        memory_bits_per_sample: Memory bits read plus written per sample.
        memory_bits: All state bits, for leakage.
        n_channels: Channels sharing the datapath.
        fs_hz: Sample rate per channel.
        cycles_per_sample: Clock cycles per sample of one channel; 1 when the
            clock runs at exactly ``n_channels * fs_hz``.

    Returns:
        Energies in pJ per sample (``logic_pj``, ``clock_pj``, ``memory_pj``),
        and powers in µW per channel (``dynamic_uw``, ``leakage_uw``,
        ``total_uw``).
    """
    logic_j = toggles_per_sample * model.toggle_j
    clock_j = 2.0 * flip_flops * cycles_per_sample * model.toggle_j
    memory_j = memory_bits_per_sample * model.memory_j_per_bit
    dynamic_w = (logic_j + clock_j + memory_j) * fs_hz
    leakage_w = (
        (gates + FLIP_FLOP_GATES * flip_flops) * model.gate_leakage_w
        + memory_bits * model.bit_leakage_w
    ) / n_channels
    return {
        "logic_pj": logic_j * 1e12,
        "clock_pj": clock_j * 1e12,
        "memory_pj": memory_j * 1e12,
        "dynamic_pj": (logic_j + clock_j + memory_j) * 1e12,
        "logic_uw": logic_j * fs_hz * 1e6,
        "clock_uw": clock_j * fs_hz * 1e6,
        "memory_uw": memory_j * fs_hz * 1e6,
        "dynamic_uw": dynamic_w * 1e6,
        "leakage_uw": leakage_w * 1e6,
        "total_uw": (dynamic_w + leakage_w) * 1e6,
    }
