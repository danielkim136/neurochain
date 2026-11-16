"""Drive Verilator, Yosys and nextpnr, and read back what they produce.

All three are optional: tests that need one skip without it, the way the
ngspice tests do. Everything the tools write goes under ``build/u7`` (ignored
by git); only numbers read from their reports reach ``results/``.

* Verilator compiles the SystemVerilog and ``rtl/tb/tb_chain.cpp`` into a
  program that plays codes from a file and writes every output to files.
* Yosys synthesises the same sources twice: for the Lattice ECP5 (LUTs,
  flip-flops, block RAM), and into NAND/NOR/NOT gates, a technology-neutral
  netlist that is simulated again to count how often each node switches.
* nextpnr places and routes the ECP5 netlist and reports the clock rate the
  slowest path allows.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from neurochain.paths import repo_root
from neurochain.rtl.spec import ChainSpec
from neurochain.rtl.stimulus import Stimulus

TOP = "nc_sbp_chain"
SOURCES = (
    "nc_sdp_ram.sv",
    "nc_const_mult.sv",
    "nc_cancel_core.sv",
    "nc_highpass_core.sv",
    "nc_sbp_chain.sv",
)
TESTBENCH = "tb/tb_chain.cpp"
DEVICE = "LFE5U-25F"
PACKAGE = "CABGA256"
SPEED_GRADE = 6


class ToolError(RuntimeError):
    """A tool is missing or returned an error."""


def rtl_dir() -> Path:
    """Directory of the SystemVerilog sources."""
    return repo_root() / "rtl"


def build_dir() -> Path:
    """Directory for everything the tools write (ignored by git)."""
    return repo_root() / "build" / "u7"


def source_paths() -> list[Path]:
    """The synthesisable sources, leaf modules first."""
    return [rtl_dir() / name for name in SOURCES]


def find_tool(name: str) -> str | None:
    """Return the path of a tool, or ``None`` if it is not installed.

    ``NEUROCHAIN_<NAME>`` overrides the search. ``~/.local/bin`` is tried after
    ``PATH`` because nextpnr-ecp5 has no Homebrew formula and is built there.
    """
    override = os.environ.get("NEUROCHAIN_" + name.upper().replace("-", "_"))
    if override:
        return override if Path(override).exists() else None
    found = shutil.which(name)
    if found:
        return found
    local = Path.home() / ".local" / "bin" / name
    return str(local) if local.exists() else None


def tools_available(*names: str) -> bool:
    """Return whether every named tool is installed."""
    return all(find_tool(name) is not None for name in names)


def _run(command: list[str], *, cwd: Path | None = None) -> str:
    """Run a command and return its output; raise :class:`ToolError` if it fails."""
    tool = find_tool(command[0])
    if tool is None:
        raise ToolError(f"{command[0]} is not installed")
    result = subprocess.run(
        [tool, *command[1:]], cwd=cwd, capture_output=True, text=True, check=False
    )
    if result.returncode != 0:
        tail = "\n".join((result.stdout + result.stderr).splitlines()[-25:])
        raise ToolError(f"{command[0]} failed:\n{tail}")
    return result.stdout + result.stderr


def tool_versions() -> dict[str, str]:
    """Return the version string of each tool that is installed."""
    versions: dict[str, str] = {}
    for name, flag in (("verilator", "--version"), ("yosys", "-V"), ("nextpnr-ecp5", "--version")):
        if find_tool(name) is None:
            continue
        text = _run([name, flag]).strip().splitlines()[0]
        match = re.search(r"\d+\.\d+(\.\d+)?", text)
        versions[name] = match.group(0) if match else "unknown (built from a source archive)"
    return versions


# ------------------------------------------------------------------------- simulation


def build_model(
    spec: ChainSpec, directory: Path, *, netlist: Path | None = None, activity: bool = False
) -> Path:
    """Compile the chain and its testbench with Verilator; return the program.

    Args:
        spec: Sets the module's parameters.
        directory: Where Verilator writes its C++ and the program; emptied first.
        netlist: Simulate this gate-level netlist (written by
            :func:`synthesise_gates` for the same ``spec``) instead of the RTL.
        activity: Build with SAIF tracing, so the program can record how often
            every net toggles. Much slower; meant for the gate-level netlist.
    """
    if find_tool("verilator") is None:
        raise ToolError("verilator is not installed")
    # Start clean: make would not notice that only the compiler flags changed.
    shutil.rmtree(directory, ignore_errors=True)
    directory.mkdir(parents=True)
    command = ["verilator", "--cc", "--exe", "--build", "-j", "0", "-O3", "--top-module", TOP]
    if netlist is None:
        command += ["--assert", "-Wall", "-Wno-DECLFILENAME", "-Wno-UNUSEDSIGNAL"]
        command += [f"-G{name}={value}" for name, value in spec.verilog_parameters().items()]
        sources = [str(p) for p in source_paths()]
    else:
        command += ["-Wno-fatal"]
        sources = [str(netlist)]
    if activity:
        command += ["--trace-saif", "--trace-underscore"]
    defines = f"-DTB_N_CH={spec.n_channels} -DTB_CODE_W={spec.code_bits}"
    command += ["-CFLAGS", f"-O2 -std=c++17 {defines}"]
    command += ["-Mdir", str(directory), "-o", "tb_chain", *sources, str(rtl_dir() / TESTBENCH)]
    _run(command)
    return directory / "tb_chain"


@dataclass(frozen=True)
class SimOutput:
    """Everything the simulated chain produced, as files mapped into memory.

    Arrays are ``(n_samples, n_channels)`` or ``(n_bins, n_channels)``: the
    order the hardware emits them in.

    Attributes:
        cancelled_raw: Raw bits of the canceller output.
        filtered_raw: Raw bits of the filter output.
        ok: 1 where the sample was not blanked.
        sums: Bin sums.
        counts: Valid samples per bin.
        report: Counts printed by the testbench (cycles, outputs, tag errors).
        spec: The chain that produced them, for the widths.
    """

    cancelled_raw: NDArray[np.uint16]
    filtered_raw: NDArray[np.uint16]
    ok: NDArray[np.uint8]
    sums: NDArray[np.uint32]
    counts: NDArray[np.uint8]
    report: dict[str, Any]
    spec: ChainSpec

    def cancelled(self, channel: int) -> NDArray[np.int64]:
        """Return one channel's canceller output as signed integers."""
        return sign_extend(self.cancelled_raw[:, channel], self.spec.cancel_bits)

    def filtered(self, channel: int) -> NDArray[np.int64]:
        """Return one channel's filter output as signed integers."""
        return sign_extend(self.filtered_raw[:, channel], self.spec.y_bits)


def sign_extend(raw: NDArray[np.uint16], bits: int) -> NDArray[np.int64]:
    """Interpret the low ``bits`` of each value as a two's-complement number."""
    sign = 1 << (bits - 1)
    return (np.asarray(raw, dtype=np.int64) ^ sign) - sign


def load_output(prefix: Path, spec: ChainSpec, report: dict[str, Any]) -> SimOutput:
    """Map the files a simulation wrote under ``prefix`` into memory."""
    n_ch = spec.n_channels

    def load(suffix: str, dtype: Any) -> Any:
        path = prefix.with_name(prefix.name + suffix)
        if path.stat().st_size == 0:
            return np.zeros((0, n_ch), dtype=dtype)
        return np.memmap(path, dtype=dtype, mode="r").reshape(-1, n_ch)

    return SimOutput(
        cancelled_raw=load(".c.bin", np.uint16),
        filtered_raw=load(".y.bin", np.uint16),
        ok=load(".ok.bin", np.uint8),
        sums=load(".sum.bin", np.uint32),
        counts=load(".cnt.bin", np.uint8),
        report=report,
        spec=spec,
    )


def run_model(
    program: Path,
    stimulus: Stimulus,
    spec: ChainSpec,
    prefix: Path,
    *,
    activity_path: Path | None = None,
) -> SimOutput:
    """Play a stimulus through a compiled model.

    Args:
        program: From :func:`build_model`, built for the same ``spec``.
        stimulus: Codes and pulse onsets.
        spec: The chain the program was built for.
        prefix: Path prefix of the files written (codes in, outputs out).
        activity_path: Write a SAIF activity file here (needs ``activity=True``
            at build time).
    """
    if stimulus.codes.shape[0] != spec.n_channels:
        raise ValueError("stimulus and specification disagree on the number of channels")
    spec.check_onsets(stimulus.onsets)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    codes_path = prefix.with_name(prefix.name + ".codes.bin")
    trig_path = prefix.with_name(prefix.name + ".trig.bin")
    np.ascontiguousarray(stimulus.codes.T).tofile(codes_path)
    stimulus.trigger().tofile(trig_path)
    command = [str(program), str(codes_path), str(trig_path), str(prefix)]
    if activity_path is not None:
        command.append(str(activity_path))
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0 or "%Error" in result.stdout + result.stderr:
        tail = "\n".join((result.stdout + result.stderr).splitlines()[-15:])
        raise ToolError(f"simulation failed:\n{tail}")
    report = json.loads(result.stdout.strip().splitlines()[-1])
    codes_path.unlink()
    return load_output(prefix, spec, report)


# -------------------------------------------------------------------------- synthesis


def _read_sources(spec: ChainSpec) -> str:
    """Return the Yosys commands that read the sources with this spec's parameters.

    ``chparam`` re-derives the top module under a generated name, so the top is
    found again and given its own name back. (``hierarchy -chparam``, the
    documented route, trips an assertion in Yosys 0.71.) Values are written as
    32 bits of two's complement because a negative decimal is not accepted.
    """
    files = " ".join(str(p) for p in source_paths())
    params = " ".join(
        f"-set {k} 32'sb{v & 0xFFFFFFFF:032b}" for k, v in spec.verilog_parameters().items()
    )
    return (
        f"read_verilog -sv -DSYNTHESIS {files}; chparam {params} {TOP}; "
        f"hierarchy -auto-top; rename -top {TOP}"
    )


def _cell_counts(stat_path: Path) -> dict[str, int]:
    """Read the cell counts of the top module from ``stat -json`` output."""
    design = json.loads(stat_path.read_text())["design"]
    return {k: int(v) for k, v in design["num_cells_by_type"].items() if not k.startswith("$scope")}


def synthesise_ecp5(
    spec: ChainSpec,
    directory: Path,
    *,
    dsp: bool = False,
    freq_mhz: float = 50.0,
    seeds: tuple[int, ...] = (1,),
) -> dict[str, Any]:
    """Synthesise for the ECP5 with Yosys, place and route with nextpnr.

    Args:
        spec: The chain.
        directory: Where the netlist, logs and reports go.
        dsp: Let Yosys use the ECP5's hard 18x18 multipliers. Off by default:
            an implant has none, and with them the LUT count hides logic.
        freq_mhz: Clock rate nextpnr aims for. It stops improving once the
            target is met, so the achieved rate depends on it.
        seeds: Placement seeds; one run each. Placement is a random search and
            the achieved rate moves by several percent between seeds.

    Returns:
        Cell counts from Yosys, utilisation from nextpnr (first seed) and the
        achieved clock rate of every seed.
    """
    directory.mkdir(parents=True, exist_ok=True)
    netlist = directory / "chain.json"
    stat = directory / "stat.json"
    _run(
        [
            "yosys",
            "-q",
            "-l",
            str(directory / "yosys.log"),
            "-p",
            f"{_read_sources(spec)}; synth_ecp5 -top {TOP} {'' if dsp else '-nodsp'} "
            f"-json {netlist}; tee -q -o {stat} stat -json",
        ]
    )
    fmax: list[float] = []
    utilisation: dict[str, int] = {}
    slowest: dict[str, Any] = {}
    for seed in seeds:
        report = directory / f"pnr_seed{seed}.json"
        _run(
            [
                "nextpnr-ecp5",
                "--25k",
                "--package",
                PACKAGE,
                "--speed",
                str(SPEED_GRADE),
                "--json",
                str(netlist),
                "--freq",
                str(freq_mhz),
                "--seed",
                str(seed),
                "--timing-allow-fail",
                "--report",
                str(report),
                "-q",
                "-l",
                str(directory / f"pnr_seed{seed}.log"),
            ]
        )
        data = json.loads(report.read_text())
        fmax.append(max(float(clock["achieved"]) for clock in data["fmax"].values()))
        if not utilisation:
            utilisation = {k: int(v["used"]) for k, v in data["utilization"].items() if v["used"]}
            available = {k: int(v["available"]) for k, v in data["utilization"].items()}
            slowest = _critical_path((directory / f"pnr_seed{seed}.log").read_text())
    cells = _cell_counts(stat)
    return {
        "n_channels": spec.n_channels,
        "dsp": dsp,
        "shift_add": spec.shift_add,
        "cells": cells,
        "lut4": cells.get("LUT4", 0),
        "flip_flops": cells.get("TRELLIS_FF", 0),
        "block_rams": cells.get("DP16KD", 0),
        "multipliers": cells.get("MULT18X18D", 0),
        "utilisation": utilisation,
        "logic_cells": utilisation.get("TRELLIS_COMB", 0),
        "logic_cells_available": available["TRELLIS_COMB"],
        "block_rams_available": available["DP16KD"],
        "fmax_mhz": fmax,
        "fmax_median_mhz": float(np.median(fmax)),
        "critical_path": slowest,
        "freq_target_mhz": freq_mhz,
        "seeds": list(seeds),
    }


def _critical_path(log: str) -> dict[str, Any]:
    """Read where the slowest register-to-register path starts and ends from a nextpnr log.

    Returns:
        The cell the path leaves (``from``), the delay from the clock edge to
        that cell's output (``clk_to_q_ns``), the cell it arrives at (``to``)
        and the whole path's delay including setup.
    """
    start = log.rfind("Critical path report for clock")
    stop = log.find("Critical path report", start + 1)
    lines = [
        line.removeprefix("Info:").split()
        for line in log[start : stop if stop > 0 else None].splitlines()
    ]
    steps = [w for w in lines if len(w) >= 5 and w[3] == "Source"]
    if not steps or steps[0][0] != "clk-to-q":
        return {}
    return {
        "from": steps[0][4].split(".")[0],
        "clk_to_q_ns": float(steps[0][1]),
        "to": steps[-1][4].split(".")[0],
        "delay_ns": float(steps[-1][2]),
    }


def synthesise_gates(spec: ChainSpec, directory: Path) -> dict[str, Any]:
    """Synthesise into NAND, NOR and NOT gates plus flip-flops, memories left whole.

    The result is not any real cell library: it is a count of two-input gate
    equivalents and a netlist whose every wire is one gate's output, which is
    what the activity measurement needs.

    Returns:
        Gate and flip-flop counts, memory bits and the netlist's path.
    """
    directory.mkdir(parents=True, exist_ok=True)
    netlist = directory / "gates.v"
    stat = directory / "stat.json"
    _run(
        [
            "yosys",
            "-q",
            "-l",
            str(directory / "yosys.log"),
            "-p",
            f"{_read_sources(spec)}; synth -top {TOP} -flatten -run begin:fine; "
            "opt -fast -full; techmap; opt -fast; abc -g cmos2; opt -fast; "
            "dfflegalize -cell $_DFF_P_ 0 -cell $_DFFE_PP_ 0; opt_clean -purge; "
            f"tee -q -o {stat} stat -json; write_verilog -noattr {netlist}",
        ]
    )
    cells = _cell_counts(stat)
    gates = sum(v for k, v in cells.items() if k in {"$_NAND_", "$_NOR_", "$_NOT_", "$_MUX_"})
    flip_flops = sum(v for k, v in cells.items() if "DFF" in k)
    memory_bits = spec.n_channels * sum(spec.state_bits.values())
    return {
        "cells": cells,
        "gates": gates,
        "flip_flops": flip_flops,
        "memories": cells.get("$mem_v2", 0),
        "memory_bits": memory_bits,
        "netlist": str(netlist),
    }


def read_saif(path: Path, instance: str = TOP) -> dict[str, int]:
    """Return the toggle count of every net inside one instance of a SAIF file.

    A toggle is one transition, rising or falling. Bits of a bus are separate
    nets. Verilator repeats the ports one level up, under ``TOP``; those copies
    are left out.
    """
    toggles: dict[str, int] = {}
    inside = False
    pattern = re.compile(r"^\s*\((\S+) \(T0 \d+\) \(T1 \d+\).*\(TC (\d+)\)\)")
    for line in path.read_text().splitlines():
        if line.strip().startswith("(INSTANCE"):
            inside = line.split()[1].rstrip(")") == instance
            continue
        match = pattern.match(line) if inside else None
        if match:
            toggles[match.group(1).replace("\\", "")] = int(match.group(2))
    return toggles
