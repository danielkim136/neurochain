"""The SBP feature path as hardware: specification, stimulus, tools and energy.

* :mod:`neurochain.rtl.spec` holds the chain's parameters, derives every word
  width from them and composes the integer Python reference the SystemVerilog
  under ``rtl/`` has to match bit for bit.
* :mod:`neurochain.rtl.stimulus` builds the ADC codes the comparison runs on.
* :mod:`neurochain.rtl.tools` drives Verilator, Yosys and nextpnr.
* :mod:`neurochain.rtl.energy` turns cell counts and simulated switching
  activity into an energy per sample.
"""
