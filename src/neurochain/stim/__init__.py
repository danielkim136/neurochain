"""Stimulation: the pulse, how its artifact reaches a recording site, and how to remove it.

* :mod:`neurochain.stim.pulse`: the current waveform and the pulse schedule.
* :mod:`neurochain.stim.propagation`: potential of point current sources in tissue.
* :mod:`neurochain.stim.cancel`: blanking and the adaptive template canceller.

The amplifier's large-signal behaviour lives with the rest of the front end, in
:mod:`neurochain.frontend.saturation`.
"""
