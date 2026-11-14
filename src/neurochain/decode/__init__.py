"""Decoders that map neural features to behaviour.

* :mod:`neurochain.decode.ridge`: a lagged linear readout, the baseline, and the
  same readout with tuned history and penalty (a Wiener filter).
* :mod:`neurochain.decode.kalman`: a steady-state Kalman filter.
* :mod:`neurochain.decode.gru`: a small recurrent network.
* :mod:`neurochain.decode.evaluate`: held-out predictions and confidence
  intervals, shared by all of them.
"""
