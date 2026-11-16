# References

Papers, datasets and tools this project relies on, grouped by the unit that
first uses them. DOIs were resolved on 2026-11-13.

## Datasets

- **Neuralink Compression Challenge.** One hour of raw N1 electrode recordings
  from non-human primate motor cortex. Original page (now returns 404):
  <https://content.neuralink.com/compression-challenge/README.html>. Archived
  copy: <https://web.archive.org/web/20250901131704/https://content.neuralink.com/compression-challenge/README.html>.
  Used in U0.
- **Nason, S. R. et al. (2021).** Example Data for "A Low-Power Band of Neuronal
  Spiking Activity Dominated by Local Single Units Improves the Performance of
  Brain-Machine Interfaces". University of Michigan, Deep Blue Data.
  <https://doi.org/10.7302/wwya-5q86>. CC BY-NC 4.0. Planned for U2 and U4.
- **Karpowicz, B. M. et al. (2025).** Data from: Stabilizing brain-computer
  interfaces through alignment of latent dynamics. Dryad.
  <https://doi.org/10.5061/dryad.q83bk3jtp>. CC0. Planned for U4 and U6.

## U0: data and band-split

- **Quiroga, R. Q., Nadasdy, Z. and Ben-Shaul, Y. (2004).** Unsupervised spike
  detection and sorting with wavelets and superparamagnetic clustering. *Neural
  Computation* 16(8), 1661-1687. <https://doi.org/10.1162/089976604774201631>.
  Source of the median-based noise estimate, σ = median(|x|)/0.6745.
- **Welch, P. D. (1967).** The use of fast Fourier transform for the estimation
  of power spectra: a method based on time averaging over short, modified
  periodograms. *IEEE Transactions on Audio and Electroacoustics* 15(2), 70-73.
  <https://doi.org/10.1109/TAU.1967.1161901>. The PSD estimator.
- **Rice, S. O. (1944).** Mathematical analysis of random noise. *Bell System
  Technical Journal* 23(3), 282-332.
  <https://doi.org/10.1002/j.1538-7305.1944.tb00874.x>. The continuous-time
  level-crossing rate; U0 explains why it overestimates for a sampled detector.
- **Christie, B. P. et al. (2015).** Comparison of spike sorting and
  thresholding of voltage waveforms for intracortical brain-machine interface
  performance. *Journal of Neural Engineering* 12(1), 016009.
  <https://doi.org/10.1088/1741-2560/12/1/016009>. Why threshold crossings are
  an acceptable substitute for spike sorting in decoding.
- **Nason, S. R. et al. (2020).** A low-power band of neuronal spiking activity
  dominated by local single units improves the performance of brain-machine
  interfaces. *Nature Biomedical Engineering* 4, 973-983.
  <https://doi.org/10.1038/s41551-020-0591-0>. The 300 to 1000 Hz spiking-band
  power feature (prediction check 4; reproduced in U2).
- **Marblestone, A. H. et al. (2013).** Physical principles for scalable neural
  recording. <https://arxiv.org/abs/1306.5709>. Background for the LFP and
  spike band definitions.
- **Virtanen, P. et al. (2020).** SciPy 1.0: fundamental algorithms for
  scientific computing in Python. *Nature Methods* 17, 261-272.
  <https://doi.org/10.1038/s41592-019-0686-2>. Filter design, filtering and
  Welch's method.

## U1: electrode and front-end model

- **Straka, M. M., Shafer, B., Vasudevan, S., Welle, C. and Rieth, L. (2018).**
  Characterizing longitudinal changes in the impedance spectra of in-vivo
  peripheral nerve electrodes. *Micromachines* 9(11), 587.
  <https://doi.org/10.3390/mi9110587>. Open access. Table 1 gives the Randles
  fits used for the electrode: Blackrock platinum Utah arrays in 0.15 M PBS
  (96 spectra) and two or more weeks after implantation in rat sciatic nerve
  (854 spectra). Values were read from the article's full-text table.
- **Harrison, R. R. and Charles, C. (2003).** A low-power low-noise CMOS
  amplifier for neural recording applications. *IEEE Journal of Solid-State
  Circuits* 38(6), 958-965. <https://doi.org/10.1109/JSSC.2003.811979>. The
  amplifier topology. From the abstract: 0.025 Hz to 7.2 kHz, 2.2 µV rms, 80 µW.
  The gain-setting capacitors (20 pF and 200 fF) are as reported for the same
  design. **Not re-verified this session:** the ±2.5 V supply, which gives the
  16 µA and the NEF of 4.0 used here. The full text was not reachable; check it
  before quoting those three numbers elsewhere.
- **Steyaert, M. S. J. and Sansen, W. M. C. (1987).** A micropower low-noise
  monolithic instrumentation amplifier for medical purposes. *IEEE Journal of
  Solid-State Circuits* 22(6), 1163-1168.
  <https://doi.org/10.1109/JSSC.1987.1052869>. Origin of the noise efficiency
  factor.
- **Lempka, S. F. et al. (2011).** Theoretical analysis of intracortical
  microelectrode recordings. *Journal of Neural Engineering* 8(4), 045006.
  <https://doi.org/10.1088/1741-2560/8/4/045006> (open manuscript:
  <https://pmc.ncbi.nlm.nih.gov/articles/PMC3196618>). Thermal noise of 6.2 µV
  and biological noise of 10.2 µV over 450 to 5000 Hz, 12 µV in total; brain
  resistivity 300 Ω·cm. Used to interpret why a real channel is noisier than the
  front-end model.
- **ngspice.** Open-source circuit simulator, version 47 used here.
  <https://ngspice.sourceforge.io/>.

## U2: ADC and features

- **Buccino, A. P. and Einevoll, G. T. (2021).** MEArec: a fast and customizable
  testbench simulator for ground-truth extracellular spiking activity.
  *Neuroinformatics* 19, 185-204. <https://doi.org/10.1007/s12021-020-09467-7>.
  The ground-truth recordings; version 1.11.0, with NEURON 9 and LFPy 2.3.
- **Mukhopadhyay, S. and Ray, G. C. (1998).** A new interpretation of nonlinear
  energy operator and its efficacy in spike detection. *IEEE Transactions on
  Biomedical Engineering* 45(2), 180-187. <https://doi.org/10.1109/10.661266>.
  The NEO detector and its threshold rule.
- **Nason, S. R. et al. (2020)** (listed under U0). The spiking-band power recipe
  and the data it is reproduced on.

## U3: power model and co-design

- **van Elzakker, M. et al. (2010).** A 10-bit charge-redistribution ADC
  consuming 1.9 µW at 1 MS/s. *IEEE Journal of Solid-State Circuits* 45(5),
  1007-1015. <https://doi.org/10.1109/JSSC.2010.2043893>. The 4.4 fJ per
  conversion step benchmark; the power model assumes ten times that.
- **Sen, S., Chatterjee, B., Barik, G. and Roy, A. (2026).** Bidirectional
  wireless communication for weakly coupled implantable brain-computer
  interfaces. <https://arxiv.org/abs/2609.05786>. The sub-10 pJ/bit target used
  for the radio, and the roughly 10 mW a link can dissipate.
- **Karpowicz, B. M. et al. (2025)** (listed under U6). The NoMAD data the Pareto
  is also evaluated on.

## U4: decoders and evaluation

- **Wu, W., Gao, Y., Bienenstock, E., Donoghue, J. P. and Black, M. J. (2006).**
  Bayesian population decoding of motor cortical activity using a Kalman filter.
  *Neural Computation* 18(1), 80-118.
  <https://doi.org/10.1162/089976606774841585>. The Kalman filter decoder's
  generative model.
- **Malik, W. Q., Truccolo, W., Brown, E. N. and Hochberg, L. R. (2011).**
  Efficient decoding with steady-state Kalman filter in neural interface
  systems. *IEEE Transactions on Neural Systems and Rehabilitation Engineering*
  19(1), 25-34. <https://doi.org/10.1109/TNSRE.2010.2092443>. Why the gain is
  computed once instead of every step.
- **Carmena, J. M. et al. (2003).** Learning to control a brain-machine
  interface for reaching and grasping by primates. *PLoS Biology* 1(2), e42.
  <https://doi.org/10.1371/journal.pbio.0000042>. Wiener filter decoding of
  movement from many channels.
- **Cho, K. et al. (2014).** Learning phrase representations using RNN
  encoder-decoder for statistical machine translation.
  <https://arxiv.org/abs/1406.1078>. The gated recurrent unit.
- **Kingma, D. P. and Ba, J. (2015).** Adam: a method for stochastic
  optimization. <https://arxiv.org/abs/1412.6980>. The optimiser that trains
  the GRU.

## U5: stimulation artifacts

- **Flesher, S. N. et al. (2016).** Intracortical microstimulation of human
  somatosensory cortex. *Science Translational Medicine* 8(361), ra141.
  <https://doi.org/10.1126/scitranslmed.aaf8083>. The reference pulse shape
  (quoted from memory; verify before relying on it).
- **Flesher, S. N. et al. (2021).** A brain-computer interface that evokes
  tactile sensations improves robotic arm control. *Science* 372, 831-836.
  <https://doi.org/10.1126/science.abd0380>. Why an implant has to record while
  it stimulates.
- **Logothetis, N. K., Kayser, C. and Oeltermann, A. (2007).** In vivo
  measurement of cortical impedance spectrum in monkeys. *Neuron* 55, 809-823.
  <https://doi.org/10.1016/j.neuron.2007.07.027>. Tissue is resistive, so the
  artifact propagates without filtering.
- **Merrill, D. R., Bikson, M. and Jefferys, J. G. R. (2005).** Electrical
  stimulation of excitable tissue: design of efficacious and safe protocols.
  *Journal of Neuroscience Methods* 141, 171-198.
  <https://doi.org/10.1016/j.jneumeth.2004.10.020>. Charge-balanced stimulation.
- **Weiss, J. M. et al. (2019).** Artifact-free recordings in human
  bidirectional brain-computer interfaces. *Journal of Neural Engineering* 16,
  016002. <https://doi.org/10.1088/1741-2552/aae748>. The blanking baseline.
- **Widrow, B. et al. (1975).** Adaptive noise cancelling: principles and
  applications. *Proceedings of the IEEE* 63, 1692-1716.
  <https://doi.org/10.1109/PROC.1975.10036>. The LMS update behind the
  template canceller.
- **Mendrela, A. E. et al. (2016).** A bidirectional neural interface circuit
  with active stimulation artifact cancellation and cross-channel common-mode
  noise suppression. *IEEE Journal of Solid-State Circuits* 51, 955-965.
  <https://doi.org/10.1109/JSSC.2015.2506651>. Alternative considered: analog
  cancellation in the front end.
- **O'Shea, D. J. and Shenoy, K. V. (2018).** ERAASR: an algorithm for removing
  electrical stimulation artifacts from multielectrode array recordings.
  *Journal of Neural Engineering* 15, 026020.
  <https://doi.org/10.1088/1741-2552/aaa365>. Alternative considered:
  cross-channel PCA removal.
- **Zhou, A. et al. (2019).** A wireless and artefact-free 128-channel
  neuromodulation device for closed-loop stimulation and recording in
  non-human primates. *Nature Biomedical Engineering* 3, 15-26.
  <https://doi.org/10.1038/s41551-018-0323-x>. Alternative considered: a front
  end with enough range never to saturate.

## U6: aging stress test

- **Barrese, J. C., Rao, N., Paroo, K., Triebwasser, C., Vargas-Irwin, C.,
  Franquemont, L. and Donoghue, J. P. (2013).** Failure mode analysis of
  silicon-based intracortical microelectrode arrays in non-human primates.
  *Journal of Neural Engineering* 10(6), 066014.
  <https://doi.org/10.1088/1741-2560/10/6/066014>. The impedance rise (2.48
  times in two weeks, then stable), the 1.53-fold noise rise used as a
  variation, and the quoted Chestek figure.
- **Chestek, C. A. et al. (2011).** Long-term stability of neural prosthetic
  control signals from silicon cortical arrays in rhesus macaque motor cortex.
  *Journal of Neural Engineering* 8(4), 045005.
  <https://doi.org/10.1088/1741-2560/8/4/045005>. The 2.4 % per month decline
  in spike amplitude, taken from Barrese et al.'s summary and not checked
  against the original text. Verify before quoting.
- **Straka, M. M. et al. (2018)** (listed under U1). The fresh electrode, and
  the reason its implanted fit was not used.
- **Williams, J. C., Hippensteel, J. A., Dilgen, J., Shain, W. and Kipke, D. R.
  (2007).** Complex impedance spectroscopy for monitoring tissue responses to
  inserted neural implants. *Journal of Neural Engineering* 4(4), 410-423.
  <https://doi.org/10.1088/1741-2560/4/4/007>. Encapsulation modelled as added
  tissue impedance (only a summary was read).
- **Sankar, V. et al. (2014).** Electrode impedance analysis of chronic
  tungsten microwire neural implants: understanding abiotic vs. biotic
  contributions. *Frontiers in Neuroengineering* 7, 13.
  <https://doi.org/10.3389/fneng.2014.00013>. Impedance rises and later falls;
  the fall is not modelled.
- **Karpowicz, B. M. et al. (2025).** Stabilizing brain-computer interfaces
  through alignment of latent dynamics. *Nature Communications* 16, 4662.
  <https://doi.org/10.1038/s41467-025-59652-y>. The NoMAD sessions and the idea
  of unsupervised alignment; its nonlinear, recurrent method is not implemented.
- **Degenhart, A. D. et al. (2020).** Stabilization of a brain-computer
  interface via the alignment of low-dimensional spaces of neural activity.
  *Nature Biomedical Engineering* 4(7), 672-685.
  <https://doi.org/10.1038/s41551-020-0542-9>. The linear precedent for
  `neurochain.align`, which uses a simpler principal-axes and Procrustes version.
- **Jarosiewicz, B. et al. (2015).** Virtual typing by people with tetraplegia
  using a self-calibrating intracortical brain-computer interface. *Science
  Translational Medicine* 7(313), 313ra179.
  <https://doi.org/10.1126/scitranslmed.aac7328>. Tracking feature statistics
  during use instead of recalibrating.
- **Schönemann, P. H. (1966).** A generalized solution of the orthogonal
  Procrustes problem. *Psychometrika* 31(1), 1-10.
  <https://doi.org/10.1007/BF02289451>. The `U·Vᵀ` solution in
  `align.procrustes_rotation` (citation from memory; verify the DOI).
- **Nason, S. R. et al. (2020)** (listed under U0) and **Buccino, A. P. and
  Einevoll, G. T. (2021)** (listed under U2). SBP and the Nason data; MEArec.

## U7: RTL

Verified means the title, venue and the number quoted were confirmed from a
publisher or repository page during U7. "From memory" means not checked.

- **Meinerzhagen, P., Sherazi, S. M. Y., Burg, A. and Rodrigues, J. N. (2012).**
  A 500 fW/bit 14 fJ/bit-access 4kb standard-cell based sub-VT memory in 65nm
  CMOS. *Proc. ESSCIRC*, 321-324.
  <https://doi.org/10.1109/ESSCIRC.2012.6341319>. The energy per memory bit
  accessed (14 fJ at 0.5 V) and the leakage per bit (0.5 pW at 0.22 V) behind
  two constants of `neurochain.rtl.energy.EnergyModel`. Title, venue, DOI and
  both numbers verified from repository abstracts; the full text was not read,
  and the author list is from memory.
- **Horowitz, M. (2014).** Computing's energy problem (and what we can do about
  it). *IEEE ISSCC Digest of Technical Papers*, 10-14.
  <https://doi.org/10.1109/ISSCC.2014.6757323>. 0.1 pJ for a 32-bit addition at
  45 nm and 0.9 V, as a cross-check on the logic energy. Title, venue and DOI
  verified; the energy figure and the page range are from memory. Verify before
  quoting.
- **Kuon, I. and Rose, J. (2007).** Measuring the gap between FPGAs and ASICs.
  *IEEE Transactions on Computer-Aided Design of Integrated Circuits and
  Systems* 26(2), 203-215. <https://doi.org/10.1109/TCAD.2006.884574>. Why an
  FPGA's power is not scaled to an ASIC's here (about 14 times in dynamic power
  on average; an alternative considered and not used). Title, venue, volume,
  pages and the factor verified from the abstract; the DOI is from memory.
- **Hartley, R. I. (1996).** Subexpression sharing in filters using canonic
  signed digit multipliers. *IEEE Transactions on Circuits and Systems II*
  43(10), 677-688. <https://doi.org/10.1109/82.539000>. Background for
  canonical signed-digit constant multiplication; the recoding in
  `nc_const_mult.sv` is the standard non-adjacent form, checked numerically,
  not taken from this paper. Citation from memory; verify before quoting.
- **Widrow, B. et al. (1975)** (listed under U5), **Nason, S. R. et al. (2020)**
  (listed under U0) and the U5 pulse-shape sources. The canceller is U5's, the
  feature is Nason's SBP; nothing new is taken from them.
- **Tools, not papers:** Verilator (<https://verilator.org>), Yosys
  (<https://yosyshq.net/yosys/>), nextpnr (<https://github.com/YosysHQ/nextpnr>),
  Project Trellis (<https://github.com/YosysHQ/prjtrellis>).
