# Closing the Loop in Epitaxy with Machine Learning

This repository contains the data-driven workflow for the joint optimization of growth and geometry in III-V multi-quantum well (MQW) microring lasers. The methodology establishes a generalisable blueprint for designing and yield-optimizing complex, non-linear optoelectronic devices by bridging the gap between fundamental epitaxy and reliable manufacturing.

-----

## Research Purpose

Achieving functional uniformity across large populations of nominally identical devices is a persistent challenge in nanoscale engineering. For microcavity lasers, device-to-device reproducibility is a critical bottleneck for scalable photonic integrated circuits (PICs).

Our work moves beyond targeting "hero" devices by explicitly incorporating population-level statistics into the design process. By targeting threshold variance alongside absolute performance, we achieved:

  * **100% lasing yield** across all optimized designs.
  * **73% reduction** in threshold variance relative to previous best values.
  * **Median lasing threshold** of 16~\mu J~cm^{-2}pulse^{-1}.
  * **Steered emission wavelength** to 1333 nm (telecommunications O-band).

-----

## Methodology & Code Features

The provided code implements a two-phase optimization and diagnostic workflow:

### 1\. Multi-Objective Bayesian Optimization (MOBO)

  * **Parameter Space**: Navigates a 7-dimensional space comprising five growth parameters (e.g., temperature, V/III ratio) and two geometry parameters (diameter and pitch).
  * **Objectives**: Jointly optimizes median lasing threshold, emission wavelength, and within-field threshold variance.
  * **Implementation**: Built using **BoTorch**, utilizing Gaussian-process surrogate models and the qN-ParEGO acquisition function.

### 2\. Morphology Diagnostics with Variational Autoencoders (VAEs)

  * **Latent Representation**: Learns a 32-dimensional compressed latent representation of ring morphology directly from optical images.
  * **Variance Attribution**: Successfully decouples geometric disorder from material disorder, quantitatively linking unmeasured morphological variations to population-level threshold fluctuations.
  * **Predictive Modeling**: Demonstrates that VAE-derived morphology descriptors provide significant predictive power for lasing thresholds beyond nominal design parameters.

-----

## Suggested Citation

If you use this code or methodology in your research, please cite the following paper:

```bibtex
@article{Athavale2026,
  author = {Athavale, Mihir R. and Church, Stephen A. and Wong, Wei Wen and Low, Andre K. Y. and Tan, Hark Hoe and Hippalgaonkar, Kedar and Parkinson, Patrick},
  title = {Closing the Loop in Epitaxy with Machine Learning: Joint Optimization of Growth and Geometry in On-Chip Lasers},
  journal = {arXiv preprint arXiv:2604.08390},
  year = {2026},
  eprint = {2604.08390},
  archivePrefix = {arXiv},
  primaryClass = {physics.optics},
  url = {https://arxiv.org/abs/2604.08390}
}
```

  * **ArXiv**: [https://arxiv.org/abs/2604.08390](https://arxiv.org/abs/2604.08390)
  * **Dataset (Figshare)**: [DOI: 10.48420/27283161](https://www.google.com/search?q=https://doi.org/10.48420/27283161)