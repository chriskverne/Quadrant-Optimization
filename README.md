# Quadrant-Optimization

Research code for **WSBD (Weighted Stochastic Block Descent)**, a freezing-based
optimizer for quantum neural networks. WSBD scores parameters by a
gradient-derived importance measure and temporarily freezes the low-importance
ones, reducing the number of circuit evaluations per training step while
retaining the full expressive capacity of the ansatz.

This repository contains the optimizer implementations and the benchmarks used
in the paper.

## Repository layout

| Path | Contents |
| --- | --- |
| `VQE/` | VQE ground-state-energy benchmark (noiseless and noisy) |
| `MNIST/`, `MNIST ACCURACY/`, `Fashion_mnist/`, `XOR/` | Classification benchmarks, one file per optimizer (WSBD, Adam, SGD, SPSA, Nelder-Mead, Bayesian, DBD, random freezing) |
| `Ip_metrics/` | Alternative importance metrics for the freezing rule (Fisher, EMA, variance, recency) |
| `layer_freezing/` | Layer-wise freezing baselines |
| `qiskit/` | Qiskit implementations, including `qiskit/VQE.py` (TFIM/XXZ Hamiltonian builders, hardware-efficient ansatz, parameter-shift training loop) |
| `helper/` | Dataset loading, QNN construction and loss helpers |
| `plots/` | Figure generation |

Most of the code uses PennyLane; `qiskit/` uses Qiskit and Qiskit Aer.

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for the
full terms. Copyright (c) 2025-2026 Christopher Kverne.

All code in this repository is original work by the author. It is not adapted
from any third-party tutorial or benchmark implementation, and it carries no
separate third-party notices. The quantum-computing constructs it uses (the
TFIM and XXZ Heisenberg Hamiltonians, hardware-efficient ansätze, and the
parameter-shift rule) are standard published methods, implemented here against
the public PennyLane and Qiskit APIs.

If you redistribute this code or substantial excerpts of it, retain the
copyright notice and the MIT permission notice, as the license requires. The
credit to retain is:

> Copyright (c) 2025-2026 Christopher Kverne — https://github.com/chriskverne/Quadrant-Optimization

## Citation

The MIT License does not require academic citation, but if you use this code,
or any excerpt or adaptation of it, in published work, please cite the WSBD
paper:

> C. Kverne, M. Akewar, Y. Huo, T. Patel, and J. Bhimani. "WSBD: Freezing-Based
> Optimizer for Quantum Neural Networks." *Proceedings of the International
> Conference on Artificial Intelligence and Statistics (AISTATS)*, 2026.
> arXiv:2602.11383.

```bibtex
@inproceedings{kverne2026wsbd,
  title     = {{WSBD}: Freezing-Based Optimizer for Quantum Neural Networks},
  author    = {Kverne, Christopher and Akewar, Mayur and Huo, Yuqian and
               Patel, Tirthak and Bhimani, Janki},
  booktitle = {Proceedings of the International Conference on Artificial
               Intelligence and Statistics (AISTATS)},
  year      = {2026},
  note      = {arXiv:2602.11383},
  url       = {https://arxiv.org/abs/2602.11383}
}
```

A machine-readable version of this citation is in [CITATION.cff](CITATION.cff).
