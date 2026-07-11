# SO-RaNN PNP Paper Code

This repository contains the paper code for the SO-RaNN experiments on
Poisson-Nernst-Planck (PNP) and Poisson-Nernst-Planck-Navier-Stokes (PNP-NS)
systems.

## Paper Information

- Title: Structure-Oriented Randomized Neural Networks for Poisson-Nernst-Planck and Poisson-Nernst-Planck-Navier-Stokes Systems
- Authors: Yunlong Li and Fei Wang
- Journal / preprint: arXiv preprint
- DOI / arXiv: arXiv:2606.19912v1
- Corresponding author: Fei Wang, feiwang.xjtu@xjtu.edu.cn
- Code maintainer: Yunlong Li
- Repository: https://github.com/XJTU-AI4SciComp-Lab/so-rann-pnp-paper-code

## Method Summary

SO-RaNN solves decoupled linearized PNP and PNP-NS subproblems with randomized
neural networks in a space-time formulation. The experiments include positivity
correction, selected-time mass matching, SAV-type post-processing for energy
dissipation diagnostics, and a divergence-free velocity approximation for the
PNP-NS system.

## Repository Structure

```text
.
├── README.md
├── LICENSE
├── CITATION.cff
├── requirements.txt
├── environment.yml
├── src/
│   ├── RNN_ref/
│   └── so_rann/
├── examples/
├── experiments/
│   ├── example4_1/
│   ├── example4_2/
│   ├── example4_3/
│   ├── example4_4/
│   ├── example4_5/
│   └── example4_6/
├── data/
├── results/
├── figures/
├── tests/
└── docs/
```

## Installation

Using conda:

```bash
conda env create -f environment.yml
conda activate so-rann-pnp
```

Using pip:

```bash
pip install -r requirements.txt
```

## Quick Start

Run a lightweight smoke example:

```bash
python examples/run_smoke.py
```

Run the basic tests:

```bash
pytest
```

If `pytest` is not installed yet, the smoke tests can also be run directly:

```bash
python tests/test_core_smoke.py
```

## Reproduce Results

The full paper-scale experiments are in `experiments/example4_*`:

```bash
python experiments/example4_1/PNP_ex1_st_re.py
python experiments/example4_2/PNP_ex2_st_re.py
python experiments/example4_3/PNP_ex3_st_re.py
python experiments/example4_4/PNP_ex4_st_re.py
python experiments/example4_5/PNPNS_ex1_st_re.py
python experiments/example4_6/PNPNS_ex2_st_re.py
```

See `docs/reproduction.md` for the mapping from paper tables and figures to
scripts. Generated figures are written under `figures/`; generated checkpoints
are written under `results/checkpoints/`.

## Data

No external dataset is required for the included scripts. Initial conditions,
manufactured solutions, and benchmark settings are defined in the corresponding
`experiments/example4_*/*_real.py` files.

## Hardware Notes

The smoke example and tests run on CPU. The full experiments are CPU-compatible
but can be expensive because they reproduce paper-scale randomized neural
network solves with many collocation, quadrature, and time-block points.

## Citation

If you use this code, please cite the corresponding paper:

```bibtex
@misc{li2026sorannpnp,
  title        = {Structure-Oriented Randomized Neural Networks for Poisson-Nernst-Planck and Poisson-Nernst-Planck-Navier-Stokes Systems},
  author       = {Li, Yunlong and Wang, Fei},
  year         = {2026},
  eprint       = {2606.19912},
  archivePrefix = {arXiv},
  primaryClass = {math.NA}
}
```

## License

This project is released under the BSD 3-Clause License. See `LICENSE`.
