# Data

This repository does not require external datasets.

All initial conditions, manufactured exact solutions, source terms, and
benchmark settings are defined directly in the scripts under
`experiments/example4_*/*_real.py`.

## Included Problem Definitions

| Paper example | Definition file | Description |
| --- | --- | --- |
| Example 4.1 | `experiments/example4_1/ex1_real.py` | Two-component PNP accuracy test with manufactured solution. |
| Example 4.2 | `experiments/example4_2/ex2_real.py` | Two-component PNP benchmark with Neumann boundary data. |
| Example 4.3 | `experiments/example4_3/ex3_real.py` | Two-component PNP benchmark with discontinuous initial values and periodic boundary data. |
| Example 4.4 | `experiments/example4_4/ex4_real.py` | Three-component PNP benchmark. |
| Example 4.5 | `experiments/example4_5/ex1_real.py` | Two-component PNP-NS accuracy test with manufactured solution. |
| Example 4.6 | `experiments/example4_6/ex2_real.py` | Two-component PNP-NS benchmark. |

## Repository Policy

Do not commit large generated arrays or checkpoint files here. If future
experiments need external data, add download or generation instructions to this
file instead of committing large `.mat`, `.npy`, `.h5`, or `.pth` files.

