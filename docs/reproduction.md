# Reproduction Guide

The scripts under `experiments/` correspond to Section 4 of the SO-RaNN
preprint.

| Paper result | Script |
| --- | --- |
| Example 4.1 / Figures 5-6 | `python experiments/example4_1/PNP_ex1_st_re.py` |
| Example 4.2 / Figures 7-11 | `python experiments/example4_2/PNP_ex2_st_re.py` |
| Example 4.3 / Figures 12-13 | `python experiments/example4_3/PNP_ex3_st_re.py` |
| Example 4.4 / Figure 14 | `python experiments/example4_4/PNP_ex4_st_re.py` |
| Example 4.5 / Tables 5-6 | `python experiments/example4_5/run_accuracy_sweep.py` |
| Example 4.6 / Figures 16-17 | `python experiments/example4_6/PNPNS_ex2_st_re.py` |

Submission-revision results are reproduced by the scripts integrated into the
corresponding example directories.

| Revised result | Script |
| --- | --- |
| Example 4.1 / Figure 4: stage-wise charge compatibility and potential re-fits | `python experiments/example4_1/run_charge_neutrality.py` |
| Example 4.1 / Table 3 and Figure 3: seeds 42-51 | `python experiments/example4_1/run_multiseed.py` |
| Example 4.1 / Table 2: RaNN and SO-RaNN width/timing sweep | `python experiments/example4_1/run_accuracy_timing.py` |
| Example 4.1 / Table 1: multi-resolution FDM comparison | `python experiments/example4_1/fdm/run_convergence.py` |
| Example 4.3 / Table 4: five-stage diagnostics | `python experiments/example4_3/run_ablation.py` |
| Example 4.5 / Tables 5-6: final-charge potential accuracy | `python experiments/example4_5/run_accuracy_sweep.py` |
| Example 4.5 / Figure 15: final charge compatibility | `python experiments/example4_5/run_charge_neutrality.py` |

The main experiment scripts write figures to `figures/` and model checkpoints
to `results/checkpoints/`. The additional scripts write figures, tabular
summaries, diagnostics, and redirected legacy outputs below
`results/revision/`. Set the environment variable `SO_RANN_RUN` only when a
custom run-specific output subdirectory is needed.

## Recovering the Example 4.3 profile figure

Example 4.3 saves the full scaler to `results/PNP_code/example4_3/scaler.pt`
before rendering the corresponding profile figure. If a plotting run is
interrupted after the solve has finished, regenerate only that figure with:

```bash
python experiments/example4_3/PNP_ex3_st_re.py --plot_only
```

If PyCharm shows `Waiting for process detach` after a native crash, terminate the
stuck Python process from PowerShell or Task Manager before launching another
run:

```powershell
Get-Process python,pythonw -ErrorAction SilentlyContinue
Stop-Process -Name python,pythonw -Force
```

The default parameters reproduce the paper-scale runs and may be expensive.
For a quick installation check, run:

```bash
python examples/run_smoke.py
pytest
```
