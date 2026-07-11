# Reproduction Guide

The scripts under `experiments/` correspond to Section 4 of the SO-RaNN
preprint.

| Paper result | Script |
| --- | --- |
| Example 4.1 / Table 1 | `python experiments/example4_1/PNP_ex1_st_re.py` |
| Example 4.2 / Figures 5-9 | `python experiments/example4_2/PNP_ex2_st_re.py` |
| Example 4.3 / Figures 10-11 | `python experiments/example4_3/PNP_ex3_st_re.py` |
| Example 4.4 / Figure 12 | `python experiments/example4_4/PNP_ex4_st_re.py` |
| Example 4.5 / Tables 2-3 | `python experiments/example4_5/PNPNS_ex1_st_re.py` |
| Example 4.6 / Figures 13-14 | `python experiments/example4_6/PNPNS_ex2_st_re.py` |

Generated figures are redirected to `figures/`. Generated model checkpoints are
redirected to `results/checkpoints/`. The numerical algorithms are otherwise
kept compatible with the original scripts.

## Recovering Figure 10

Example 4.3 saves the full scaler to `results/PNP_code/example4_3/scaler.pt`
before rendering Figure 10. If a plotting run is interrupted after the solve has
finished, regenerate only Figure 10 with:

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
