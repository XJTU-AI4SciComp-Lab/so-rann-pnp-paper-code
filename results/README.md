# Results

This directory stores generated numerical outputs that are not figures.

The experiment scripts redirect generated model checkpoints to:

```text
results/checkpoints/
```

For example, Example 4.3 writes model checkpoints under:

```text
results/checkpoints/PNP_code/model_ex3/
```

and writes the mass/energy correction scale used by the plotting stage to:

```text
results/PNP_code/example4_3/scaler_phi.txt
results/PNP_code/example4_3/scaler.pt
```

`scaler.pt` allows Figure 10 to be regenerated without rerunning the full
Example 4.3 solve:

```bash
python experiments/example4_3/PNP_ex3_st_re.py --plot_only
```

## Reproduction Notes

The full paper-scale scripts can be expensive. For a quick environment check,
run:

```bash
python examples/run_smoke.py
python tests/test_core_smoke.py
```

For paper-scale reproduction, see `docs/reproduction.md`.

## Repository Policy

Generated checkpoints and large numerical arrays are ignored by `.gitignore`.
Do not commit large `.pth`, `.npy`, `.npz`, `.mat`, `.h5`, or `.pkl` files unless
they are explicitly required for a lightweight reproducibility artifact.
