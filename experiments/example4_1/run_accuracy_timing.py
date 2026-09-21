from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
for path in (PROJECT_ROOT, SRC_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from so_rann.experiment_io import output_path, save_json, save_rows
from so_rann.reproducibility import set_seed
from PNP_ex1_st_re import main


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="RaNN and SO-RaNN accuracy/timing runs for the current accuracy/timing table"
    )
    parser.add_argument("--widths", type=int, nargs="+", default=[200, 400, 800, 1600])
    parser.add_argument("--methods", choices=["rann", "sorann"], nargs="+",
                        default=["rann", "sorann"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--report-points", type=int, default=51)
    parser.add_argument("--quadrature-order", type=int, default=20)
    parser.add_argument("--picard-max", type=int, default=30)
    parser.add_argument("--penalty", type=float, default=100.0)
    return parser.parse_args()


def run(args: argparse.Namespace) -> list[dict[str, float | int | str]]:
    if args.repeats < 1:
        raise ValueError("--repeats must be at least 1")

    rows: list[dict[str, float | int | str]] = []
    detailed_runs: list[dict[str, float | int | str]] = []
    for method in args.methods:
        for width in args.widths:
            samples = []
            errors = None
            for repeat in range(1, args.repeats + 1):
                os.environ["SO_RANN_RUN"] = (
                    f"example4_1/accuracy_timing/{method}_m{width}_seed{args.seed}_run{repeat}"
                )
                set_seed(args.seed)
                diagnostics = main(
                    r_c1=2.0,
                    r_c2=2.0,
                    r_phi=2.0,
                    n=args.n,
                    Nt=1,
                    Nre=args.report_points,
                    n_int=args.quadrature_order,
                    m=width,
                    tt=args.picard_max,
                    p1=args.penalty,
                    device=torch.device("cpu"),
                    method=method,
                )
                results = diagnostics["results"]
                errors = {
                    "c1_l2_error": float(results["c1_error_all"][-1]),
                    "c2_l2_error": float(results["c2_error_all"][-1]),
                    "phi_l2_error": float(results["phi_error_all"][-1]),
                }
                if method == "sorann":
                    final_fit_error = float(
                        diagnostics["potential_diagnostics"]
                        ["phi_l2_error_final_charge_fit"]
                    )
                    if not np.isclose(
                        errors["phi_l2_error"], final_fit_error,
                        rtol=1.0e-10, atol=1.0e-12,
                    ):
                        raise RuntimeError(
                            "Reported phi error is not from the final-charge "
                            "Poisson fit: "
                            f"reported={errors['phi_l2_error']}, "
                            f"final_fit={final_fit_error}"
                        )
                runtime = float(diagnostics["timings"]["total"])
                picard_iterations_total = int(sum(diagnostics["picard_iterations"]))
                samples.append(runtime)
                detailed_runs.append({
                    "method": method,
                    "width": width,
                    "seed": args.seed,
                    "repeat": repeat,
                    **errors,
                    "picard_iterations_total": picard_iterations_total,
                    "runtime_seconds": runtime,
                })

            assert errors is not None
            row = {
                "method": method,
                "width": width,
                "seed": args.seed,
                **errors,
                "picard_iterations_total": picard_iterations_total,
                "runtime_seconds": float(np.median(samples)),
                "runtime_min_seconds": float(np.min(samples)),
                "runtime_max_seconds": float(np.max(samples)),
                "repeats": args.repeats,
            }
            rows.append(row)
            print(row)

    os.environ["SO_RANN_RUN"] = "example4_1/accuracy_timing/aggregate"
    save_rows(output_path("example4_1_accuracy_timing.csv"), rows)
    save_rows(output_path("example4_1_accuracy_timing_all_runs.csv"), detailed_runs)
    save_json(output_path("example4_1_accuracy_timing.json"), {
        "summary": rows,
        "runs": detailed_runs,
        "runtime_statistic": "median",
    })
    return rows


if __name__ == "__main__":
    run(parse_args())


