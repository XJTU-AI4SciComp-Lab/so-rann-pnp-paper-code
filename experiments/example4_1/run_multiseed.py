from __future__ import annotations

import argparse
import os
import sys
import traceback
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
for path in (PROJECT_ROOT, SRC_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from so_rann.experiment_io import charge_series, output_path, save_json, save_rows
from so_rann.reproducibility import set_seed
from PNP_ex1_st_re import main


ERROR_KEYS = ("c1_error_all", "c2_error_all", "phi_error_all")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ten-seed accuracy experiment for Example 4.1")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(42, 52)))
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--blocks", type=int, default=1)
    parser.add_argument("--report-points", type=int, default=51)
    parser.add_argument("--quadrature-order", type=int, default=20)
    parser.add_argument("--neurons", type=int, default=400)
    parser.add_argument("--picard-max", type=int, default=30)
    parser.add_argument("--penalty", type=float, default=100.0)
    parser.add_argument("--quick", action="store_true", help="small smoke-test configuration")
    return parser.parse_args()


def _empty_row(seed: int) -> dict[str, object]:
    return {
        "seed": seed,
        "status": "failed",
        "c1_final_l2_error": np.nan,
        "c2_final_l2_error": np.nan,
        "phi_final_l2_error": np.nan,
        "max_abs_charge_defect": np.nan,
        "final_abs_charge_defect": np.nan,
        "picard_iterations_total": np.nan,
        "error_message": "",
    }


def run(args: argparse.Namespace) -> list[dict[str, object]]:
    if args.quick:
        args.n, args.report_points, args.quadrature_order = 4, 5, 6
        args.neurons, args.picard_max = 40, 2

    rows: list[dict[str, object]] = []
    for seed in args.seeds:
        row = _empty_row(seed)
        os.environ["SO_RANN_RUN"] = f"example4_1/multiseed/seed_{seed}"
        try:
            set_seed(seed)
            diagnostics = main(
                r_c1=2.0,
                r_c2=2.0,
                r_phi=2.0,
                n=args.n,
                Nt=args.blocks,
                Nre=args.report_points,
                n_int=args.quadrature_order,
                m=args.neurons,
                tt=args.picard_max,
                p1=args.penalty,
                device=torch.device("cpu"),
            )
            results = diagnostics["results"]
            charge = charge_series(diagnostics)
            row.update(
                status="ok",
                c1_final_l2_error=float(results["c1_error_all"][-1]),
                c2_final_l2_error=float(results["c2_error_all"][-1]),
                phi_final_l2_error=float(results["phi_error_all"][-1]),
                max_abs_charge_defect=float(np.max(charge["abs_charge_defect"])),
                final_abs_charge_defect=float(charge["abs_charge_defect"][-1]),
                picard_iterations_total=int(sum(diagnostics["picard_iterations"])),
            )
        except Exception as exc:  # failed realizations remain visible in the CSV
            row["error_message"] = f"{type(exc).__name__}: {exc}"
            traceback.print_exc()
        rows.append(row)

    os.environ["SO_RANN_RUN"] = "example4_1/multiseed/aggregate"
    save_rows(output_path("example4_1_multiseed.csv"), rows)
    successful = [row for row in rows if row["status"] == "ok"]
    if not successful:
        raise RuntimeError("All seed realizations failed; inspect the per-seed output directories")

    metric_columns = {
        "c1": "c1_final_l2_error",
        "c2": "c2_final_l2_error",
        "phi": "phi_final_l2_error",
    }
    statistics: dict[str, object] = {
        "requested_seeds": args.seeds,
        "successful_seeds": [int(row["seed"]) for row in successful],
        "failed_seeds": [int(row["seed"]) for row in rows if row["status"] != "ok"],
        "metrics": {},
    }
    for label, column in metric_columns.items():
        values = np.asarray([float(row[column]) for row in successful])
        statistics["metrics"][label] = {
            "mean": float(np.mean(values)),
            "sample_std": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        }
    save_json(output_path("example4_1_multiseed_summary.json"), statistics)

    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    seeds = np.asarray([int(row["seed"]) for row in successful])
    colors = {"c1": "tab:blue", "c2": "tab:orange", "phi": "tab:green"}
    for label, column in metric_columns.items():
        values = np.asarray([float(row[column]) for row in successful])
        ax.plot(seeds, values, marker="o", color=colors[label], label=rf"$e_{{{label}}}$")
        ax.axhline(np.mean(values), linestyle="--", linewidth=1.2, color=colors[label], alpha=0.8,
                   label=rf"mean $e_{{{label}}}$")
    ax.set_yscale("log")
    ax.set_xlabel("Random seed")
    ax.set_ylabel(r"Final-time $L^2$ error")
    ax.set_xticks(seeds)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(ncol=2, fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path("example4_1_multiseed_accuracy.png"), dpi=300)
    plt.close(fig)
    print(statistics)
    return rows


if __name__ == "__main__":
    run(parse_args())
