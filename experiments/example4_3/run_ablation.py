from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import defaultdict
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
from PNP_ex3_st_re import main


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Strategy-wise correction diagnostics for Example 4.3")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--blocks", type=int, default=10)
    parser.add_argument("--report-points", type=int, default=51)
    parser.add_argument("--quadrature-order", type=int, default=20)
    parser.add_argument("--neurons", type=int, default=800)
    parser.add_argument("--picard-max", type=int, default=30)
    parser.add_argument("--penalty", type=float, default=100.0)
    parser.add_argument("--reuse", action="store_true", help="reuse existing diagnostic CSV files without solving again")
    parser.add_argument("--quick", action="store_true", help="small one-block smoke test")
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def summarize(
    stage_rows: list[dict[str, str]],
    sav_rows: list[dict[str, str]],
    equation_rows: list[dict[str, str]],
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Collapse the full ten-block diagnostics to one representative row per strategy."""
    raw_stages = {"raw_picard", "additional_np_raw"}
    cutoff_stages = {"positivity_cutoff", "additional_np_cutoff"}
    scaled_stages = {"first_mass_scaling", "final_mass_scaling"}

    positivity_before = min(
        min(float(row["min_c1"]), float(row["min_c2"]))
        for row in stage_rows if row["stage"] in raw_stages
    )
    positivity_after = min(
        min(float(row["min_c1"]), float(row["min_c2"]))
        for row in stage_rows if row["stage"] in cutoff_stages
    )

    mass_before = max(
        max(float(row["relative_mass_defect_c1"]), float(row["relative_mass_defect_c2"]))
        for row in stage_rows if row["stage"] in cutoff_stages
    )
    mass_after = max(
        max(float(row["relative_mass_defect_c1"]), float(row["relative_mass_defect_c2"]))
        for row in stage_rows if row["stage"] in scaled_stages
    )

    sav_by_block: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in sav_rows:
        sav_by_block[int(row["block"])].append(row)
    sav_positive_increments = []
    sav_endpoint_ratios = []
    for block_rows in sav_by_block.values():
        ordered = sorted(block_rows, key=lambda row: int(row["local_step"]))
        values = np.asarray([float(row["R1"]) for row in ordered])
        sav_positive_increments.extend(np.maximum(np.diff(values), 0.0))
        sav_endpoint_ratios.append(values[-1] / values[0])
    max_sav_positive_increment = float(max(sav_positive_increments, default=0.0))
    worst_sav_endpoint_ratio = float(max(sav_endpoint_ratios))
    max_xi_deviation = max(abs(float(row["xi"]) - 1.0) for row in sav_rows)

    equation_by_strategy: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in equation_rows:
        equation_by_strategy[row["strategy"]].append(row)

    def worst_values(strategy: str, before_field: str, after_field: str) -> tuple[float, float]:
        rows = equation_by_strategy[strategy]
        return (
            max(float(row[before_field]) for row in rows),
            max(float(row[after_field]) for row in rows),
        )

    np_stationarity_before, np_stationarity_after = worst_values(
        "additional_np_solve", "before_stationarity_max", "after_stationarity_max"
    )
    poisson_before, poisson_after = worst_values("final_poisson_fit", "before_max", "after_max")

    strategy_rows = [
        {
            "strategy": "positivity_cutoff",
            "targeted_property": "value_level_positivity",
            "primary_metric": "minimum_over_diagnostic_quadrature_points_before",
            "primary_value": positivity_before,
            "secondary_metric": "minimum_over_diagnostic_quadrature_points_after",
            "secondary_value": positivity_after,
        },
        {
            "strategy": "mass_scaling",
            "targeted_property": "selected_time_mass_matching",
            "primary_metric": "max_relative_mass_defect_before",
            "primary_value": mass_before,
            "secondary_metric": "max_relative_mass_defect_after",
            "secondary_value": mass_after,
        },
        {
            "strategy": "sav_update",
            "targeted_property": "sav_auxiliary_variable_monotonicity",
            "primary_metric": "max_positive_increment_of_R1",
            "primary_value": max_sav_positive_increment,
            "secondary_metric": "max_block_endpoint_ratio_R1_end_over_R1_start",
            "secondary_value": worst_sav_endpoint_ratio,
        },
        {
            "strategy": "additional_np_solve",
            "targeted_property": "np_least_squares_refit_with_sav_scaled_potential",
            "primary_metric": "max_normalized_np_lstsq_stationarity_before",
            "primary_value": np_stationarity_before,
            "secondary_metric": "max_normalized_np_lstsq_stationarity_after",
            "secondary_value": np_stationarity_after,
        },
        {
            "strategy": "final_poisson_fit",
            "targeted_property": "poisson_least_squares_refitting",
            "primary_metric": "max_normalized_poisson_lstsq_residual_before",
            "primary_value": poisson_before,
            "secondary_metric": "max_normalized_poisson_lstsq_residual_after",
            "secondary_value": poisson_after,
        },
    ]
    summary = {
        "scope": "all temporal blocks and both occurrences of repeated corrections",
        "positivity_cutoff": {"minimum_before": positivity_before, "minimum_after": positivity_after},
        "mass_scaling": {"max_relative_defect_before": mass_before, "max_relative_defect_after": mass_after},
        "sav_update": {
            "max_positive_increment_R1": max_sav_positive_increment,
            "max_block_endpoint_ratio_R1": worst_sav_endpoint_ratio,
            "max_abs_xi_minus_one": max_xi_deviation,
            "note": "R1 is the constructed SAV auxiliary sequence, not the physical free energy.",
        },
        "additional_np_solve": {
            "max_normalized_np_lstsq_stationarity_before": np_stationarity_before,
            "max_normalized_np_lstsq_stationarity_after": np_stationarity_after,
        },
        "final_poisson_fit": {"max_normalized_poisson_lstsq_residual_before": poisson_before, "max_normalized_poisson_lstsq_residual_after": poisson_after},
    }
    return strategy_rows, summary


def run(args: argparse.Namespace) -> list[dict[str, object]]:
    if args.quick:
        args.n, args.blocks, args.report_points = 4, 1, 5
        args.quadrature_order, args.neurons, args.picard_max = 5, 40, 2
    os.environ["SO_RANN_RUN"] = f"example4_3/ablation_seed_{args.seed}"
    set_seed(args.seed)
    if not args.reuse:
        main(
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
            diagnostics_only=True,
        )

    diagnostics_dir = ("diagnostics",)
    stage_rows = read_rows(output_path(*diagnostics_dir, "example4_3_stage_ablation.csv"))
    sav_rows = read_rows(output_path(*diagnostics_dir, "example4_3_sav_energy.csv"))
    equation_rows = read_rows(output_path(*diagnostics_dir, "example4_3_equation_residuals.csv"))
    strategy_rows, summary = summarize(stage_rows, sav_rows, equation_rows)
    summary = {"seed": args.seed, **summary}
    save_rows(output_path(*diagnostics_dir, "example4_3_strategy_ablation.csv"), strategy_rows)
    save_json(output_path("diagnostics", "example4_3_stage_ablation_summary.json"), summary)
    print(summary)
    return strategy_rows


if __name__ == "__main__":
    run(parse_args())
