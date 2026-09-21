from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
for path in (PROJECT_ROOT, SRC_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from so_rann.experiment_io import output_path, save_json, save_rows
from so_rann.reproducibility import set_seed
from PNPNS_ex1_st_re import main


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Example 4.5 accuracy sweep with the final-charge Poisson fit"
    )
    parser.add_argument("--widths", type=int, nargs="+", default=[200, 400, 800, 1600])
    parser.add_argument("--reynolds", type=float, nargs="+", default=[10.0, 1000.0])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--report-points", type=int, default=51)
    parser.add_argument("--quadrature-order", type=int, default=20)
    parser.add_argument("--picard-max", type=int, default=20)
    parser.add_argument("--penalty", type=float, default=100.0)
    return parser.parse_args()


def run(args: argparse.Namespace) -> list[dict[str, float | int]]:
    rows: list[dict[str, float | int]] = []
    for reynolds in args.reynolds:
        for width in args.widths:
            os.environ["SO_RANN_RUN"] = (
                f"example4_5/accuracy_refit/re{reynolds:g}_m{width}_seed{args.seed}"
            )
            set_seed(args.seed)
            diagnostics = main(
                r_c1=4.0,
                r_c2=4.0,
                r_phi=4.0,
                r_u=4.0,
                r_p=4.0,
                n=args.n,
                Nt=1,
                Nre=args.report_points,
                n_int=args.quadrature_order,
                m=width,
                tt=args.picard_max,
                p1=args.penalty,
                device=torch.device("cpu"),
                reynolds=reynolds,
            )
            if not diagnostics["potential_diagnostics"]["reported_phi_uses_final_charge"]:
                raise RuntimeError("Reported potential is not fitted to the final charge density")
            results = diagnostics["results"]
            row = {
                "reynolds": reynolds,
                "width": width,
                "seed": args.seed,
                "c1_l2_error": float(results["c1_error_all"][-1]),
                "c2_l2_error": float(results["c2_error_all"][-1]),
                "phi_l2_error": float(results["phi_error_all"][-1]),
                "u1_l2_error": float(results["u1_error_all"][-1]),
                "u2_l2_error": float(results["u2_error_all"][-1]),
                "p_l2_error": float(results["p_error_all"][-1]),
                "divergence_l2_error": float(results["Divergence_free_all"][-1]),
                "picard_iterations_total": int(sum(diagnostics["picard_iterations"])),
                "runtime_seconds": float(diagnostics["timings"]["total"]),
            }
            rows.append(row)
            print(row)

    os.environ["SO_RANN_RUN"] = "example4_5/accuracy_refit/aggregate"
    save_rows(output_path("example4_5_accuracy_refit.csv"), rows)
    save_json(output_path("example4_5_accuracy_refit.json"), rows)
    return rows


if __name__ == "__main__":
    run(parse_args())
