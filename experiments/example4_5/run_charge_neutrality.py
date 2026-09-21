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

from so_rann.experiment_io import save_charge_diagnostics
from so_rann.reproducibility import set_seed
from PNPNS_ex1_st_re import main


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Charge-neutrality diagnostic for Example 4.5")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--blocks", type=int, default=1)
    parser.add_argument("--report-points", type=int, default=51)
    parser.add_argument("--quadrature-order", type=int, default=20)
    parser.add_argument("--neurons", type=int, default=400)
    parser.add_argument("--picard-max", type=int, default=20)
    parser.add_argument("--penalty", type=float, default=100.0)
    parser.add_argument("--quick", action="store_true", help="small smoke-test configuration")
    return parser.parse_args()


def run(args: argparse.Namespace) -> dict[str, float]:
    if args.quick:
        args.n, args.report_points, args.quadrature_order = 4, 5, 5
        args.neurons, args.picard_max = 30, 1
    os.environ["SO_RANN_RUN"] = f"example4_5/charge_seed_{args.seed}"
    set_seed(args.seed)
    diagnostics = main(
        r_c1=4.0,
        r_c2=4.0,
        r_phi=4.0,
        r_u=4.0,
        r_p=4.0,
        n=args.n,
        Nt=args.blocks,
        Nre=args.report_points,
        n_int=args.quadrature_order,
        m=args.neurons,
        tt=args.picard_max,
        p1=args.penalty,
        device=torch.device("cpu"),
        reynolds=1000.0,
    )
    if not diagnostics["potential_diagnostics"]["reported_phi_uses_final_charge"]:
        raise RuntimeError("Reported potential is not fitted to the final charge density")
    summary = save_charge_diagnostics(diagnostics, "example4_5", z1=1.0, z2=-1.0)
    print(summary)
    return summary


if __name__ == "__main__":
    run(parse_args())


