from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_DIR = PROJECT_ROOT / "src"
for path in (PROJECT_ROOT, SRC_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from so_rann.experiment_io import output_path, save_json, save_rows
from pnp_example4_1_bdf2 import solve_example4_1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Multi-resolution BDF2 finite-difference experiment for Example 4.1")
    parser.add_argument(
        "--grids",
        type=int,
        nargs="+",
        default=[21, 41, 81, 161, 321],
        help=(
            "grid sizes; the default doubles the number of subintervals at "
            "each refinement: N-1 = 20, 40, 80, 160, 320"
        ),
    )
    parser.add_argument(
        "--dts", type=float, nargs="+", default=None,
        help="time steps paired with --grids; if omitted, use dt=h/4",
    )
    parser.add_argument("--final-time", type=float, default=1.0)
    parser.add_argument(
        "--append-existing", action="store_true",
        help=(
            "reuse completed requested resolutions from the existing JSON/CSV "
            "and run only missing requested grids"
        ),
    )
    parser.add_argument(
        "--stop-after-seconds", type=float, default=None,
        help="stop after the first newly completed grid whose solver time reaches this value",
    )
    parser.add_argument("--quick", action="store_true", help="two very small smoke-test resolutions")
    return parser.parse_args()


def _recompute_orders(rows: list[dict]) -> None:
    rows.sort(key=lambda row: int(row["grid_points"]))
    for index, row in enumerate(rows):
        for error_name in ("c1", "c2", "phi"):
            row[f"{error_name}_observed_order"] = None
        if index == 0:
            continue
        previous = rows[index - 1]
        log_refinement = math.log(float(previous["h"]) / float(row["h"]))
        for error_name in ("c1", "c2", "phi"):
            previous_error = float(previous[f"{error_name}_l2_error"])
            current_error = float(row[f"{error_name}_l2_error"])
            row[f"{error_name}_observed_order"] = math.log(previous_error / current_error) / log_refinement


def _save(rows: list[dict], csv_path: Path, json_path: Path) -> None:
    _recompute_orders(rows)
    save_rows(csv_path, rows)
    save_json(json_path, {"runs": rows})


def run(args: argparse.Namespace) -> list[dict[str, float | int | None]]:
    if args.quick:
        args.grids, args.dts, args.final_time = [9, 13], [2.5e-2, 1.25e-2], 0.1
    if args.dts is None:
        args.dts = [1.0 / (2.0 * (grid - 1)) for grid in args.grids]
    if len(args.grids) != len(args.dts):
        raise ValueError("--grids and --dts must contain the same number of entries")
    if args.stop_after_seconds is not None and args.stop_after_seconds <= 0:
        raise ValueError("--stop-after-seconds must be positive")

    os.environ["SO_RANN_RUN"] = "example4_1/fdm_convergence"
    csv_path = output_path("example4_1_fdm_convergence.csv")
    json_path = output_path("example4_1_fdm_convergence.json")
    rows = []
    if args.append_existing and json_path.exists():
        with json_path.open("r", encoding="utf-8") as handle:
            requested_grids = set(args.grids)
            rows = [
                row for row in json.load(handle).get("runs", [])
                if int(row["grid_points"]) in requested_grids
            ]
    completed_grids = {int(row["grid_points"]) for row in rows}

    for grid, dt in zip(args.grids, args.dts):
        if grid in completed_grids:
            print(f"Skipping completed grid N={grid}")
            continue
        result = solve_example4_1(grid_points=grid, dt=dt, final_time=args.final_time)
        row = result.as_dict()
        rows.append(row)
        completed_grids.add(grid)
        _save(rows, csv_path, json_path)
        print(row)
        if args.stop_after_seconds is not None and result.runtime_seconds >= args.stop_after_seconds:
            print(
                f"Stopping after N={grid}: runtime {result.runtime_seconds:.3f} s "
                f"reached target {args.stop_after_seconds:.3f} s"
            )
            break

    return rows


if __name__ == "__main__":
    run(parse_args())
