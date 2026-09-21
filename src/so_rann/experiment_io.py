"""Output and diagnostic helpers for the paper-scale experiments."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from .paths import legacy_output_path as repository_legacy_output_path
from .paths import result_path as repository_result_path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = PROJECT_ROOT / "results" / "revision"


def _safe_run_name() -> Path:
    """Return a repository-relative run name from ``SO_RANN_RUN``."""
    raw_path = Path(os.environ.get("SO_RANN_RUN", "default"))
    parts = []
    for part in raw_path.parts:
        if part in {"", ".", "..", "/", "\\", raw_path.anchor}:
            continue
        name = Path(part).name
        if name not in {"", ".", ".."}:
            parts.append(name)
    return Path(*parts) if parts else Path("default")


def output_path(*parts: str | os.PathLike[str]) -> Path:
    """Return an output path below ``results/revision/<run>``."""
    path = OUTPUT_ROOT / _safe_run_name()
    for part in parts:
        candidate = Path(part)
        path = path / (candidate.name if candidate.is_absolute() else candidate)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def legacy_output_path(path: str | os.PathLike[str]) -> Path:
    """Route revision runs separately while preserving the default layout."""
    if os.environ.get("SO_RANN_RUN"):
        return output_path("legacy", Path(path))
    return repository_legacy_output_path(path)


def result_path(*parts: str | os.PathLike[str]) -> Path:
    """Return a result path in the current run or the repository default."""
    if os.environ.get("SO_RANN_RUN"):
        return output_path("results", *parts)
    return repository_result_path(*parts)


def to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def save_rows(path: Path, rows: list[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError("Cannot write an empty table")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def charge_series(
    diagnostics: Mapping[str, Any],
    z1: float = 1.0,
    z2: float = -1.0,
    mass_keys: tuple[str, str] = ("m1_all", "m2_all"),
) -> dict[str, np.ndarray]:
    """Compute absolute and normalized charge defects from stored masses."""
    results = diagnostics["results"]
    time = to_numpy(diagnostics["time"]).reshape(-1)
    m1 = to_numpy(results[mass_keys[0]]).reshape(-1)
    m2 = to_numpy(results[mass_keys[1]]).reshape(-1)
    charge = z1 * m1 + z2 * m2
    denominator = np.abs(z1) * np.abs(m1) + np.abs(z2) * np.abs(m2)
    normalized = np.abs(charge) / np.maximum(denominator, np.finfo(float).eps)
    return {
        "time": time,
        "mass_c1": m1,
        "mass_c2": m2,
        "charge_defect": charge,
        "abs_charge_defect": np.abs(charge),
        "normalized_charge_defect": normalized,
    }


def save_charge_diagnostics(
    diagnostics: Mapping[str, Any],
    label: str,
    z1: float = 1.0,
    z2: float = -1.0,
) -> dict[str, Any]:
    """Save charge-neutrality data, summary, and the two-panel figure."""
    import matplotlib.pyplot as plt

    results = diagnostics["results"]
    stage_keys = {
        "raw": ("m1_raw_all", "m2_raw_all"),
        "cut": ("m1_cut_all", "m2_cut_all"),
        "final": ("m1_all", "m2_all"),
    }
    available_stages = {
        stage: charge_series(diagnostics, z1=z1, z2=z2, mass_keys=keys)
        for stage, keys in stage_keys.items()
        if keys[0] in results and keys[1] in results
    }
    if not available_stages:
        available_stages = {"final": charge_series(diagnostics, z1=z1, z2=z2)}
    series = available_stages["final"]

    rows = []
    for index in range(series["time"].size):
        row = {"time": float(series["time"][index])}
        for stage, values in available_stages.items():
            for key in (
                "mass_c1",
                "mass_c2",
                "charge_defect",
                "abs_charge_defect",
                "normalized_charge_defect",
            ):
                row[f"{stage}_{key}"] = float(values[key][index])
        rows.append(row)
    save_rows(output_path("diagnostics", f"{label}_charge_neutrality.csv"), rows)

    with plt.rc_context({"font.family": "serif", "ps.fonttype": 42, "pdf.fonttype": 42}):
        fig, axes = plt.subplots(2, 1, figsize=(6.2, 6.2), sharex=True)
        colors = {"raw": "tab:blue", "cut": "tab:orange", "final": "tab:green"}
        labels = {"raw": "raw", "cut": "post-cut-off", "final": "final mass-corrected"}
        plot_stages = {
            stage: values for stage, values in available_stages.items() if stage != "cut"
        }
        for stage, values in plot_stages.items():
            axes[0].semilogy(
                values["time"],
                np.maximum(values["abs_charge_defect"], 1e-18),
                color=colors.get(stage),
                linewidth=1.2,
                label=labels.get(stage, stage),
            )
        single_final_stage = tuple(plot_stages) == ("final",)
        axes[0].set_ylabel(
            r"$\delta_Q^{\mathrm{final}}(t)$" if single_final_stage else r"$\delta_Q^s(t)$"
        )
        axes[0].grid(True, color="0.85", linestyle="--", linewidth=0.6)
        for stage, values in plot_stages.items():
            axes[1].semilogy(
                values["time"],
                np.maximum(values["normalized_charge_defect"], 1e-18),
                color=colors.get(stage),
                linewidth=1.2,
                label=labels.get(stage, stage),
            )
        axes[1].set_xlabel(r"$t$")
        axes[1].set_ylabel(
            r"$(\delta_Q^{\mathrm{rel}})^{\mathrm{final}}(t)$"
            if single_final_stage
            else r"$(\delta_Q^{\mathrm{rel}})^s(t)$"
        )
        axes[1].legend(fontsize=8, loc="best")
        axes[1].grid(True, color="0.85", linestyle="--", linewidth=0.6)
        fig.tight_layout()
        fig.savefig(output_path("diagnostics", f"{label}_charge_neutrality.png"), dpi=300)
        fig.savefig(
            output_path("diagnostics", f"{label}_charge_neutrality.pdf"),
            format="pdf",
            bbox_inches="tight",
        )
        fig.savefig(
            output_path("diagnostics", f"{label}_charge_neutrality.eps"),
            format="eps",
            bbox_inches="tight",
        )
        plt.close(fig)

    summary: dict[str, Any] = {
        "max_abs_charge_defect": float(np.max(series["abs_charge_defect"])),
        "final_abs_charge_defect": float(series["abs_charge_defect"][-1]),
        "max_normalized_charge_defect": float(np.max(series["normalized_charge_defect"])),
        "final_normalized_charge_defect": float(series["normalized_charge_defect"][-1]),
        "stages": {
            stage: {
                "max_abs_charge_defect": float(np.max(values["abs_charge_defect"])),
                "final_abs_charge_defect": float(values["abs_charge_defect"][-1]),
                "max_normalized_charge_defect": float(np.max(values["normalized_charge_defect"])),
                "final_normalized_charge_defect": float(values["normalized_charge_defect"][-1]),
            }
            for stage, values in available_stages.items()
        },
    }
    if "potential_diagnostics" in diagnostics:
        summary["potential_diagnostics"] = {
            key: bool(value) if isinstance(value, (bool, np.bool_)) else float(value)
            for key, value in diagnostics["potential_diagnostics"].items()
        }
    save_json(output_path("diagnostics", f"{label}_charge_neutrality_summary.json"), summary)
    return summary
