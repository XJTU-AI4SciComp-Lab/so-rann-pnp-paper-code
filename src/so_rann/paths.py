"""Path helpers for repository-local outputs."""

from pathlib import Path, PurePosixPath


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIGURES_DIR = PROJECT_ROOT / "figures"
RESULTS_DIR = PROJECT_ROOT / "results"


def ensure_parent(path: Path) -> Path:
    """Create the parent directory for an output path and return the path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def figure_path(*parts: object) -> Path:
    """Return a path below ``figures/`` and create its parent directory."""
    return ensure_parent(FIGURES_DIR.joinpath(*(str(part) for part in parts)))


def result_path(*parts: object) -> Path:
    """Return a path below ``results/`` and create its parent directory."""
    return ensure_parent(RESULTS_DIR.joinpath(*(str(part) for part in parts)))


def checkpoint_path(*parts: object) -> Path:
    """Return a checkpoint path below ``results/checkpoints/``."""
    return result_path("checkpoints", *parts)


def legacy_output_path(path: object) -> Path:
    """Map legacy local absolute output paths to repository-local paths.

    The original scripts saved figures and checkpoints under a local project
    tree.  The final file names are preserved, while generated artifacts are
    redirected to ``figures/`` or
    ``results/checkpoints/`` so the repository can run after download.
    """
    normalized = str(path).replace("\\", "/")
    legacy = PurePosixPath(normalized)
    parts = legacy.parts

    try:
        marker = parts.index("Code_NN")
        relative = parts[marker + 1 :]
    except ValueError:
        relative = parts

    lowered = [part.lower() for part in relative]
    if any(part.startswith("model_") for part in lowered):
        return checkpoint_path(*relative)

    suffix = legacy.suffix.lower()
    if suffix in {".eps", ".pdf", ".png", ".jpg", ".jpeg", ".svg"}:
        return figure_path(*relative)

    return result_path(*relative)
