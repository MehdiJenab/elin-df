"""
Utilities for visualizing the phase-space distributions stored in QuietStart_Input.json.

The script reads the semi-structured output produced by the quiet-start generators,
extracts the electron/ion distribution functions (df_elc, df_ion), and renders a
2-D sketch f(x, v) for each species.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

MODULE_DIR = Path(__file__).resolve().parent


def _flatten_sections(payload: Dict) -> Dict:
    """Convert the list of single-key dicts under 'kinetic' into one mapping."""
    sections: Dict[str, object] = {}
    for block in payload.get("kinetic", []):
        sections.update(block)
    return sections


def _load_sections(path: Path) -> Dict:
    """Load QuietStart data and make sure the expected sections are available."""
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Failed to parse {path}. The file must be a valid JSON document."
        ) from exc

    sections = _flatten_sections(payload)
    for required in ("input_pattern", "velocity", "solitons"):
        if required not in sections:
            raise KeyError(f"Section '{required}' not found in {path}")
    return sections


def _iter_phase_rows(soliton_blocks: Iterable[Dict]) -> Iterable[Tuple[List[float], Dict]]:
    """
    Yield (scalars, distributions) pairs for each entry inside soliton_Data.

    scalars
        Ordered list of phase-point values (X, PHI, ...).
    distributions
        Dict such as {'df_elc': [...], 'df_ion': [...]}.
    """
    for block in soliton_blocks:
        rows = block.get("soliton_Data", [])
        for raw_entry in rows:
            scalars: List[float] = []
            distributions: Dict[str, List[float]] = {}
            for item in raw_entry:
                if isinstance(item, (int, float)):
                    scalars.append(float(item))
                elif isinstance(item, list):
                    scalars.extend(float(value) for value in item)
                elif isinstance(item, dict):
                    for key, value in item.items():
                        distributions[key] = value
            if scalars or distributions:
                yield scalars, distributions


def _extract_species_profiles(
    sections: Dict,
) -> Tuple[List[str], Dict[str, List[float]], Dict[str, Dict[float, np.ndarray]], Dict[str, str]]:
    """
    Build helpers needed for plotting:

    Returns
    -------
    pattern : list[str]
        Input pattern describing the order of scalar fields.
    velocities : dict
        Mapping velocity_<species> -> velocity grid.
    profiles : dict
        Mapping df_<species> -> {x_value: df(v)}.
    df_to_velocity : dict
        Mapping df_<species> -> velocity_<species>.
    """
    pattern: List[str] = sections["input_pattern"]
    try:
        x_index = pattern.index("X")
    except ValueError as exc:
        raise ValueError("Input pattern missing 'X' entry") from exc

    velocities: Dict[str, List[float]] = {}
    for entry in sections.get("velocity", []):
        velocities.update(entry)
    if not velocities:
        raise ValueError("Velocity grids are missing in the input file")

    df_to_velocity: Dict[str, str] = {}
    for vel_key in velocities:
        if vel_key.startswith("velocity_"):
            suffix = vel_key.split("velocity_", 1)[1]
            df_to_velocity[f"df_{suffix}"] = vel_key

    profiles: Dict[str, Dict[float, np.ndarray]] = {
        df_key: {} for df_key in df_to_velocity
    }

    for scalars, distributions in _iter_phase_rows(sections.get("solitons", [])):
        if len(scalars) <= x_index:
            raise ValueError("Phase-point record is missing the X coordinate")
        x_value = scalars[x_index]
        for df_key, samples in distributions.items():
            if df_key not in profiles:
                continue
            if x_value in profiles[df_key]:
                raise ValueError(
                    f"Multiple entries encountered for x={x_value} ({df_key})"
                )
            profiles[df_key][x_value] = np.asarray(samples, dtype=float)

    return pattern, velocities, profiles, df_to_velocity


def _build_grids(
    profiles: Dict[str, Dict[float, np.ndarray]],
    velocities: Dict[str, List[float]],
    df_to_velocity: Dict[str, str],
) -> Tuple[np.ndarray, Dict[str, Dict[str, np.ndarray]]]:
    """Convert sparse profile dicts into dense (velocity, x) grids."""
    x_positions = sorted({x for species in profiles.values() for x in species})
    if not x_positions:
        raise ValueError("No phase-space samples were found in soliton_Data")

    grids: Dict[str, Dict[str, np.ndarray]] = {}
    for df_key, samples_by_x in profiles.items():
        vel_key = df_to_velocity[df_key]
        velocity_axis = np.asarray(velocities[vel_key], dtype=float)
        n_vel = velocity_axis.size
        columns: List[np.ndarray] = []
        for x in x_positions:
            sample = samples_by_x.get(x)
            if sample is None:
                columns.append(np.full(n_vel, np.nan, dtype=float))
            else:
                vector = np.asarray(sample, dtype=float)
                if vector.size != n_vel:
                    raise ValueError(
                        f"{df_key} at x={x} has {vector.size} points, "
                        f"but velocity grid '{vel_key}' has {n_vel}"
                    )
                columns.append(vector)
        grids[df_key] = {
            "velocities": velocity_axis,
            "values": np.vstack(columns).T,  # shape (n_vel, n_x)
        }
    return np.asarray(x_positions, dtype=float), grids


def _plot_grids(
    x_positions: np.ndarray,
    grids: Dict[str, Dict[str, np.ndarray]],
    cmap: str,
    output: Path | None,
    show: bool,
) -> None:
    """Render a panel for each species."""
    if not grids:
        raise ValueError("No df_* entries matched the available velocity grids")

    ordered_keys = sorted(grids)
    fig, axes = plt.subplots(
        1, len(ordered_keys), figsize=(6 * len(ordered_keys), 4), squeeze=False
    )

    for ax, df_key in zip(axes[0], ordered_keys):
        payload = grids[df_key]
        vel_axis = payload["velocities"]
        values = payload["values"]
        mesh_x, mesh_v = np.meshgrid(x_positions, vel_axis)
        pcm = ax.pcolormesh(mesh_x, mesh_v, values, shading="auto", cmap=cmap)
        species = df_key.split("df_", 1)[1]
        ax.set_title(f"{species} distribution")
        ax.set_xlabel("x")
        ax.set_ylabel("velocity")
        fig.colorbar(pcm, ax=ax, label="f(x, v)")

    fig.tight_layout()
    if output:
        fig.savefig(output, dpi=300)
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_quiet_start(
    input_path: Path, cmap: str = "viridis", output: Path | None = None, show: bool = False
) -> None:
    """High-level convenience wrapper used by the CLI."""
    sections = _load_sections(input_path)
    _, velocities, profiles, df_to_velocity = _extract_species_profiles(sections)
    x_positions, grids = _build_grids(profiles, velocities, df_to_velocity)
    _plot_grids(x_positions, grids, cmap=cmap, output=output, show=show)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render df_elc/df_ion as 2-D phase-space sketches.",
    )
    default_input = MODULE_DIR / "QuietStart_Input.json"
    parser.add_argument(
        "--input",
        type=Path,
        default=default_input,
        help=f"QuietStart_Input.json file (default: {default_input})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path to save the figure (e.g. plots/df.png).",
    )
    parser.add_argument(
        "--cmap",
        default="viridis",
        help="Matplotlib colormap to use (default: viridis).",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the plot window instead of just saving the figure.",
    )
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()
    plot_quiet_start(args.input, cmap=args.cmap, output=args.output, show=args.show)


if __name__ == "__main__":
    main()
