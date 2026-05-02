#!/usr/bin/env python3
"""Plot metrics across multiple training runs.

This script searches recursively for ``metrics.jsonl`` files (produced by
``tools.Logger``). Run folders are expected as ``YYYYMMDD_backbone_game_seed``
on the first path segment under ``--root``. For each game and metric it writes
one figure: one line per backbone, each line is the mean over seeds (curves
interpolated to a common step grid). Output layout: ``output_dir/<game>/<metric>.png``.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot all run metrics from metrics.jsonl files.")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("logdir"),
        help="Root directory to search recursively for metrics.jsonl files (default: logdir).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("plots"),
        help="Directory where plot PNGs will be written (default: plots).",
    )
    parser.add_argument(
        "--metrics",
        type=str,
        default="",
        help=(
            "Comma-separated regex patterns to keep metrics, e.g. "
            "'episode/score,train/loss/.*,fps/fps'. If empty, all metrics are plotted."
        ),
    )
    parser.add_argument(
        "--exclude",
        type=str,
        default="",
        help="Comma-separated regex patterns to skip metrics.",
    )
    parser.add_argument(
        "--smooth",
        type=float,
        default=0.0,
        help="EMA smoothing factor in [0, 1). 0 disables smoothing.",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=1500,
        help="Max points per run after downsampling (default: 1500).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=140,
        help="Output image DPI (default: 140).",
    )
    parser.add_argument(
        "--legend-limit",
        type=int,
        default=20,
        help="Show legend only when number of runs <= legend-limit (default: 20).",
    )
    return parser.parse_args()


def split_patterns(raw: str) -> List[re.Pattern]:
    if not raw.strip():
        return []
    return [re.compile(part.strip()) for part in raw.split(",") if part.strip()]


def metric_allowed(name: str, include: Sequence[re.Pattern], exclude: Sequence[re.Pattern]) -> bool:
    if include and not any(p.search(name) for p in include):
        return False
    if exclude and any(p.search(name) for p in exclude):
        return False
    return True


def sanitize_filename(name: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", name)
    return safe.strip("_") or "metric"


def ema(values: np.ndarray, alpha: float) -> np.ndarray:
    if alpha <= 0.0:
        return values
    out = np.empty_like(values, dtype=np.float64)
    out[0] = float(values[0])
    for i in range(1, len(values)):
        out[i] = alpha * float(values[i]) + (1.0 - alpha) * out[i - 1]
    return out


def downsample_indices(n: int, max_points: int) -> np.ndarray:
    if n <= max_points:
        return np.arange(n)
    # Keep first and last point while spacing intermediate points evenly.
    idx = np.linspace(0, n - 1, max_points, dtype=np.int64)
    idx[0] = 0
    idx[-1] = n - 1
    return np.unique(idx)


class ParsedRun(NamedTuple):
    date: str
    backbone: str
    game: str
    seed: str


def parse_experiment_folder(run_name: str) -> Optional[ParsedRun]:
    """Parse ``YYYYMMDD_backbone_game_seed`` from the first path segment under root.

    Example: ``20260501_gru_bank_heist_100`` → date, gru, bank_heist, 100.
    """
    first = Path(run_name).parts[0] if run_name else ""
    parts = first.split("_")
    if len(parts) < 4:
        return None
    if not re.fullmatch(r"\d{8}", parts[0]):
        return None
    if not parts[-1].isdigit():
        return None
    date, backbone, seed = parts[0], parts[1], parts[-1]
    game = "_".join(parts[2:-1])
    if not game:
        return None
    return ParsedRun(date=date, backbone=backbone, game=game, seed=seed)


def average_curves(curves: List[Tuple[np.ndarray, np.ndarray]]) -> Tuple[np.ndarray, np.ndarray]:
    """Interpolate each curve onto a common step grid and return elementwise mean."""
    if not curves:
        return np.array([]), np.array([])
    if len(curves) == 1:
        return curves[0]
    mins = [float(s.min()) for s, _ in curves]
    maxs = [float(s.max()) for s, _ in curves]
    lo, hi = max(mins), min(maxs)
    if hi < lo or not math.isfinite(lo) or not math.isfinite(hi):
        return np.array([]), np.array([])
    all_steps = np.unique(np.concatenate([s for s, _ in curves]))
    all_steps = all_steps[(all_steps >= lo) & (all_steps <= hi)]
    if len(all_steps) == 0:
        return np.array([]), np.array([])
    ys = []
    for s, v in curves:
        ys.append(np.interp(all_steps, s, v))
    mean_y = np.mean(np.vstack(ys), axis=0)
    return all_steps, mean_y


def load_metrics_file(path: Path) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    # metric_name -> (steps, values)
    steps_map: Dict[str, List[float]] = defaultdict(list)
    vals_map: Dict[str, List[float]] = defaultdict(list)

    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                print(f"[warn] {path}:{line_no} invalid JSON, skipping line")
                continue
            if "step" not in row:
                continue
            step = row["step"]
            if not isinstance(step, (int, float)):
                continue
            for key, value in row.items():
                if key == "step":
                    continue
                if isinstance(value, (int, float)) and not (isinstance(value, float) and math.isnan(value)):
                    steps_map[key].append(float(step))
                    vals_map[key].append(float(value))

    out: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for key in vals_map:
        steps = np.asarray(steps_map[key], dtype=np.float64)
        values = np.asarray(vals_map[key], dtype=np.float64)
        if len(steps) == 0:
            continue
        order = np.argsort(steps, kind="mergesort")
        out[key] = (steps[order], values[order])
    return out


def main() -> None:
    args = parse_args()
    include = split_patterns(args.metrics)
    exclude = split_patterns(args.exclude)

    metrics_files = sorted(args.root.rglob("metrics.jsonl"))
    if not metrics_files:
        raise SystemExit(f"No metrics.jsonl files found under: {args.root}")

    runs = []  # [(run_name, path, metric_dict)]
    for mf in metrics_files:
        metric_dict = load_metrics_file(mf)
        if metric_dict:
            run_name = str(mf.parent.relative_to(args.root)) if mf.parent != args.root else mf.parent.name
            runs.append((run_name, mf, metric_dict))

    if not runs:
        raise SystemExit("No plottable metrics found in discovered metrics.jsonl files.")

    all_metric_names = sorted({m for _, _, d in runs for m in d.keys()})
    kept_metric_names = [m for m in all_metric_names if metric_allowed(m, include, exclude)]
    if not kept_metric_names:
        raise SystemExit("No metrics matched the include/exclude filters.")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Found runs: {len(runs)}")
    print(f"Found metrics: {len(all_metric_names)}")
    print(f"Plotting metrics: {len(kept_metric_names)}")

    # Group: (date, game, backbone) -> list of (run_name, metric_dict) per seed
    grouped: Dict[
        Tuple[str, str, str], List[Tuple[str, Dict[str, Tuple[np.ndarray, np.ndarray]]]]
    ] = defaultdict(list)
    unparsed: List[str] = []
    for run_name, _, metric_dict in runs:
        parsed = parse_experiment_folder(run_name)
        if parsed is None:
            unparsed.append(run_name)
            continue
        key = (parsed.date, parsed.game, parsed.backbone)
        # Defer metric extraction until we know which metrics exist per group
        grouped[key].append((run_name, metric_dict))

    if unparsed:
        for name in unparsed[:10]:
            print(f"[warn] run folder not YYYYMMDD_backbone_game_seed (skipped): {name}")
        if len(unparsed) > 10:
            print(f"[warn] ... and {len(unparsed) - 10} more unparsed runs")

    if not grouped:
        raise SystemExit(
            "No runs matched YYYYMMDD_backbone_game_seed on the first path segment; "
            "nothing to plot by game/backbone."
        )

    # Per (date, game, backbone): for each metric, average curves over seeds (one metric_dict per seed)
    games = sorted({g for _, g, _ in grouped})
    for metric in kept_metric_names:
        for game in games:
            fig, ax = plt.subplots(figsize=(9.0, 5.0))
            n_curves = 0
            # Collect lines: label -> (steps, values) after seed average per (date, backbone)
            series_keys = sorted({(d, b) for d, g, b in grouped if g == game})
            dates_for_game = {d for d, g, _ in grouped.keys() if g == game}
            for date, backbone in series_keys:
                seed_curves: List[Tuple[np.ndarray, np.ndarray]] = []
                for (d, g, b), members in grouped.items():
                    if d != date or g != game or b != backbone:
                        continue
                    for _run_name, metric_dict in members:
                        if metric not in metric_dict:
                            continue
                        steps, values = metric_dict[metric]
                        if len(steps) == 0:
                            continue
                        idx = downsample_indices(len(steps), args.max_points)
                        x = steps[idx].copy()
                        y = values[idx].copy()
                        if args.smooth > 0.0 and len(y) > 1:
                            y = ema(y, args.smooth)
                        seed_curves.append((x, y))
                if not seed_curves:
                    continue
                avg_x, avg_y = average_curves(seed_curves)
                if len(avg_x) == 0:
                    continue
                if len(dates_for_game) > 1:
                    label = f"{backbone} ({date})"
                else:
                    label = backbone
                ax.plot(avg_x, avg_y, linewidth=1.1, alpha=0.9, label=label)
                n_curves += 1

            if n_curves == 0:
                plt.close(fig)
                continue

            ax.set_title(f"{game} — {metric} (mean over seeds)")
            ax.set_xlabel("step")
            ax.set_ylabel(metric)
            ax.grid(True, linestyle="--", alpha=0.3)
            if n_curves <= args.legend_limit:
                ax.legend(fontsize=8, loc="best")
            fig.tight_layout()

            game_dir = args.output_dir / sanitize_filename(game)
            game_dir.mkdir(parents=True, exist_ok=True)
            out_path = game_dir / f"{sanitize_filename(metric)}.png"
            fig.savefig(out_path, dpi=args.dpi)
            plt.close(fig)

    print(f"Saved plots to: {args.output_dir} (one subfolder per game)")


if __name__ == "__main__":
    main()
