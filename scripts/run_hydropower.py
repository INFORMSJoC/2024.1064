"""Run a hydropower model on a stored triangulation and instance.

Example:
    python scripts/run_hydropower.py adaptive/inst005.txt June
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pwlopt.hydropower.hpf_pwl.hydro_cc import HydroCC
from pwlopt.hydropower.hpf_pwl.hydro_dcc import HydroDCC
from pwlopt.hydropower.hpf_pwl.hydro_dlog import HydroDLog
from pwlopt.hydropower.hpf_pwl.hydro_gib import HydroGIB
from pwlopt.hydropower.hpf_pwl.hydro_inc import HydroInc
from pwlopt.hydropower.hpf_pwl.hydro_log_grid import GridTriangulation, HydroLogGrid
from pwlopt.hydropower.hpf_pwl.hydro_mc import HydroMC
from pwlopt.hydropower.hydro_data import HydroData

FORMULATIONS = {
    "cc": HydroCC,
    "dcc": HydroDCC,
    "dlog": HydroDLog,
    "gib": HydroGIB,
    "inc": HydroInc,
    "mc": HydroMC,
    "log_grid": HydroLogGrid,
}


def resolve_triangulation(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        candidates = (ROOT / "data" / "triangulations" / path, ROOT / path)
        path = next((candidate for candidate in candidates if candidate.is_file()), candidates[0])
    if not path.is_file():
        raise FileNotFoundError(f"Triangulation file not found: {path}")
    return path


def resolve_biclique_cover(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        candidates = (ROOT / "results" / "biclique_cover" / path, ROOT / path)
        path = next((candidate for candidate in candidates if candidate.is_file()), candidates[0])
    if not path.is_file():
        raise FileNotFoundError(f"Biclique-cover file not found: {path}")
    return path


def load_grid_triangulation(filename: Path) -> GridTriangulation:
    lines = [line.strip() for line in filename.read_text(encoding="utf-8").splitlines() if line.strip()]
    first_count, second_count, dx, dy = (int(value) for value in lines[0].split(","))
    expected_points = (dx + 1) * (dy + 1)
    expected_triangles = 2 * dx * dy

    if (first_count, second_count) == (expected_points, expected_triangles):
        npoints, ntriangles = first_count, second_count
    elif (first_count, second_count) == (expected_triangles, expected_points):
        ntriangles, npoints = first_count, second_count
    else:
        raise ValueError(
            f"Grid file {filename} has counts ({first_count}, {second_count}), "
            f"expected nodes/triangles ({expected_points}, {expected_triangles})"
        )

    points_start = 1
    points_end = points_start + npoints
    points = np.array([[float(value) for value in line.split(",")] for line in lines[points_start:points_end]])
    triangles_end = points_end + ntriangles
    triangles = [tuple(int(value) for value in line.split(",")) for line in lines[points_end:triangles_end]]
    diagonal_lines = lines[triangles_end : triangles_end + dx]
    index_lines = lines[triangles_end + dx : triangles_end + 2 * dx + 1]
    diagonal = np.array([[int(value) for value in line.split(",") if value] for line in diagonal_lines])
    indices = np.array([[int(value) for value in line.split(",") if value] for line in index_lines])

    if len(triangles) != ntriangles or diagonal.shape != (dx, dy) or indices.shape != (dx + 1, dy + 1):
        raise ValueError(f"Grid file {filename} has incomplete grid sections")
    if not np.array_equal(indices, np.arange(expected_points).reshape(dx + 1, dy + 1)):
        raise ValueError(f"Grid point indices in {filename} are not in the expected row-major order")

    xmin, ymin = points.min(axis=0)
    xmax, ymax = points.max(axis=0)
    expected_points = np.array(
        [
            [xmin + i * (xmax - xmin) / dx, ymin + j * (ymax - ymin) / dy]
            for i in range(dx + 1)
            for j in range(dy + 1)
        ]
    )
    if not np.allclose(points, expected_points):
        raise ValueError(f"Grid points in {filename} do not match the declared grid dimensions")
    return GridTriangulation(diagonal, dx, dy, 0, 1, 0, 1)


def resolve_month(month: str) -> Path:
    instances = ROOT / "data" / "hydro_instances"
    matches = [path for path in instances.iterdir() if path.is_dir() and path.name.lower() == month.lower()]
    if not matches:
        available = ", ".join(sorted(path.name for path in instances.iterdir() if path.is_dir()))
        raise FileNotFoundError(f"Unknown month {month!r}. Available months: {available}")
    return matches[0]


def load_data(month: str) -> HydroData:
    directory = resolve_month(month)
    return HydroData.from_file(
        str(directory / "plant.json"),
        str(directory / "price.json"),
        str(directory / "inflow.json"),
        str(directory / "pump_cost.json"),
    )


def solution_report(model, triangulation: Path, month: str, formulation: str) -> dict:
    try:
        triangulation_name = str(triangulation.relative_to(ROOT))
    except ValueError:
        triangulation_name = str(triangulation)
    return {
        "month": month,
        "formulation": formulation,
        "triangulation": triangulation_name,
        "objective": model.obj,
        "duration": model.dur,
        "nodes": model.nnodes,
        "rows": model.rows,
        "columns": model.cols,
        "binaries": model.binaries,
        "best_objective": model.bestobj,
        "best_bound": model.bestbound,
        "volume": model.vSol.tolist(),
        "flow": model.qSol.tolist(),
        "power": model.pSol.tolist(),
        "spillage": model.sSol.tolist(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "triangulation",
        help="Triangulation path, relative to data/triangulations (for example adaptive/inst005.txt)",
    )
    parser.add_argument("month", help="Hydropower instance month, for example June")
    parser.add_argument(
        "--formulation",
        choices=sorted(FORMULATIONS),
        default="cc",
        help="MILP formulation (default: cc)",
    )
    parser.add_argument(
        "--biclique-cover",
        type=Path,
        help="Read a GIB biclique cover, relative to results/biclique_cover",
    )
    parser.add_argument(
        "--grid",
        action="store_true",
        help="Run HydroLogGrid using the grid triangulation file (alias for --formulation log_grid)",
    )
    parser.add_argument("--output", type=Path, help="Write JSON results to this file instead of stdout")
    parser.add_argument("--time-limit", type=int, help="Solver time limit in seconds")
    parser.add_argument("--threads", type=int, default=-1, help="Number of solver threads (default: -1)")
    parser.add_argument("--quiet", action="store_true", help="Suppress solver output")
    args = parser.parse_args()

    if not args.quiet:
        logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    if args.biclique_cover and args.formulation != "gib":
        parser.error("--biclique-cover can only be used with --formulation gib")
    if args.grid and args.formulation not in {"cc", "log_grid"}:
        parser.error("--grid can only be combined with --formulation log_grid")

    triangulation = resolve_triangulation(args.triangulation)
    biclique_cover = resolve_biclique_cover(str(args.biclique_cover)) if args.biclique_cover else None
    data = load_data(args.month)
    is_grid = args.grid or args.formulation == "log_grid"
    formulation = "log_grid" if is_grid else args.formulation
    model = FORMULATIONS[formulation](data)
    if is_grid:
        model.setGridTrg(load_grid_triangulation(triangulation))
    else:
        model.readTrg(str(triangulation))
    if formulation == "gib":
        if biclique_cover:
            print(f"GIB: loading biclique cover from {biclique_cover}", file=sys.stderr)
        else:
            print("GIB: no biclique cover supplied; computing one", file=sys.stderr)
        model.buildPWL(biclique_file=str(biclique_cover) if biclique_cover else None)
        print(f"GIB: biclique cover ready ({len(model.B.reds)} bicliques)", file=sys.stderr)
    model.buildModel()
    model.solve(soltimelimit=args.time_limit, threads=args.threads, verbose=not args.quiet)

    result = json.dumps(solution_report(model, triangulation, args.month, formulation), indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(result + "\n", encoding="utf-8")
    else:
        print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())