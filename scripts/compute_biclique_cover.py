"""Compute a biclique cover for a stored triangulation.

Example:
    python scripts/compute_biclique_cover.py adaptive/inst005.txt \
        --output results/biclique_cover/adaptive/inst005.json
"""

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pwlopt.modeling.biclique_cover_IP import BicliqueCover
from pwlopt.utils import read_trg


def resolve_triangulation(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        candidates = (ROOT / "data" / "triangulations" / path, ROOT / path)
        path = next((candidate for candidate in candidates if candidate.is_file()), candidates[0])
    if not path.is_file():
        raise FileNotFoundError(f"Triangulation file not found: {path}")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "triangulation",
        help="Triangulation path, relative to data/triangulations (for example adaptive/inst005.txt)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Write the biclique cover as JSON to this file",
    )
    parser.add_argument(
        "--time-limit",
        type=int,
        default=100,
        help="Time limit in seconds for each biclique subproblem (default: 100)",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress solver progress logging")
    args = parser.parse_args()

    if args.time_limit <= 0:
        parser.error("--time-limit must be positive")

    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="[%(levelname)s] %(message)s",
    )

    try:
        triangulation_path = resolve_triangulation(args.triangulation)
    except FileNotFoundError as error:
        parser.error(str(error))

    triangles, points = read_trg(triangulation_path)
    cover = BicliqueCover(triangles, points)
    cover.solve(timer=True, limit=args.time_limit)

    print(
        f"Computed {len(cover.reds)} bicliques for "
        f"{len(points)} points and {len(triangles)} triangles"
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        cover.write_bicliques(args.output)
        print(f"Wrote biclique cover to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())