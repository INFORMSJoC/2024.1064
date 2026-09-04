from itertools import combinations

import numpy as np
from numpy.random import Generator, default_rng

from ..utils import with_logger


def segment_intersection(a, b, c, d):
    r = b - a
    s = d - c
    cross = r[0] * s[1] - r[1] * s[0]

    if abs(cross) < 1e-12:
        return None

    t = ((c - a)[0] * s[1] - (c - a)[1] * s[0]) / cross
    u = ((c - a)[0] * r[1] - (c - a)[1] * r[0]) / cross

    if 0 <= t <= 1 and 0 <= u <= 1:
        return a + t * r

    return None

class PoissonDiskSampler:
    def __init__(self, triangle: object, radius: float):
        self.radius = radius
        self.samples = []
        if radius <= 0:
            raise ValueError("radius must be positive")

        self.triangle = triangle

        self.lo = np.min(triangle.P, axis=0)
        self.hi = np.max(triangle.P, axis=0)

        # Ebeida: base-cell diagonal = radius
        self.s0 = radius / np.sqrt(2.0)

        width = self.hi - self.lo
        self.nx = int(np.ceil(width[0] / self.s0))
        self.ny = int(np.ceil(width[1] / self.s0))

        self.grid = {}


    def _base_cell(self, p):
        return (
            int(np.floor((p[0] - self.lo[0]) / self.s0)),
            int(np.floor((p[1] - self.lo[1]) / self.s0)),
        )

    def _base_cell_bounds(self, i, j):
        lower = self.lo + np.array([i * self.s0, j * self.s0])
        return lower, lower + self.s0

    def _cell_bounds(self, i, j, level):
        side = self.s0 / (2**level)
        lower = self.lo + np.array([i * side, j * side])
        return lower, lower + side

    def _disk_free(self, p):
        i, j = self._base_cell(p)
        r2 = self.radius**2

        for di in range(-2, 3):
            for dj in range(-2, 3):
                idx = self.grid.get((i + di, j + dj))
                if idx is None:
                    continue

                q = self.samples[idx]
                if np.sum((p - q) ** 2) < r2:
                    return False

        return True

    def _cell_covered(self, lo, hi):
        corners = np.array([
            [lo[0], lo[1]],
            [hi[0], lo[1]],
            [lo[0], hi[1]],
            [hi[0], hi[1]],
        ])

        center = 0.5 * (lo + hi)
        i, j = self._base_cell(center)
        r2 = self.radius**2

        for di in range(-2, 3):
            for dj in range(-2, 3):
                idx = self.grid.get((i + di, j + dj))
                if idx is None:
                    continue

                p = self.samples[idx]

                if np.all(np.sum((corners - p) ** 2, axis=1) <= r2):
                    return True

        return False

    def sample(
        self,
        active_darts: float = 0.5,
        max_level: int = 10,
        rng: Generator | None = None,
    ):
        if rng is None:
            rng = default_rng()

        active = []

        # Initial cells
        for i in range(self.nx):
            for j in range(self.ny):
                lo, hi = self._base_cell_bounds(i, j)

                if self.triangle._cell_intersects(lo, hi):
                    active.append((i, j))

        level = 0

        while active:
            n_darts = max(
                1,
                int(np.ceil(active_darts * len(active))),
            )

            for _ in range(n_darts):
                if not active:
                    break

                pos = rng.integers(len(active))
                i, j = active[pos]

                # Already occupied base cell
                if (i, j) in self.grid:
                    active[pos] = active[-1]
                    active.pop()
                    continue

                lo, hi = self._cell_bounds(i, j, level)
                p = rng.uniform(lo, hi)

                if not self.triangle._contains(p):
                    continue

                if not self._disk_free(p):
                    continue

                # Accept sample
                idx = len(self.samples)
                self.samples.append(p)

                base_idx = self._base_cell(p)

                if base_idx in self.grid:
                    raise RuntimeError(
                        "Base grid contains two samples."
                    )

                self.grid[base_idx] = idx

                active[pos] = active[-1]
                active.pop()

            if not active or level >= max_level:
                break

            # Refine
            new_active = []

            for i, j in active:
                for ci, cj in (
                    (2 * i, 2 * j),
                    (2 * i + 1, 2 * j),
                    (2 * i, 2 * j + 1),
                    (2 * i + 1, 2 * j + 1),
                ):
                    lo, hi = self._cell_bounds(
                        ci, cj, level + 1
                    )

                    if not self.triangle._cell_intersects(lo, hi):
                        continue

                    if self._cell_covered(lo, hi):
                        continue

                    new_active.append((ci, cj))

            active = new_active
            level += 1

        return np.asarray(self.samples)



@with_logger        
class EdgeSampler:
    def __init__(self, triangle: object, edge: tuple[int, int], radius: float):
        self.radius = radius
        self.samples = np.empty((0, 2))
        if radius <= 0:
            raise ValueError("radius must be positive")
        self.edge = edge
        self.triangle = triangle

    def sample(self):
        p1 = self.triangle.pts[self.edge[0]]
        p2 = self.triangle.pts[self.edge[1]]
        distance = np.linalg.norm(p2 - p1)
        n_intervals = int(np.ceil(distance / self.radius))
        points = np.linspace(p1, p2, n_intervals + 1)
        self.samples = np.vstack((self.samples, points))
        return self.samples        