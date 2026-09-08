from matplotlib.patches import Circle
from matplotlib.pyplot import subplots
from numpy import arctan2, array, degrees, dot, min, sum, vstack
from numpy.linalg import norm, solve
from scipy.spatial import Voronoi

from ..utils import with_logger
from .sampling import EdgeSampler
from .sampling_cpp import PoissonDiskSampler


@with_logger
class Triangle:
    def __init__(self, idxs, pts):
        self.logger.debug(
            "Initialize Triangle with indices %s and points %s", idxs, pts
        )
        self.idxs = tuple(sorted(int(i) for i in idxs))
        self.pts = pts
        self._circumcenter = None
        self._angles = None
        self._area = None
        self._sample = None
        self._edge_sample = None
        self._radius = None
        self._edges = None

    def __str__(self):
        return f"Triangle{self.idxs}"

    @property
    def P(self):
        return self.pts[list(self.idxs)]

    @property
    def edges(self):
        if self._edges is None:
            i, j, k = sorted(self.idxs)
            self._edges = (
                (i, j),
                (j, k),
                (i, k),
            )
        return self._edges

    def __hash__(self):
        return hash(self.idxs)

    def __eq__(self, other):
        return isinstance(other, Triangle) and self.idxs == other.idxs

    def sample(self, r: float):
        if self._sample is None or self._radius != r:
            if self._sample is None:
                self.logger.info(
                    "Triangle is not sampled yet, Maximal Poisson Disk Sampling with radius %.5f",
                    r,
                )
            elif self._radius != r:
                self.logger.info(
                    "Current sample radius (%.5f) differs from required sample radius %.5f, start resampling",
                    self._radius,
                    r,
                )
            self._radius = r
            sampler = PoissonDiskSampler(self, r)
            # edge_sampler = EdgeSampler(self, r)
            self._sample = sampler.sample()
            # self._sample = vstack((self._sample, edge_sampler.sample()))
        return self._sample

    def edge_sample(self, r: float):
        if self._edge_sample is None or self._radius != r:
            if self._edge_sample is None:
                self.logger.info(
                    "Triangle edges not sampled yet, Maximal Poisson Disk Sampling with radius %.5f",
                    r,
                )
            elif self._radius != r:
                self.logger.info(
                    "Current edge sample radius (%.5f) differs from required sample radius %.5f, start resampling",
                    self._radius,
                    r,
                )
            self._edge_sample = {}
            for e in self.edges:
                edge_sampler = EdgeSampler(self, e, r)
                self._edge_sample[e] = edge_sampler.sample()
        return self._edge_sample


    @property
    def circumcenter(self):
        if self._circumcenter is None:
            a, b, c = self.P
            A = b - a
            B = c - a
            AdotA = dot(A, A)
            BdotB = dot(B, B)
            cross = A[0] * B[1] - A[1] * B[0]

            if abs(cross) < 1e-12:
                # Degenerate triangle: return average as fallback
                return (a + b + c) / 3.0

            self._circumcenter = a + (
                BdotB * array([-A[1], A[0]]) - AdotA * array([-B[1], B[0]])
            ) / (2 * cross)
        return self._circumcenter

    @property
    def angles(self):
        if self._angles is None:
            a, b, c = self.P

            # Vectors at each vertex
            ab = b - a
            ac = c - a

            ba = a - b
            bc = c - b

            ca = a - c
            cb = b - c

            def angle(u, v):
                cross = u[0] * v[1] - u[1] * v[0]
                dotprod = dot(u, v)
                return arctan2(abs(cross), dotprod)

            angles = [
                angle(ab, ac),
                angle(ba, bc),
                angle(ca, cb),
            ]
            self._angles = angles
        return self._angles

    @property
    def min_angle(self):
        return min(self.angles)

    @property
    def min_angle_deg(self):
        return degrees(self.min_angle)

    @property
    def area(self):
        """Compute area of triangle"""
        if self._area is None:
            x1, y1 = self.P[0]
            x2, y2 = self.P[1]
            x3, y3 = self.P[2]
            self._area = 0.5 * abs(
                x1 * y2 + x2 * y3 + x3 * y1 - (y1 * x2 + y2 * x3 + y3 * x1)
            )
        return self._area

    def in_circumcircle(self, p, tol=1e-12):
        a, b, c = self.P

        a = a - p
        b = b - p
        c = c - p

        det = (
            dot(a, a) * (b[0] * c[1] - b[1] * c[0])
            - dot(b, b) * (a[0] * c[1] - a[1] * c[0])
            + dot(c, c) * (a[0] * b[1] - a[1] * b[0])
        )

        orientation = (self.P[1][0] - self.P[0][0]) * (self.P[2][1] - self.P[0][1]) - (
            self.P[1][1] - self.P[0][1]
        ) * (self.P[2][0] - self.P[0][0])

        return det > tol if orientation > 0 else det < -tol

    def _contains(self, p):
        """Return True if p is inside or on the boundary."""

        a, b, c = self.P

        def cross(u, v):
            return u[0] * v[1] - u[1] * v[0]

        c1 = cross(b - a, p - a)
        c2 = cross(c - b, p - b)
        c3 = cross(a - c, p - c)

        return (c1 >= 0 and c2 >= 0 and c3 >= 0) or (c1 <= 0 and c2 <= 0 and c3 <= 0)

    def _cell_fully_inside(self, lo, hi):
        """Return True if the entire square [lo, hi] is inside triangle."""

        x0, y0 = lo
        x1, y1 = hi

        corners = array(
            [
                [x0, y0],
                [x1, y0],
                [x0, y1],
                [x1, y1],
            ]
        )

        return all(self._contains(p) for p in corners)

    def _cell_intersects(self, lo, hi):
        """Return True if a square intersects the triangle."""

        x0, y0 = lo
        x1, y1 = hi

        corners = array(
            [
                [x0, y0],
                [x1, y0],
                [x0, y1],
                [x1, y1],
            ]
        )

        # Any triangle vertex inside the square
        if any(x0 <= p[0] <= x1 and y0 <= p[1] <= y1 for p in self.P):
            return True

        # Any square corner inside the triangle
        if any(self._contains(p) for p in corners):
            return True

        # Triangle edge / square edge intersections
        triangle_edges = [
            (self.P[0], self.P[1]),
            (self.P[1], self.P[2]),
            (self.P[2], self.P[0]),
        ]

        square_edges = [
            (corners[0], corners[1]),
            (corners[0], corners[2]),
            (corners[1], corners[3]),
            (corners[2], corners[3]),
        ]

        for a, b in triangle_edges:
            for c, d in square_edges:
                if self._segments_intersect(a, b, c, d):
                    return True

        return False

    @staticmethod
    def _segments_intersect(a, b, c, d):
        """Return True if two closed line segments intersect."""

        def orientation(p, q, r):
            return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])

        o1 = orientation(a, b, c)
        o2 = orientation(a, b, d)
        o3 = orientation(c, d, a)
        o4 = orientation(c, d, b)

        return (o1 * o2 <= 0) and (o3 * o4 <= 0)

    @staticmethod
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

    def is_covered(self, radius, tol=1e-6):
        points = self.sample(radius)
        vor = Voronoi(points)

        # Candidate points: triangle vertices + Voronoi vertices inside triangle
        candidates = list(self.P)

        for v in vor.vertices:
            if self._contains(v):
                candidates.append(v)

        # Voronoi ridge ∩ triangle edge
        edges = [
            (self.P[0], self.P[1]),
            (self.P[1], self.P[2]),
            (self.P[2], self.P[0]),
        ]

        for ridge, vertices in zip(vor.ridge_points, vor.ridge_vertices):
            if -1 in vertices:
                continue

            a, b = vor.vertices[vertices]
            for c, d in edges:
                p = self.segment_intersection(a, b, c, d)
                if p is not None:
                    candidates.append(p)

        r2 = radius**2

        for x in candidates:
            nearest_d2 = min(sum((points - x) ** 2, axis=1))
            if nearest_d2 > r2 + tol:
                self.logger.warning(
                    "Nearest point is further than %.5f + tol%.5f: %.5f",
                    r2,
                    tol,
                    nearest_d2,
                )
                return False

        return True

    def plot_sample(self, fig=None, ax=None):
        if fig is None or ax is None:
            fig, ax = subplots(figsize=(15, 10))
        if self._sample is None:
            raise ValueError("Triangle not sampled yet!")
        sample = self._sample

        ax.scatter(sample[:, 0], sample[:, 1], s=10 * self._radius, c="blue")

        self.plot_triangle(fig, ax)

        for p in self.sample(self._radius):
            circle = Circle(
                p,
                radius=self._radius,
                facecolor="blue",
                edgecolor="blue",
                alpha=0.1,
            )
            ax.add_patch(circle)

        ax.set_aspect("equal")
        return fig, ax

    def plot_triangle(self, fig=None, ax=None):
        if fig is None or ax is None:
            fig, ax = subplots(figsize=(15, 10))

        triangle_plot = vstack([self.P, self.P[0]])

        ax.plot(triangle_plot[:, 0], triangle_plot[:, 1], linewidth=1, color="k")

        ax.set_aspect("equal")
        return fig, ax

    def lin_grad(self, fun):
        p1, p2, p3 = self.P
        vals = fun(self.P)

        if vals.shape != (3,):
            raise ValueError("values must have shape (3,)")

        A = array(
            [
                p2 - p1,
                p3 - p1,
            ]
        )

        return solve(A, vals[1:] - vals[0])

    def get_lip(self, fun):
        return norm(self.lin_grad(fun))

    def lint(self, pts, fun):
        grad = self.lin_grad(fun)
        vals = fun(self.P)
        self.logger.debug(
            "shapes -> vals[0]: %s, (pts-self.P[0]): %s, grad: %s",
            vals[0].shape,
            (pts - self.P[0]).shape,
            grad.shape,
        )
        return vals[0] + (pts - self.P[0]) @ grad
