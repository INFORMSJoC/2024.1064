from matplotlib.pyplot import subplots
from numpy import array, asarray, ndarray, vstack
from scipy.spatial import Delaunay

from ..utils import with_logger
from .triangle import Triangle


@with_logger
class Triangulation:
    def __init__(self, trg: list[Triangle], pts: array):
        self.trg = trg
        self.pts = pts
        self.edges_to_triangles = {}
        self._build_edge_map()

    def _build_edge_map(self) -> None:
        for triangle in self.trg:
            for edge in triangle.edges:
                edge = tuple(sorted(edge))
                self.edges_to_triangles.setdefault(edge, set()).add(triangle)

    @classmethod
    def from_points(cls, points: ndarray) -> "Triangulation":
        points = asarray(points, dtype=float)

        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError("points must have shape (n, 2)")

        if len(points) < 3:
            raise ValueError("At least 3 points are required")

        delaunay = Delaunay(points)

        triangles = [Triangle(list(indices), points) for indices in delaunay.simplices]

        return cls(triangles, points)

    @property
    def edges(self):
        return self.edges_to_triangles.keys()

    @property
    def min_angle(self):
        return min(t.min_angle for t in self.trg)

    def _add_triangle(self, triangle: Triangle) -> None:
        self.logger.info("Add new triangle: %s", triangle)

        self.trg.append(triangle)

        for edge in triangle.edges:
            edge = tuple(sorted(edge))
            self.edges_to_triangles.setdefault(edge, set()).add(triangle)

    def _remove_triangle(self, triangle: Triangle) -> None:
        self.trg.remove(triangle)

        for edge in triangle.edges:
            edge = tuple(sorted(edge))
            triangles = self.edges_to_triangles[edge]

            triangles.remove(triangle)

            if not triangles:
                del self.edges_to_triangles[edge]

    def _get_opposite_edges(self, point_idx: int) -> list[tuple[int, int]]:
        edges = []
        for t in self.trg:
            if point_idx not in t.idxs:
                continue
            for e in t.edges:
                if point_idx not in e:
                    edges.append(e)
        return edges

    def _repair_delaunay_property(self, point_idx: int) -> None:
        """Restore the Delaunay property after inserting point_idx."""
        self.logger.info("Try repairing Delaunay property for idx = %d", point_idx)
        stack = [edge for edge in self._get_opposite_edges(point_idx)]

        while stack:
            edge = tuple(sorted(stack.pop()))

            # Boundary edges cannot be flipped.
            adjacent = self.edges_to_triangles.get(edge, set())
            if len(adjacent) != 2:
                continue

            t1, t2 = adjacent

            if point_idx not in t1.idxs and point_idx not in t2.idxs:
                continue

            i, j = edge

            k = next(x for x in t1.idxs if x not in edge)
            l = next(x for x in t2.idxs if x not in edge)

            if not t1.in_circumcircle(self.pts[l]):
                self.logger.info(
                    "Skip, point %d is not in the circumcircle of triangle %s", l, t1
                )
                continue

            self.logger.info(
                "Flip non-Delaunay edge (%d, %d) -> (%d, %d)",
                i,
                j,
                k,
                l,
            )

            self._remove_triangle(t1)
            self._remove_triangle(t2)

            new_t1 = Triangle((k, l, i), self.pts)
            new_t2 = Triangle((k, l, j), self.pts)

            self._add_triangle(new_t1)
            self._add_triangle(new_t2)

            for new_edge in new_t1.edges + new_t2.edges:
                new_edge = tuple(sorted(new_edge))

                if k in new_edge or l in new_edge and new_edge != tuple(sorted((k, l))):
                    stack.append(new_edge)

    def _insert_on_edge(self, p: ndarray, edge: tuple[int, int]) -> None:
        """Insert point p lying on an existing edge."""
        i, j = edge
        point_idx = len(self.pts)

        adjacent = list(self.edges_to_triangles[tuple(sorted(edge))])

        # Add the point first so newly created triangles can reference it.
        self.logger.info("Insert new point %s with index %d", p, len(self.pts))
        self.pts = vstack((self.pts, p))

        if len(adjacent) == 1:
            triangle = adjacent[0]

            k = next(idx for idx in triangle.idxs if idx not in (i, j))

            self._remove_triangle(triangle)

            self._add_triangle(Triangle((i, point_idx, k), self.pts))
            self._add_triangle(Triangle((point_idx, j, k), self.pts))

        elif len(adjacent) == 2:
            t1, t2 = adjacent
            k = next(idx for idx in t1.idxs if idx not in (i, j))
            l = next(idx for idx in t2.idxs if idx not in (i, j))

            self._remove_triangle(t1)
            self._remove_triangle(t2)

            self._add_triangle(Triangle((i, point_idx, k), self.pts))
            self._add_triangle(Triangle((point_idx, j, k), self.pts))
            self._add_triangle(Triangle((i, l, point_idx), self.pts))
            self._add_triangle(Triangle((point_idx, l, j), self.pts))

        else:
            raise ValueError(f"Edge {edge} has {len(adjacent)} adjacent triangles")

        self._repair_delaunay_property(len(self.pts) - 1)

    def insert_point(self, p: ndarray, on_edge: tuple[int, int] | None = None) -> None:
        if on_edge is not None:
            self._insert_on_edge(p, on_edge)
            return

        point_idx = len(self.pts)

        # Find cavity
        bad = [triangle for triangle in self.trg if triangle.in_circumcircle(p)]

        # Find cavity boundary
        edge_count: dict[tuple[int, int], int] = {}

        for triangle in bad:
            for edge in triangle.edges:
                edge_count[edge] = edge_count.get(edge, 0) + 1

        boundary = [edge for edge, count in edge_count.items() if count == 1]

        # Remove bad triangles
        bad_set = set(bad)
        for t in bad_set:
            self.logger.info("Remove triangle: %s", t)
            self._remove_triangle(t)

        # Add point
        self.logger.info("Insert new point %s with index %d", p, len(self.pts))
        self.pts = vstack((self.pts, p))

        # Retriangulate cavity
        for i, j in boundary:
            new_trg = Triangle((i, j, point_idx), self.pts)
            self._add_triangle(new_trg)
            # self.trg.append(new_trg)

    def plot_triangulation(self, fig=None, ax=None):
        if fig is None or ax is None:
            fig, ax = subplots(figsize=(15, 10))

        for triangle in self.trg:
            triangle.plot_triangle(fig, ax)

        ax.set_aspect("equal")
        return fig, ax

    def is_delaunay(self, tol=1e-12) -> bool:
        """Validate that the triangulation satisfies the Delaunay condition."""

        for triangle in self.trg:
            for edge in triangle.edges:
                adjacent = [
                    t
                    for t in self.trg
                    if t is not triangle and edge[0] in t.idxs and edge[1] in t.idxs
                ]

                if len(adjacent) != 1:
                    continue

                other = adjacent[0]

                # Vertex opposite the shared edge
                opposite = next(idx for idx in other.idxs if idx not in edge)

                if triangle.in_circumcircle(
                    self.pts[opposite],
                    tol=tol,
                ):
                    return False

        return True

    def is_edge_border(self, edge: tuple[int, int]) -> bool:
        edge = tuple(sorted(edge))
        return len(self.edges_to_triangles[edge]) == 1

    def split_segment(self, seg: tuple[int, int]) -> None:
        """Split a segment at its midpoint."""
        i, j = seg
        p = 0.5 * (self.pts[i] + self.pts[j])
        self.logger.info("Split segment (%d, %d) at %s", i, j, p)
        self.insert_point(p, on_edge=seg)

    def split_triangle(self, triangle) -> None:
        p = triangle.circumcenter
        self.logger.info(
            "Split triangle %s at %s (min angle: %.3f)",
            triangle,
            p,
            triangle.min_angle_deg,
        )
        self.insert_point(p)

    def is_segment_encroached(self, seg, tol=1e-12) -> bool:
        i, j = seg

        a = self.pts[i]
        b = self.pts[j]

        midpoint = 0.5 * (a + b)
        radius_sq = 0.25 * ((a - b) @ (a - b))

        for k, p in enumerate(self.pts):
            if k == i or k == j:
                continue

            distance_sq = (p - midpoint) @ (p - midpoint)

            if distance_sq < radius_sq - tol:
                return True

        return False

    def point_encroaches(
        self, p: ndarray, segments: list[tuple[int, int]], tol: float = 1e-12
    ) -> tuple[bool, tuple[int, int]]:
        for seg in segments:
            i, j = seg

            a = self.pts[i]
            b = self.pts[j]

            midpoint = 0.5 * (a + b)
            radius_sq = 0.25 * ((a - b) @ (a - b))

            distance_sq = (p - midpoint) @ (p - midpoint)

            if distance_sq < radius_sq - tol:
                return True, seg
        return False, None

    def delaunay_refine(
        self,
        min_angle_lb: float,
        tol: float = 1e-12,
    ) -> None:
        """Ruppert's Delaunay refinement algorithm."""

        while True:
            encroached_segment = None

            for edge in self.edges:
                if not self.is_edge_border(edge):
                    continue

                if self.is_segment_encroached(edge, tol=tol):
                    encroached_segment = edge
                    break

            if encroached_segment is not None:
                self.split_segment(encroached_segment)
                continue

            bad_triangle = None

            for triangle in self.trg:
                if triangle.min_angle < min_angle_lb:
                    bad_triangle = triangle
                    break

            if bad_triangle is None:
                break

            p = bad_triangle.circumcenter

            segments = [edge for edge in self.edges if self.is_edge_border(edge)]

            encroaches, segment = self.point_encroaches(
                p,
                segments,
                tol=tol,
            )

            if encroaches:
                assert segment is not None
                self.split_segment(segment)
            else:
                self.split_triangle(bad_triangle)
