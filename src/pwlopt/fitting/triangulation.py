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

    
    @classmethod
    def from_points(cls, points: ndarray) -> "Triangulation":
        points = asarray(points, dtype=float)

        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError("points must have shape (n, 2)")

        if len(points) < 3:
            raise ValueError("At least 3 points are required")

        delaunay = Delaunay(points)

        triangles = [
            Triangle(list(indices), points)
            for indices in delaunay.simplices
        ]

        return cls(triangles, points)


    def insert_point(self, p: ndarray) -> None:
        point_idx = len(self.pts)

        # Find cavity
        bad = [
            triangle
            for triangle in self.trg
            if triangle.in_circumcircle(p)
        ]

        # Find cavity boundary
        edge_count: dict[tuple[int, int], int] = {}

        for triangle in bad:
            for edge in triangle.edges:
                edge = tuple(sorted(edge))
                edge_count[edge] = edge_count.get(edge, 0) + 1

        boundary = [
            edge for edge, count in edge_count.items()
            if count == 1
        ]

        # Remove bad triangles
        bad_set = set(bad)
        for t in bad_set:
            self.logger.info("Remove triangle: %s", t)
        self.trg = [
            triangle
            for triangle in self.trg
            if triangle not in bad_set
        ]

        # Add point
        self.logger.info("Insert new point %s with index %d", p, len(self.pts))
        self.pts = vstack((self.pts, p))

        # Retriangulate cavity
        for i, j in boundary:
            new_trg = Triangle((i, j, point_idx), self.pts)
            self.logger.info("Add new triangle: %s", new_trg)
            self.trg.append(new_trg)

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
                    t for t in self.trg
                    if t is not triangle and edge[0] in t.idxs and edge[1] in t.idxs
                ]

                if len(adjacent) != 1:
                    continue

                other = adjacent[0]

                # Vertex opposite the shared edge
                opposite = next(
                    idx for idx in other.idxs
                    if idx not in edge
                )

                if triangle.in_circumcircle(
                    self.pts[opposite],
                    tol=tol,
                ):
                    return False

        return True
