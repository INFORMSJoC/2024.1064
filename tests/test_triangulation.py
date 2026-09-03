from itertools import product

import pytest
from numpy import array, radians
from numpy.random import default_rng

from pwlopt.fitting.triangulation import Triangulation


@pytest.mark.parametrize("seed", range(20))
def test_incremental_insertion_preserves_delaunay(seed):
    rng = default_rng(seed)
    base_pts = array([[0,0],[0,1],[1,0],[1,1]])
    points = rng.uniform(0.0, 1.0, size=(100, 2))

    triangulation = Triangulation.from_points(base_pts)

    for i, point in enumerate(points):
        triangulation.insert_point(point)

        assert triangulation.is_delaunay(), (
            f"Delaunay property violated with seed={seed}, "
            f"after inserting point {i}: {point}"
        )


@pytest.mark.parametrize("min_angle_lb, seed", product([5,10,15,20], range(4)))
def test_delaunay_refinement(min_angle_lb, seed):
    rng = default_rng(seed)
    base_pts = array([[0,0],[0,1],[1,0],[1,1]])
    
    points = rng.uniform(0.0, 1.0, size=(2, 2))

    triangulation = Triangulation.from_points(base_pts)
    for i, point in enumerate(points):
            triangulation.insert_point(point)

    triangulation.delaunay_refine(min_angle_lb=radians(min_angle_lb))
    assert triangulation.is_delaunay()
    assert triangulation.min_angle >= radians(min_angle_lb)
