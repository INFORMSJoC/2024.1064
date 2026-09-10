import heapq
import itertools

import numpy as np
import pyintval as iv

from ..utils import with_logger


@with_logger
class Box2D:
    def __init__(self, ivalx:iv.Interval, ivaly:iv.Interval, bisect_x=True):
        self.ival = [ivalx, ivaly]
        self.xmin = ivalx.lo
        self.xmax = ivalx.hi
        self.ymin = ivaly.lo
        self.ymax = ivaly.hi
        self.vertices = np.array(
            [
                [self.xmin, self.ymin],
                [self.xmin, self.ymax],
                [self.xmax, self.ymin],
                [self.xmax, self.ymax],
            ]
        )
        self.bisect_x = bisect_x

    def bisect(self):
        mid = (self.ival[1 - self.bisect_x].hi + self.ival[1 - self.bisect_x].lo) / 2

        if self.bisect_x:
            left = Box2D(
                iv.Interval(self.ival[0].lo, mid), self.ival[1], 1 - self.bisect_x
            )
            right = Box2D(
                iv.Interval(mid, self.ival[0].hi), self.ival[1], 1 - self.bisect_x
            )
        else:
            left = Box2D(
                self.ival[0], iv.Interval(self.ival[1].lo, mid), 1 - self.bisect_x
            )
            right = Box2D(
                self.ival[0], iv.Interval(mid, self.ival[1].hi), 1 - self.bisect_x
            )

        return left, right

    def intersects_triangle(self, trg):
        # Triangle edge normals
        axes = []
        for i in range(3):
            edge = trg[(i + 1) % 3] - trg[i]
            normal = np.array([-edge[1], edge[0]])  # Perpendicular
            axes.append(normal)

        # SAT test
        for axis in axes:
            if np.allclose(axis, 0):
                continue
            axis = axis / np.linalg.norm(axis)

            proj_tri = np.dot(trg, axis)
            proj_box = np.dot(self.vertices, axis)

            if proj_tri.max() < proj_box.min() or proj_box.max() < proj_tri.min():
                return False  # Separating axis found

        return True

    def __repr__(self):
        return f"{self.ival[0]} x {self.ival[1]}"

    def max_diam(self):
        return max(
            self.ival[0].hi - self.ival[0].lo,
            self.ival[1].hi - self.ival[1].lo,
        )

    def __le__(self, other):
        return self.ival[0][0].lo < other.ival[0][0].lo or (
            self.ival[0][0].lo == other.ival[0][0].lo
            and self.ival[1][0].lo < other.ival[1][0].lo
        )

    def __lt__(self, other):
        return self.ival[0][0].lo < other.ival[0][0].lo or (
            self.ival[0][0].lo == other.ival[0][0].lo
            and self.ival[1][0].lo < other.ival[1][0].lo
        )


class IntervalBB:
    def __init__(self, f, box):
        self.f = f
        self.box = box
        self.boxes = []
        self.maxrange = []

    def estimate_range(self, gridsize=(100, 100)):
        fmin = 0.0
        fmax = 0.0

        dx = (self.box.xmax - self.box.xmin) / gridsize[0]
        dy = (self.box.ymax - self.box.ymin) / gridsize[1]

        for i in range(gridsize[0]):
            for j in range(gridsize[1]):
                ivalx = iv.Interval(
                    self.box.xmin + i * dx,
                    self.box.xmin + (i + 1) * dx,
                )
                ivaly = iv.Interval(
                    self.box.ymin + j * dy,
                    self.box.ymin + (j + 1) * dy,
                )

                frange = self.f(ivalx, ivaly)

                fmin = min(fmin, frange.lo)
                fmax = max(fmax, frange.hi)

        return fmin, fmax
    
    def maximize(self, ftol, dtol):
        frange = self.f(self.box.ival)

        working = []
        counter = itertools.count()

        max_inf = -np.inf
        max_sup = -np.inf

        self.maxrange.append((max_inf, max_sup))

        heapq.heappush(
            working,
            (-frange.lo, frange.hi, next(counter), self.box),
        )

        while working:
            neg_inf, sup, _, next_box = heapq.heappop(working)

            inf = -neg_inf

            if inf > max_inf:
                max_inf = inf

                self.maxrange.append((max_inf, max_sup))
                print(f"New max inf: [{max_inf}, {max_sup}]")

            if (
                sup - inf > ftol
                and next_box.max_diam() > dtol
                and sup > max_inf
            ):
                left, right = next_box.bisect()

                range_left = self.f(left.ival)
                range_right = self.f(right.ival)

                heapq.heappush(
                    working,
                    (
                        -range_left.lo,
                        range_left.hi,
                        next(counter),
                        left,
                    ),
                )

                heapq.heappush(
                    working,
                    (
                        -range_right.lo,
                        range_right.hi,
                        next(counter),
                        right,
                    ),
                )

            else:
                max_sup = max(max_sup, sup)
                self.boxes.append(next_box)

        self.maxrange = np.asarray(self.maxrange)

        return max_inf, max_sup