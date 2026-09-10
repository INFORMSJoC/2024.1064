from itertools import product
from math import ceil, log2

import xpress as xp
from numpy import array

from ...utils import with_logger
from ..hydro import Hydro


@with_logger
class GridTriangulation:
    def __init__(self, diag_ind, dx, dy, xmin, xmax, ymin, ymax):
        self.dx = dx
        self.dy = dy
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
        self.diag_ind = diag_ind
        # self.pts = product(linspace(xmin, xmax, dx+1), linspace(ymin, ymax, dy+1))
        self.pts = []
        self.pidx = {}
        idx = 0
        for i in range(self.dx + 1):
            for j in range(self.dy + 1):
                px = xmin + i * (xmax - xmin) / dx
                py = ymin + j * (ymax - ymin) / dy
                self.pts.append([px, py])
                self.pidx[(i, j)] = idx
                idx += 1
        self.pts = array(self.pts)

        self.trg = []
        for i in range(self.dx):
            for j in range(self.dy):
                if diag_ind[i, j] == 1:
                    self.trg.append(
                        [self.pidx[i, j], self.pidx[i, j + 1], self.pidx[i + 1, j + 1]]
                    )
                    self.trg.append(
                        [self.pidx[i, j], self.pidx[i + 1, j], self.pidx[i + 1, j + 1]]
                    )
                else:
                    self.trg.append(
                        [self.pidx[i, j], self.pidx[i, j + 1], self.pidx[i + 1, j]]
                    )
                    self.trg.append(
                        [
                            self.pidx[i + 1, j + 1],
                            self.pidx[i + 1, j],
                            self.pidx[i, j + 1],
                        ]
                    )


@with_logger
class HydroLogGrid(Hydro):
    def setGridTrg(self, grid_trg: GridTriangulation):
        self.logger.info("Set up grid triangulation")
        self.grid_trg = grid_trg
        self.dx = grid_trg.dx
        self.dy = grid_trg.dy
        self.trg = grid_trg.trg
        self.pts = grid_trg.pts
        self.pts = self.pts * array([
                    (self.Qmax[0] - self.Qmin[0]),
                    (self.Vmax - self.Vmin),
                ]) + array([self.Qmin[0], self.Vmin])
        self.pts *= array([1, self.scale])
        self.pidx = grid_trg.pidx
        self.diag_ind = grid_trg.diag_ind

    def readGridTrg(self, filename):
        with open(filename, "r") as f:
            lines = f.readlines()
        npts, _ntrg, dx, dy = (
            int(lines[0].split(",")[0]),
            int(lines[0].split(",")[1]),
            int(lines[0].split(",")[2]),
            int(lines[0].split(",")[3]),
        )

        pts = []
        for l in lines[1 : npts + 1]:
            x, y = float(l.split(",")[0]), float(l.split(",")[1])
            pts.append([x, y])
        pts = array(pts)

        trg = []
        for l in lines[npts + 1 :]:
            u, v, w = int(l.split(",")[0]), int(l.split(",")[1]), int(l.split(",")[2])
            self.trg.append([u, v, w])

        self.trg = trg
        self.pts = pts
        self.dx = dx
        self.dy = dy

    # Construct Grey code of n bits
    def grey_code(self, n):
        c = [[0],[1]]
        for j in range(n-1):
            c = [[0] + h for h in c] + [[1] + h for h in c[::-1]]

        c = [c[0]] + c + [c[-1]]
        return c

    # Construct the biclique cover of the line graph on d+1 nodes
    def SOS2_biclique_cover(self, d):
        N = d+1
        n = ceil(log2(d))
        h = self.grey_code(n)
        A = [[i-1 for i in range(1, N+1) if h[i][j] == h[i-1][j] == 0] for j in range(n)]
        B = [[i-1 for i in range(1, N+1) if h[i][j] == h[i-1][j] == 1] for j in range(n)]
        return [A, B]

    # Construct the biclique cover of the columns and rows of the grid, i.e. the cover for the conflicts between nodes that are too far apart in x or y direction
    def aggregated_SOS2_biclique_cover(self):
        A1, B1 = self.SOS2_biclique_cover(self.dx)
        A2, B2 = self.SOS2_biclique_cover(self.dy)
        AA = [list(product(range(self.dx+1), a)) for a in A1] + [list(product(a, range(self.dy+1))) for a in A2]
        BB = [list(product(range(self.dx+1), b)) for b in B1] + [list(product(b, range(self.dy+1))) for b in B2]
        return AA, BB


    # Cover the diagonals
    def diagonal_bicliques(self):
        AA = []
        BB = []
        for t in range(3):
            A = []
            B = []
            f = True
            for k in range(t, self.dx, 3):
                for i in range(k, self.dx+1):
                    j = i - k
                    u, v = (i,j), (i+1,j+1)
                    if 0 <= j < self.dy-1 and 0 <= i < self.dx-1 and self.diag_ind[u] == 0:
                            if f:
                                A.append(u)
                                B.append(v)
                            else:
                                A.append(v)
                                B.append(u)
                            f = not f
            f = True
            for k in range(3-t, self.dy, 3):
                for j in range(k, self.dy+1):
                    i = j - k
                    u, v = (i,j), (i+1,j+1)
                    if 0 <= j < self.dy-1 and 0 <= i < self.dx-1 and self.diag_ind[u] == 0:
                            if f:
                                A.append(u)
                                B.append(v)
                            else:
                                A.append(v)
                                B.append(u)
                            f = not f
            AA.append(list(set(A)))
            BB.append(list(set(B)))
        return AA, BB

    # Cover the antidiagonals
    def antidiagonal_bicliques(self):
        AA = []
        BB = []
        for t in range(3):
            A = []
            B = []
            f = True
            for k in range(t, self.dx, 3):
                for j in range(self.dy+1):
                    i = self.dx-1-j-k
                    u, v = (i+1,j), (i, j+1)
                    if 0 <= i < self.dx-1 and 0 <= j < self.dy-1 and self.diag_ind[i,j] == 1:
                            if f:
                                A.append(u)
                                B.append(v)
                            else:
                                A.append(v)
                                B.append(u)
                            f = not f
            f = True
            for k in range(3-t, self.dy, 3):
                for j in range(k, self.dy+1):
                    i = self.dx-1-j+k
                    u, v = (i+1,j), (i,j+1)
                    if 0 <= i < self.dx-1 and 0 <= j < self.dy-1 and self.diag_ind[i,j] == 1:
                            if f:
                                A.append(u)
                                B.append(v)
                            else:
                                A.append(v)
                                B.append(u)
                            f = not f
            AA.append(list(set(A)))
            BB.append(list(set(B)))
        return AA, BB

    def addPWL(self, j, t) -> None:
        A_far, B_far = self.aggregated_SOS2_biclique_cover()
        A_diag, B_diag = self.diagonal_bicliques()
        A_adiag, B_adiag = self.antidiagonal_bicliques()

        A = A_far + A_diag + A_adiag
        B = B_far + B_diag + B_adiag

        x = array(
            [
                [self.prob.addVariable(lb=0, ub=1) for i in range(self.dx + 1)]
                for j in range(self.dy + 1)
            ]
        )
        z = array([self.prob.addVariable(vartype=xp.binary) for i in range(len(A))])

        # self.prob.addVariable(x, z)

        self.prob.addConstraint(x.sum() == self.g[j, t])

        for i in range(len(A)):
            self.prob.addConstraint(sum(x[v] for v in A[i]) <= z[i])
            self.prob.addConstraint(sum(x[v] for v in B[i]) <= 1 - z[i])

        self.prob.addConstraint(
            self.u[j, t] * self.flowByPump[j]
            + sum(
                x[i, j] * self.pts[self.pidx[i, j], 0]
                for i, j in product(range(self.dx + 1), range(self.dy + 1))
            )
            == self.q[j, t]
        )

        self.prob.addConstraint(
            sum(
                x[i, j] * self.pts[self.pidx[i, j], 1]
                for i, j in product(range(self.dx + 1), range(self.dy + 1))
            )
            <= self.v[t] + self.u[j, t] * self.Vmax
        )
        self.prob.addConstraint(
            sum(
                x[i, j] * self.pts[self.pidx[i, j], 1]
                for i, j in product(range(self.dx + 1), range(self.dy + 1))
            )
            >= self.v[t] - self.u[j, t] * self.Vmin
        )

        self.prob.addConstraint(
            self.u[j, t] * self.maxPowerConsumed[j]
            + sum(
                x[i, j]
                * self.hpf(
                    self.pts[self.pidx[i, j], 0],
                    self.pts[self.pidx[i, j], 1] / self.scale,
                )
                for i, j in product(range(self.dx + 1), range(self.dy + 1))
            )
            == self.p[j, t]
        )
