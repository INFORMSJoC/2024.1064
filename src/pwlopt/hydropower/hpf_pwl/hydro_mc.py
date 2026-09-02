import xpress as xp
from numpy import array, concatenate, dot, ones
from numpy.linalg import lstsq, solve
from scipy.spatial import ConvexHull

from ..hydro import Hydro


class HydroMC(Hydro):
    def get_ieqs(self, t):
        vertices = self.pts[t]
        if len(t) == 3:
            Q = ConvexHull(vertices)
            A = Q.equations[:, :-1]
            b = -Q.equations[:, -1]

        elif len(t) == 2:
            P1 = self.pts[t[0]]
            P2 = self.pts[t[1]]
            x1, y1 = P1
            x2, y2 = P2

            A = array([[1, 0], [-1, 0], [-1, 0], [1, 0], [0, -1], [0, 1]])
            b = array([x1, -x1, -min(x1, x2), max(x1, x2), -min(y1, y2), max(y1, y2)])

        return A, b

    def get_LINT(self, t):
        vertices = self.pts[t]
        b = [self.hpf(p[0], p[1] / self.scale) for p in vertices]
        if len(t) == 3:
            A = concatenate((vertices, ones((3, 1))), axis=1)
            sol = solve(A, b)
            m = sol[:-1]
            c = sol[-1]
        elif len(t) == 2:
            A = concatenate((vertices, ones((2, 1))), axis=1)
            sol = lstsq(A, b, rcond=None)[0]
            m = sol[:-1]
            c = sol[-1]

        return m, c

    def addPWL(self, j, t) -> None:
        x = {
            tuple(t): array([self.prob.addVariable(lb=self.flowByPump[0], ub=self.Qmax[0]), self.prob.addVariable()])
            for t in self.trg
        }
        y = {tuple(t): self.prob.addVariable(vartype=xp.binary) for t in self.trg}

        M, C = {}, {}
        for tri in self.trg:
            m, c = self.get_LINT(tri)
            M[tuple(tri)] = m
            C[tuple(tri)] = c

        self.prob.addConstraint(sum(y[tuple(t)] for t in self.trg) == self.g[j, t])

        for tri in self.trg:
            A, b = self.get_ieqs(tri)
            LHS = dot(A, x[tuple(tri)])
            RHS = y[tuple(tri)] * b
            for k in range(len(LHS)):
                self.prob.addConstraint(LHS[k] <= RHS[k])

        self.prob.addConstraint(
            self.u[j, t] * self.flowByPump[j] + sum(x[tuple(t)][0] for t in self.trg)
            == self.q[j, t]
        )
        self.prob.addConstraint(
            sum(x[tuple(t)][1] for t in self.trg)
            <= self.v[t] + self.u[j, t] * self.Vmax
        )
        self.prob.addConstraint(
            sum(x[tuple(t)][1] for t in self.trg)
            >= self.v[t] - self.u[j, t] * self.Vmin
        )
        self.prob.addConstraint(
            self.u[j, t] * self.maxPowerConsumed[j]
            + sum(
                dot(x[tuple(t)], M[tuple(t)]) + y[tuple(t)] * C[tuple(t)]
                for t in self.trg
            )
            == self.p[j, t]
        )
