import xpress as xp

from ..hydro import Hydro


class HydroDCC(Hydro):
    def addPWL(self, j, t) -> None:
        x = {tuple(t): {j: self.prob.addVariable() for j in t} for t in self.trg}
        y = {tuple(t): self.prob.addVariable(vartype=xp.binary) for t in self.trg}

        for tri in self.trg:
            self.prob.addConstraint(
                sum(x[tuple(tri)][vertex] for vertex in tri) == y[tuple(tri)]
            )

        self.prob.addConstraint(sum(y[tuple(t)] for t in self.trg) == self.g[j, t])

        self.prob.addConstraint(
            self.u[j, t] * self.flowByPump[j]
            + sum(
                sum(x[tuple(t)][vertex] * self.pts[vertex, 0] for vertex in t)
                for t in self.trg
            )
            == self.q[j, t]
        )
        self.prob.addConstraint(
            sum(
                sum(x[tuple(t)][vertex] * self.pts[vertex, 1] for vertex in t)
                for t in self.trg
            )
            <= self.v[t] + self.u[j, t] * self.Vmax
        )
        self.prob.addConstraint(
            sum(
                sum(x[tuple(t)][vertex] * self.pts[vertex, 1] for vertex in t)
                for t in self.trg
            )
            >= self.v[t] - self.u[j, t] * self.Vmin
        )
        self.prob.addConstraint(
            self.u[j, t] * self.maxPowerConsumed[j]
            + sum(
                sum(
                    x[tuple(t)][vertex]
                    * self.hpf(self.pts[vertex, 0], self.pts[vertex, 1] / self.scale)
                    for vertex in t
                )
                for t in self.trg
            )
            == self.p[j, t]
        )
