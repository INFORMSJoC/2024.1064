import xpress as xp

from ...utils import with_logger
from ..hydro import Hydro


@with_logger
class HydroCC(Hydro):
    def addPWL(self, j, t) -> None:
        x = [self.prob.addVariable() for _ in range(len(self.pts))]
        y = {tuple(t): self.prob.addVariable(vartype=xp.binary) for t in self.trg}

        self.prob.addConstraint(sum(y[tuple(t)] for t in self.trg) == 1)
        self.prob.addConstraint(sum(x[i] for i in range(len(self.pts))) == self.g[j, t])

        for i in range(len(self.pts)):
            self.prob.addConstraint(
                x[i] <= sum(y[tuple(t)] for t in self.trg if i in t)
            )

        self.prob.addConstraint(
            self.u[j, t] * self.flowByPump[j]
            + sum(x[i] * self.pts[i, 0] for i in range(len(self.pts)))
            == self.q[j, t]
        )
        self.prob.addConstraint(
            sum(x[i] * self.pts[i, 1] for i in range(len(self.pts)))
            <= self.v[t] + self.u[j, t] * self.Vmax
        )
        self.prob.addConstraint(
            sum(x[i] * self.pts[i, 1] for i in range(len(self.pts)))
            >= self.v[t] - self.u[j, t] * self.Vmin
        )
        self.prob.addConstraint(
            self.u[j, t] * self.maxPowerConsumed[j]
            + sum(
                x[i] * self.hpf(self.pts[i, 0], self.pts[i, 1] / self.scale)
                for i in range(len(self.pts))
            )
            == self.p[j, t]
        )
