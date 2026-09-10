from math import ceil, log2

import xpress as xp

from ..hydro import Hydro


class HydroDLog(Hydro):
    def addPWL(self, j, t) -> None:
        def to_binary(n, digits=-1):
            l = [int(d) for d in str(bin(n))[2:]]
            l = [0] * (digits - len(l)) + l
            return l

        logn = ceil(log2(len(self.trg)))
        encode = [to_binary(i, logn) for i in range(len(self.trg))]
        P0 = [
            [i for i in range(len(self.trg)) if encode[i][l] == 0] for l in range(logn)
        ]
        P1 = [
            [i for i in range(len(self.trg)) if encode[i][l] == 1] for l in range(logn)
        ]

        x = {tuple(t): {j: self.prob.addVariable() for j in t} for t in self.trg}
        y = [self.prob.addVariable(vartype=xp.binary) for _ in range(logn)]

        for l in range(logn):
            self.prob.addConstraint(
                sum(sum(x[tuple(self.trg[i])][j] for j in self.trg[i]) for i in P0[l])
                <= y[l]
            )
            self.prob.addConstraint(
                sum(sum(x[tuple(self.trg[i])][j] for j in self.trg[i]) for i in P1[l])
                <= 1 - y[l]
            )

        self.prob.addConstraint(
            sum(
                sum(x[tuple(self.trg[i])][j] for j in self.trg[i])
                for i in range(len(self.trg))
            )
            == self.g[j, t]
        )
        self.prob.addConstraint(
            self.u[j, t] * self.flowByPump[j]
            + sum(
                sum(x[tuple(self.trg[i])][j] * self.pts[j][0] for j in self.trg[i])
                for i in range(len(self.trg))
            )
            == self.q[j, t]
        )
        self.prob.addConstraint(
            sum(
                sum(x[tuple(self.trg[i])][j] * self.pts[j][1] for j in self.trg[i])
                for i in range(len(self.trg))
            )
            <= self.v[t] + self.u[j, t] * self.Vmax
        )
        self.prob.addConstraint(
            sum(
                sum(x[tuple(self.trg[i])][j] * self.pts[j][1] for j in self.trg[i])
                for i in range(len(self.trg))
            )
            >= self.v[t] - self.u[j, t] * self.Vmin
        )
        self.prob.addConstraint(
            self.u[j, t] * self.maxPowerConsumed[j]
            + sum(
                sum(
                    x[tuple(self.trg[i])][j]
                    * self.hpf(self.pts[j, 0], self.pts[j, 1] / self.scale)
                    for j in self.trg[i]
                )
                for i in range(len(self.trg))
            )
            == self.p[j, t]
        )
