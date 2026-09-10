import json

import xpress as xp

from ...modeling.biclique_cover_IP import BicliqueCover
from ...modeling.blocking_coloring import TriangulationColoring
from ..hydro import Hydro


class HydroGIB(Hydro):
    def addPWL(self, j, t) -> None:
        AA, BB = self.B.reds, self.B.blues
        J = range(len(self.pts))
        x = {i: self.prob.addVariable(lb=0, ub=1) for i in J}
        L = {i: self.prob.addVariable(vartype=xp.binary) for i in range(len(AA))}

        if self.color.ncolor > 1:
            self.logger.debug("Set up coloring constraints for turbine %d and time %d", j, t)
            C = {
                c: self.prob.addVariable(vartype=xp.binary)
                for c in range(self.color.ncolor)
            }

            self.prob.addConstraint(
                sum(C[c] for c in range(self.color.ncolor)) == self.g[j, t]
            )

            pi = {i: set() for i in J}

            for tri, c in self.color.coloring.items():
                for i in tri:
                    pi[i].add(c)
            patterns = set()

            for i in J:
                pi[i] = tuple(sorted(pi[i]))
                patterns.add(pi[i])

            for pi_ in patterns:
                self.prob.addConstraint(
                    sum(x[i] for i in J if tuple(pi[i]) == tuple(pi_))
                    <= sum(C[c] for c in pi_)
                )

        self.prob.addConstraint(sum(x[k] for k in J) == self.g[j, t])

        self.logger.debug("Set up biclique cover constraints")
        for i in range(len(AA)):
            self.prob.addConstraint(sum(x[k] for k in AA[i]) <= L[i])
            self.prob.addConstraint(sum(x[k] for k in BB[i]) <= 1 - L[i])

        self.prob.addConstraint(
            self.u[j, t] * self.flowByPump[j] + sum(x[i] * self.pts[i, 0] for i in J)
            == self.q[j, t]
        )
        self.prob.addConstraint(
            sum(x[i] * self.pts[i, 1] for i in J)
            <= self.v[t] + self.u[j, t] * self.Vmax
        )
        self.prob.addConstraint(
            sum(x[i] * self.pts[i, 1] for i in J)
            >= self.v[t] - self.u[j, t] * self.Vmin
        )
        self.prob.addConstraint(
            self.u[j, t] * self.maxPowerConsumed[j]
            + sum(
                x[i] * self.hpf(self.pts[i, 0], self.pts[i, 1] / self.scale) for i in J
            )
            == self.p[j, t]
        )

    def buildPWL(self, biclique_file=None, color_file=None):
        self.B = BicliqueCover(self.trg, self.pts)
        if biclique_file:
            self.logger.info("Read biclique cover from file: %s", biclique_file)
            with open(biclique_file, "r") as json_file:
                data = json.load(json_file)
                reds, blues = [], []
                for dic in data:
                    reds.append(dic["A"])
                    blues.append(dic["B"])
                self.B.reds = reds
                self.B.blues = blues
        else:
            self.logger.info("Solve biclique cover with MIP-based greedy heuristic")
            self.B.solve()

        if color_file:
            self.logger.info("Read coloring from file: %s", color_file)
            self.color = TriangulationColoring(self.trg, self.pts)
            self.color.readColoring(color_file)
        else:
            self.logger.info("Color blocking hypergraph")
            self.color = TriangulationColoring(self.trg, self.pts)
            self.color.color()