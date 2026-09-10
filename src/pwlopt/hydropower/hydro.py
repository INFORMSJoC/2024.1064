import matplotlib.pyplot as plt
import numpy as np
import xpress as xp

from ..utils import read_trg, with_logger
from .hydro_data import HydroData


@with_logger
class Hydro:
    def __init__(self, data: HydroData, scale: float = 1e-5):
        self.data = data
        self.maxT = data.maxT
        self.nperiod = data.nperiod
        self.T = range(self.nperiod)
        self.J = data.J
        self.deltaT = data.deltaT
        self.C = data.C
        self.inflow = data.inflow
        self.price = data.price
        self.pumpCost = data.pump_cost
        self.V0, self.VT = data.V0, data.VT
        self.Vmin, self.Vmax = data.Vmin, data.Vmax
        self.U0, self.G0 = data.U0, data.G0
        self.flowByPump, self.Qmin, self.Qmax = data.flow_by_pump, data.Qmin, data.Qmax
        self.rampDown, self.rampUp = data.ramp_down, data.ramp_up
        self.maxPowerConsumed, self.maxPowerProduced = (
            data.max_power_consumed,
            data.max_power_produced,
        )
        self.Y, self.W = data.Y, data.W
        self.theta = data.theta
        self.maxSpillage = data.max_spillage
        self.scale = scale

    def readTrg(self, filename):
        self.logger.info("Read triangulation from file %s", filename)
        self.trg, self.pts = read_trg(filename)
        self.logger.info("Triangulation has %d triangles and %d points", len(self.trg), len(self.pts))
        self.logger.info("Scaling pointset to match hydropower plant data")
        self.pts = self.pts * np.array([
            (self.Qmax[0] - self.Qmin[0]),
            (self.Vmax - self.Vmin),
        ]) + np.array([self.Qmin[0], self.Vmin])
        self.pts *= np.array([1, self.scale])

    def hpf(self, q, v):
        L = [
            4.09863600116008,
            -1.25535942295343,
            0.160530264942775,
            -9.76201903589132e-03,
            0.000309429429972963,
            -4.92928898248035e-06,
            3.11519548768e-08,
        ]
        Llb = 384
        K = [307.395, 3.88e-05, -4.37e-12, 2.65e-19, -8.87e-27, 1.55e-34, -1.11e-42]
        R0 = 0.01
        H = range(len(L))

        def f(q, v):
            return (
                9.81
                / 1000
                * q
                * sum(
                    L[h] * q**h * (sum(K[k] * v**k for k in H) - Llb - R0 * q**2)
                    for h in H
                )
            )

        return (q >= self.Qmin[0]) * f(q, v) + self.maxPowerConsumed[0] * (
            q < self.Qmin[0]
        )

    def addVars(self, prob):
        self.logger.info("Set up variables")
        self.q = np.array(
            [
                [prob.addVariable(lb=self.flowByPump[j], ub=self.Qmax[j]) for t in self.T]
                for j in self.J
            ]
        )  # water flow in unit j in period t
        self.v = np.array(
            [
                prob.addVariable(lb=self.scale * self.Vmin, ub=self.scale * self.Vmax)
                for t in self.T
            ]
        )  # water volume in the basin in period t
        self.p = np.array(
            [
                [
                    prob.addVariable(lb=self.maxPowerConsumed[j], ub=self.maxPowerProduced[j])
                    for t in self.T
                ]
                for j in self.J
            ]
        )  # power generated or consumed by unit j in period t
        self.s = np.array(
            [prob.addVariable(lb=0.0, ub=self.maxSpillage) for t in self.T]
        )  # spillage in period t
        self.w = np.array(
            [[prob.addVariable(vartype=xp.binary) for t in self.T] for j in self.J]
        )  # Startup phase of turbine j in period t
        self.g = np.array(
            [[prob.addVariable(vartype=xp.binary) for t in self.T] for j in self.J]
        )  # Status of turbine j in period t
        self.y = np.array(
            [[prob.addVariable(vartype=xp.binary) for t in self.T] for j in self.J]
        )  # Startup phase of turbine j in period t
        self.u = np.array(
            [[prob.addVariable(vartype=xp.binary) for t in self.T] for j in self.J]
        )  # Status of pump j in period t
        # prob.addVariable(self.q, self.v, self.p, self.s, self.w, self.g, self.y, self.u)

    def addConstraints(self, prob):
        self.logger.info("Set up constraints")
        prob.addConstraint(
            self.v[0]
            - self.scale * self.V0
            - self.scale
            * 3600
            * self.deltaT
            * (self.inflow[0] - sum(self.q[j, 0] for j in self.J) - self.s[0])
            == 0
        )
        prob.addConstraint(
            self.scale * self.VT
            - self.v[self.T[-1]]
            - self.scale
            * 3600
            * self.deltaT
            * (
                self.inflow[self.T[-1]]
                - sum(self.q[j, self.T[-1]] for j in self.J)
                - self.s[self.T[-1]]
            )
            == 0
        )

        for j in self.J:
            self.prob.addConstraint(self.g[j, 0] - self.G0[j] - self.w[j, 0] <= 0)
            self.prob.addConstraint(self.u[j, 0] - self.U0[j] - self.y[j, 0] <= 0)

        for t in self.T[1:]:
            prob.addConstraint(
                self.v[t]
                - self.v[t - 1]
                - self.scale
                * 3600
                * self.deltaT
                * (self.inflow[t] - sum(self.q[j, t] for j in self.J) - self.s[t])
                == 0
            )

            for j in self.J:
                prob.addConstraint(self.g[j, t] - self.g[j, t - 1] - self.w[j, t] <= 0)
                prob.addConstraint(self.u[j, t] - self.u[j, t - 1] - self.y[j, t] <= 0)
            prob.addConstraint(
                sum(self.q[j, t] - self.q[j, t - 1] for j in self.J) + self.rampDown
                >= 0
            )
            prob.addConstraint(
                sum(self.q[j, t] - self.q[j, t - 1] for j in self.J) - self.rampUp <= 0
            )

        for t in self.T:
            prob.addConstraint(
                self.s[t]
                - sum(
                    self.W[j] * self.w[j, t] + self.Y[j] * self.y[j, t] for j in self.J
                )
                >= 0
            )
            prob.addConstraint(
                sum(self.q[j, t] for j in self.J) + self.s[t] - self.theta >= 0
            )
            for j in self.J:
                prob.addConstraint(self.g[j, t] + self.u[j, t] <= 1)
                prob.addConstraint(
                    self.q[j, t]
                    - (self.flowByPump[j] * self.u[j, t] + self.Qmin[j] * self.g[j, t])
                    >= 0
                )
                prob.addConstraint(
                    self.q[j, t]
                    - (self.flowByPump[j] * self.u[j, t] + self.Qmax[j] * self.g[j, t])
                    <= 0
                )

    def setObjective(self, prob):
        self.logger.info("Set up objective function")
        prob.setObjective(
            sum(
                sum(
                    self.deltaT * self.price[t] * self.p[j, t]
                    - self.C[j] * self.w[j, t]
                    - self.pumpCost[t] * self.y[j, t]
                    for j in self.J
                )
                for t in self.T
            ),
            sense=xp.maximize,
        )

    def addPWL(self, j, t, fun):
        pass

    def buildModel(self):
        # print(f">>> Build MILP model for {self.__class__.__name__}")
        self.logger.info("Build MILP model for %s", self.__class__.__name__)
        self.prob = xp.problem(name=f"Hydropower Scheduling with {self.__class__.__name__}")
        self.addVars(self.prob)
        self.addConstraints(self.prob)
        self.setObjective(self.prob)
        for t in self.T:
            for j in self.J:
                self.addPWL(j, t)

    def solve(self, soltimelimit=None, cutstrategy=-1, threads=-1, verbose=True):
        self.cutstrategy = cutstrategy
        self.threads = threads
        self.prob.controls.cutstrategy = cutstrategy
        self.prob.controls.threads = threads
        self.prob.controls.xslp_log = -1
        self.prob.controls.miprelstop = 0.0001
        if soltimelimit:
            self.logger.info("Solution time limit set to %d", soltimelimit)
            self.prob.controls.soltimelimit = soltimelimit
        if not verbose:
            self.prob.setControl("outputlog", 0)

        self.prob.solve()
        self.getSolution()

    def getSolution(self):
        self.logger.info("Retrieve solution")
        self.vSol = self.prob.getSolution(self.v)
        self.qSol = self.prob.getSolution(self.q)
        self.pSol = self.prob.getSolution(self.p)
        self.sSol = self.prob.getSolution(self.s)
        self.wSol = self.prob.getSolution(self.w)
        self.uSol = self.prob.getSolution(self.u)
        self.gSol = self.prob.getSolution(self.g)
        self.ySol = self.prob.getSolution(self.y)
        self.profit = [
            [
                self.deltaT * self.pSol[j, t] * self.price[t]
                - self.C[0] * self.wSol[j, t]
                - self.pumpCost[t] * self.ySol[j, t]
                for t in self.T
            ]
            for j in self.J
        ]

        self.obj = self.prob.attributes.objval
        self.nnodes = self.prob.getAttrib("nodes")
        self.simplexiter = self.prob.getAttrib("simplexiter")
        self.dur = self.prob.getAttrib("time")
        self.rows = self.prob.getAttrib("rows")
        self.cols = self.prob.getAttrib("cols")
        self.elems = self.prob.getAttrib("elems")
        self.binaries = self.prob.getAttrib("mipents")
        self.bestobj = self.prob.getAttrib("mipbestobjval")
        self.bestbound = self.prob.getAttrib("bestbound")
        self.logger.info("Solution process attributes:")
        self.logger.info(" - nodes:              %d", self.nnodes)
        self.logger.info(" - simplex iterations: %d", self.simplexiter)
        self.logger.info(" - duration:           %f", self.dur)

    def plotSolution(self, f, filename=None):
        fig, ax = plt.subplots(1, 1, figsize=(16, 8))
        ax.plot(self.qSol[0], label="$q_{j,t}$")
        ax.plot(self.price[0 : self.nperiod], label="Price")
        ax.plot(self.pSol[0], label="Power")

        ax.legend()
        if filename:
            fig.savefig(filename)
        plt.show()

    def writeResults(self, filename, month, size, inst):
        result_str = f"{self.__class__.__name__},{self.threads},{self.cutstrategy},{len(self.T)},{month},{size},{inst},{self.nnodes},{self.simplexiter},{self.dur},{self.bestobj},{self.bestbound},{self.actualObj},{self.rows},{self.cols},{self.binaries},{self.elems}\n"
        print(result_str)
        if filename:
            with open(filename, "a+") as f:
                f.write(result_str)


    def evaluateSolution(self, f):
        totalErr = 0
        for j in self.J:
            for t in self.T:
                err = np.abs(
                    self.pSol[j, t] - f(self.qSol[j, t], self.vSol[t] / self.scale)
                )
                print(
                    j,
                    t,
                    "Lin.power:",
                    self.pSol[j, t],
                    ", NL.power:",
                    f(self.qSol[j, t], self.vSol[t] / self.scale),
                    ", err:",
                    err,
                    ", u:",
                    self.uSol[j, t],
                    ", g:",
                    self.gSol[j, t],
                    ", q:",
                    self.qSol[j, t],
                    ", v:",
                    self.vSol[t],
                )
                totalErr += err
        print(
            "Total err:",
            totalErr,
            ", avg.err:",
            totalErr / 168,
            ", adjusted err.:",
            totalErr / self.uSol[0].sum(),
        )
        # prob.setObjective(sum(sum(self.deltaT * self.price[t] * self.p[j,t] - self.C[j] * self.w[j,t] - self.pumpCost[t] * self.y[j,t] for j in self.J) for t in self.T), sense=xp.maximize)
        self.actualObj = sum(
            sum(
                self.deltaT
                * self.price[t]
                * f(self.qSol[j, t], self.vSol[t] / self.scale)
                - self.C[j] * self.wSol[j, t]
                - self.pumpCost[t] * self.ySol[j, t]
                for j in self.J
            )
            for t in self.T
        )
        print(
            f"opt_pwl = {self.obj}, opt_nl = {self.actualObj}, err = {100 * (self.obj - self.actualObj) / self.obj:.2f}%"
        )

    def print_aux_solution(self):
        for j in self.J:
            for t in self.T:
                print(
                    j,
                    t,
                    self.prob.getSolution(self.deltas[j, t]),
                    self.prob.getSolution(self.ys[j, t]),
                )
