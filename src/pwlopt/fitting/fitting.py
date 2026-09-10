from collections.abc import Callable

from matplotlib.pyplot import subplots
from numpy import abs, argmax, inf, linspace, meshgrid, radians, sin, stack

from ..utils import with_logger
from .triangulation import Triangulation


@with_logger
class PWLApproximation:
    def __init__(
        self,
        initial_triangulation: Triangulation,
        fun: Callable[[float, float], float],
        Lip: float,
        alpha_min_deg: float = 20,
    ) -> None:
        self.triangulation = initial_triangulation
        self.fun = fun
        self.Lip = Lip
        self.alpha_min_deg = alpha_min_deg
        self.global_max_err = inf
        self.err_hist = []
        self.logger.info(
            "Initialize PWL approximator for function with Lipschitz const. %.3f, minimal angle %.3f", self.Lip, self.alpha_min_deg
        )

    def approximate(self, eps: float, theta=0.5) -> None:
        self.logger.info(
            "Run PWL approximation alg. with required error eps = %.5f, theta = %.5f", eps, theta
        )
        new_pt, on_edge = None, None
        iter_cnt = 0
        while self.global_max_err > eps * theta:
            # Error improvement step
            if new_pt is not None:
                self.triangulation.insert_point(new_pt, on_edge=on_edge)
                self.err_hist.append(self.global_max_err)

            # Lipschitz-correction step
            self.logger.info("Perform Lipschitz-correction step")
            self.triangulation.delaunay_refine(radians(self.alpha_min_deg))
            self.logger.info("Upper bound for Lipschitz const. of approximator: %.5f", self.Lip / sin(self.triangulation.min_angle))

            # Sampling step
            iter_max_err = 0
            new_pt = None
            for t in self.triangulation.trg:
                radius = eps * theta / (self.Lip + t.get_lip(self.fun))
                sample_pts = t.sample(radius)

                fvals = self.fun(sample_pts)
                flint = t.lint(sample_pts, self.fun)
                self.logger.debug("%s, %s", fvals.shape, flint.shape)
                err = abs(fvals - flint)

                idx = argmax(err)
                max_err_pt = sample_pts[idx]
                max_err = err[idx]

                if max_err > iter_max_err:
                    self.logger.info("Iteration #%d: new max error found: %.5f", iter_cnt, iter_max_err)
                    iter_max_err = max_err
                    new_pt, on_edge = max_err_pt, None

                
                edge_sample = t.edge_sample(radius)
                for e, sample_pts in edge_sample.items():
                    fvals = self.fun(sample_pts)
                    flint = t.lint(sample_pts, self.fun)
                    self.logger.debug("%s, %s", fvals.shape, flint.shape)
                    err = abs(fvals - flint)

                    idx = argmax(err)
                    max_err_pt = sample_pts[idx]
                    max_err = err[idx]

                    if max_err > iter_max_err:
                        self.logger.info("Iteration #%d: new max error found: %.5f", iter_cnt, iter_max_err)
                        iter_max_err = max_err
                        new_pt, on_edge = max_err_pt, e



            self.global_max_err = iter_max_err
            iter_cnt += 1

    def plot_function(self, fig=None, ax=None):
        if fig is None or ax is None:
            fig, ax = subplots(figsize=(15, 10))

        x = linspace(0, 1, 200)
        y = linspace(0, 1, 200)

        X, Y = meshgrid(x, y)
        P = stack((X, Y), axis=-1)

        Z = self.fun(P)

        ax.contourf(X, Y, Z, levels=20)