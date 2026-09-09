import itertools as it
import json
from time import time

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import xpress as xp

from ..utils import with_logger


@with_logger
class BicliqueCover:
    def __init__(self, trg, pts):
        self.trg = trg
        self.pts = pts
        self.npts = len(pts)
        self.trg_graph = self._create_trg_graph()
        self.conflict_graph = self._create_trg_complement_graph()
        self.border = [
            v
            for v in self.trg_graph.nodes()
            if self.pts[v, 0] in [0, 1] or self.pts[v, 1] in [0, 1]
        ]
        self.cnt_complement_edges = len(self.conflict_graph.edges())
        self.cut_value = 0
        self.weight = len(self.conflict_graph.edges())
        self.it_timer = []
        self.number_of_new_edges_covered = []
        self.reds = []
        self.blues = []
        self.opt_gap = []
        self.dual_edge_cut = []
        self.primal_path = []
        self.init_weights()
        self.build_biclique_IP()

    def init_weights(self):
        self.logger.info("Initialize edge weights")
        self.W = np.zeros((self.npts, self.npts))
        for p1, p2 in it.product(np.arange(self.npts), np.arange(self.npts)):
            if p1 != p2 and not self.trg_graph.has_edge(p1, p2):
                self.W[p1, p2] = 1

    def _create_trg_graph(self):
        graph = nx.Graph()
        for t in self.trg:
            u, v, w = t
            graph.add_node(u, pos=self.pts[u])
            graph.add_node(v, pos=self.pts[v])
            graph.add_node(w, pos=self.pts[w])
            graph.add_edge(u, v)
            graph.add_edge(u, w)
            graph.add_edge(v, w)
        return graph

    def plot_trg_graph(self):
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, aspect="equal")

        pos = nx.get_node_attributes(self.trg_graph, "pos")
        nx.draw(self.trg_graph, pos=pos, ax=ax, edge_color="grey", node_color="blue")
        plt.show()

    def _create_trg_complement_graph(self):
        complement = nx.Graph()
        for u, v in it.combinations(self.trg_graph.nodes(), 2):
            complement.add_node(u, pos=self.pts[u])
            if (u, v) not in self.trg_graph.edges():
                complement.add_edge(u, v, weight=1)
        return complement

    def build_biclique_IP(self):
        self.logger.info("Set up IP for Maximum-Weight Biclique Problem (MWBP)")
        self.p = xp.problem()
        self.x1 = {
            u: self.p.addVariable(vartype=xp.binary)
            for u in self.conflict_graph.nodes()
        }
        self.x2 = {
            u: self.p.addVariable(vartype=xp.binary)
            for u in self.conflict_graph.nodes()
        }
        self.y = {
            (u, v): self.p.addVariable(vartype=xp.binary)
            for u, v in self.conflict_graph.edges()
        }

        for u in self.conflict_graph.nodes():
            self.p.addConstraint(self.x1[u] + self.x2[u] <= 1)

        for u, v in it.combinations(self.conflict_graph.nodes(), 2):
            if (u, v) not in self.conflict_graph.edges() and (
                v,
                u,
            ) not in self.conflict_graph.edges():
                self.p.addConstraint(self.x1[u] + self.x2[v] <= 1)
                self.p.addConstraint(self.x1[v] + self.x2[u] <= 1)

        self.p.addConstraint(
            xp.Sum(self.x1[u] for u in self.conflict_graph.nodes()) >= 1
        )
        self.p.addConstraint(
            xp.Sum(self.x2[u] for u in self.conflict_graph.nodes()) >= 1
        )

        for u, v in self.conflict_graph.edges():
            self.p.addConstraint(self.x1[u] + self.x2[v] <= 1 + self.y[u, v])
            self.p.addConstraint(self.x1[v] + self.x2[u] <= 1 + self.y[u, v])
            self.p.addConstraint(self.x1[u] + self.x2[u] >= self.y[u, v])
            self.p.addConstraint(self.x1[v] + self.x2[v] >= self.y[u, v])
            self.p.addConstraint(self.x1[u] + self.x1[v] >= self.y[u, v])
            self.p.addConstraint(self.x2[u] + self.x2[v] >= self.y[u, v])

    def add_biclique_IP(self, limit=-1, verbose=False, timer=False):
        self.p.controls.xslp_log = -1
        if not verbose:
            self.p.setControl("outputlog", 0)

        if limit > 0:
            self.p.setControl("soltimelimit", limit)

        start = time()
        self.p.setObjective(
            xp.Sum(self.y[u, v] * self.W[u, v] for u, v in self.conflict_graph.edges()),
            sense=xp.maximize,
        )
        self.p.solve()

        red = [
            u
            for u in self.conflict_graph.nodes()
            if self.p.getSolution(self.x1[u]) > 0.99
        ]
        blue = [
            u
            for u in self.conflict_graph.nodes()
            if self.p.getSolution(self.x2[u]) > 0.99
        ]

        cut_val = self.p.attributes.objval
        if self.p.attributes.bestbound > 0:
            gap = (
                self.p.attributes.bestbound - self.p.attributes.objval
            ) / self.p.attributes.bestbound
        else:
            gap = 0

        x1 = np.zeros(self.npts)
        x2 = np.zeros(self.npts)

        x1[red] = 1
        x2[blue] = 1

        self.W[np.where((np.outer(x1, x2) + np.outer(x2, x1)) == 1)] = 0

        self.weight = np.sum(self.W) * 0.5

        if not timer:
            self.logger.info(
                "MWBP covers %d edges, remaining: %d (%.2f%%)",
                int(cut_val),
                int(self.weight),
                100 * self.weight / self.cnt_complement_edges,
            )
        else:
            dur = time() - start
            self.logger.info(
                "MWBP covers %d edges, remaining: %d (%.2f%%), time: %f s",
                int(cut_val),
                int(self.weight),
                100 * self.weight / self.cnt_complement_edges,
                dur,
            )
            self.it_timer.append(dur)
        self.number_of_new_edges_covered.append(cut_val)
        self.reds.append(red)
        self.blues.append(blue)
        self.opt_gap.append(gap)
        self.p.postSolve()

    def plot_cut(self, red, blue):
        node_col = []
        for v in self.trg_graph.nodes():
            if v in red:
                node_col.append("red")
            elif v in blue:
                node_col.append("blue")
            else:
                node_col.append("grey")
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, aspect="equal")

        posH = nx.get_node_attributes(self.trg_graph, "pos")
        nx.draw(self.trg_graph, pos=posH, ax=ax, edge_color="grey", node_color=node_col)
        plt.show()

    def solve(self, timer=True, limit=100):
        while self.weight > 0:
            self.add_biclique_IP(
                limit=limit, timer=timer
            )
        self.logger.info(
            "Biclique cover found, total number of bicliques: %d", len(self.reds)
        )

    def to_json(self):
        dic = {
            "ntrg": len(self.trg),
            "npts": len(self.pts),
            "nedges": self.cnt_complement_edges,
            "nbicliques": len(self.reds),
            "timer": self.it_timer,
            "cover_gain": self.number_of_new_edges_covered,
            "gap": self.opt_gap,
            "strategy": self.strategy,
        }
        return dic

    def write_bicliques(self, filename):
        cover = []
        for A, B in zip(self.reds, self.blues):
            biclique = {"A": A, "B": B}
            cover.append(biclique)
        with open(filename, "w+") as fp:
            json.dump(cover, fp)
