from math import atan2

import networkx as nx
import xpress as xp
from numpy import arctan2, pi

from ...utils import with_logger
from ..hydro import Hydro
from ..hydro_data import HydroData


def angle_clockwise(v1, v2):
    """Computes the signed angle from v1 to v2 in a clockwise direction."""
    x1, y1 = v1
    x2, y2 = v2

    cross = x1 * y2 - y1 * x2  # Cross product: sin(θ)
    dot = x1 * x2 + y1 * y2  # Dot product: cos(θ)

    angle = arctan2(cross, dot)  # Signed angle (-π, π)
    return angle if angle >= 0 else (2 * pi + angle)


def find_eulerian_circuit(G, pos):
    if not nx.is_eulerian(G):
        print("Graph is not Eulerian!")
        return None

    D = G.copy()
    leaves = [v for v in D.nodes if D.degree(v) == 2]
    start_node = max(leaves, key=lambda node: (-pos[node][1], pos[node][0]))

    curr_path = [start_node]
    circuit = []

    while curr_path:
        curr_node = curr_path[-1]

        if D.out_degree(curr_node) > 0:
            next_node = next(iter(D.neighbors(curr_node)))
            if len(curr_path) > 1:
                last_node = curr_path[-2]
                last_edge_vector = pos[last_node] - pos[curr_node]
                next_node = max(
                    list(D.neighbors(curr_node)),
                    key=lambda node: angle_clockwise(
                        last_edge_vector, pos[node] - pos[curr_node]
                    ),
                )
            curr_path.append(next_node)
            D.remove_edge(curr_node, next_node)
        else:
            circuit.append(curr_path.pop())

    circuit.reverse()
    return circuit


def sort_triangle_clockwise(t, pts):
    cx = sum(pts[v][0] for v in t) / 3
    cy = sum(pts[v][1] for v in t) / 3

    # Sort by angle in descending order (clockwise)
    sorted_points = sorted(
        t, key=lambda v: atan2(pts[v][1] - cy, pts[v][0] - cx), reverse=False
    )
    return sorted_points


def sort(trg, pts):

    for i in range(len(trg)):
        trg[i] = sort_triangle_clockwise(trg[i], pts)

    D = nx.Graph()
    G = nx.Graph()
    F = nx.Graph()
    H = nx.DiGraph()

    pt_pos = {}

    for i, p in enumerate(pts):
        pt_pos[i] = p
        G.add_node(i)

    edge_counter = {
        (i, j): 0 for i in range(len(pts)) for j in range(len(pts)) if i < j
    }

    for i, p1 in enumerate(pts):
        for j, p2 in enumerate(pts):
            for t in trg:
                if i in t and j in t and i < j:
                    edge_counter[i, j] += 1
                    if edge_counter[i, j] == 2:
                        G.add_edge(i, j)

    # Compute the center of each triangle
    for index, triangle in enumerate(trg):
        D.add_node(index, vertices=tuple(sorted(triangle)))

    # Add edges between nodes (triangles that share an edge)
    primal_edges = {}
    for i, triangle1 in enumerate(trg):
        for j, triangle2 in enumerate(trg):
            if i < j:
                shared_nodes = set(triangle1).intersection(set(triangle2))
                if len(shared_nodes) == 2:
                    D.add_edge(i, j)
                    primal_edges[(i, j)] = tuple(shared_nodes)
                    primal_edges[(j, i)] = tuple(shared_nodes)

    edge_cover = nx.min_edge_cover(D)

    edge_cover_primal = [sorted(primal_edges[e]) for e in edge_cover]

    for u, v in edge_cover_primal:
        F.add_edge(u, v)

    ncomp = nx.number_connected_components(F)
    for u, v in nx.subgraph(G, F).edges:
        e = sorted((u, v))
        if e in F.edges:
            continue
        else:
            F.add_edge(*e)
            if ncomp > nx.number_connected_components(F):
                ncomp = nx.number_connected_components(F)
                continue
            else:
                F.remove_edge(*e)

    # print(nx.is_tree(F))
    # print(nx.number_connected_components(F))
    for u, v in F.edges:
        H.add_edge(u, v)
        H.add_edge(v, u)

    c = find_eulerian_circuit(H, pts)
    c = c[:-1]

    dir_edge_to_triangle = {}

    for t in trg:
        dir_edge_to_triangle[t[0], t[1]] = t
        dir_edge_to_triangle[t[1], t[2]] = t
        dir_edge_to_triangle[t[2], t[0]] = t

    final_trg = []

    i = 0
    processed = {tuple(sorted(t)): False for t in trg}

    while i < len(c):
        s, u, v, w = c[(i - 1) % len(c)], c[i], c[(i + 1) % len(c)], c[(i + 2) % len(c)]
        t0 = dir_edge_to_triangle[s, u]
        t1 = dir_edge_to_triangle[u, v]
        t2 = dir_edge_to_triangle[v, w]
        if t0 == t1:
            if processed[tuple(sorted([s, u, v]))]:
                continue
            final_trg.append((s, u, v))
            processed[tuple(sorted([s, u, v]))] = True
            # print([s,u,v])
            i += 3
        elif t1 == t2:
            if processed[tuple(sorted([u, v, w]))]:
                continue
            final_trg.append((u, v, w))
            processed[tuple(sorted([u, v, w]))] = True
            # print([u,v,w])
            i += 2
        else:
            t1_third_node = next(x for x in t1 if x != u and x != v)
            if processed[tuple(sorted([u, t1_third_node, v]))]:
                i += 1
                continue
            final_trg.append((u, t1_third_node, v))
            processed[tuple(sorted([u, t1_third_node, v]))] = True
            # print([u, t1_third_node, v])
            i += 1
    return final_trg

@with_logger
class HydroInc(Hydro):
    def __init__(self, data: HydroData):
        super().__init__(data)
        self.deltas = {(j, t): {} for j in self.J for t in self.T}
        self.ys = {(j, t): {} for j in self.J for t in self.T}

    def buildModel(self):
        self.logger.info("Compute ordering of triangles and nodes")
        self.ordered_trg = sort(self.trg, self.pts)
        super().buildModel()

    def addPWL(self, j, t) -> None:
        self.logger.debug("Add piecewise linear function representation for turbine %d at period %d", j, t)

        T0 = self.ordered_trg[0]
        v0 = self.pts[T0[0]]

        delta = {T: {v: self.prob.addVariable(lb=0.0, ub=1.0) for v in T[1:]} for T in self.ordered_trg}
        y = {T: self.prob.addVariable(vartype=xp.binary) for T in self.ordered_trg[:-1]}

        self.deltas[j, t] = delta
        self.ys[j, t] = y

        for T in self.ordered_trg:
            self.prob.addConstraint(sum(delta[T][v] for v in T[1:]) <= self.g[j, t])

        for i in range(len(self.ordered_trg) - 1):
            T_curr = self.ordered_trg[i]
            T_next = self.ordered_trg[i + 1]
            self.prob.addConstraint(
                sum(delta[T_next][v] for v in T_next[1:]) <= y[T_curr]
            )
            self.prob.addConstraint(y[T_curr] <= delta[T_curr][T_curr[-1]])

        self.prob.addConstraint(
            self.u[j, t] * self.flowByPump[j]
            + self.g[j, t] * v0[0]
            + sum(
                sum(delta[T][v] * (self.pts[v][0] - self.pts[T[0]][0]) for v in T[1:])
                for T in self.ordered_trg
            )
            == self.q[j, t]
        )
        self.prob.addConstraint(
            self.g[j, t] * v0[1]
            + sum(
                sum(delta[T][v] * (self.pts[v][1] - self.pts[T[0]][1]) for v in T[1:])
                for T in self.ordered_trg
            )
            <= self.v[t] + self.u[j, t] * self.Vmax
        )
        self.prob.addConstraint(
            self.g[j, t] * v0[1]
            + sum(
                sum(delta[T][v] * (self.pts[v][1] - self.pts[T[0]][1]) for v in T[1:])
                for T in self.ordered_trg
            )
            >= self.v[t] - self.u[j, t] * self.Vmin
        )
        self.prob.addConstraint(
            self.u[j, t] * self.maxPowerConsumed[j]
            + self.g[j, t] * self.hpf(v0[0], v0[1] / self.scale)
            + sum(
                sum(
                    delta[T][v]
                    * (
                        self.hpf(self.pts[v, 0], self.pts[v, 1] / self.scale)
                        - self.hpf(self.pts[T[0], 0], self.pts[T[0], 1] / self.scale)
                    )
                    for v in T[1:]
                )
                for T in self.ordered_trg
            )
            == self.p[j, t]
        )
