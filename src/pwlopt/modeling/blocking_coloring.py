import json
from itertools import combinations, product

from matplotlib.patches import Polygon
from matplotlib.pyplot import figure, show
from networkx import Graph, greedy_color
from numpy import mean
from pysat.card import CardEnc, EncType
from pysat.formula import CNF
from pysat.solvers import Glucose3

from ..utils import get_rgb, incident_triangles, induced_triangles, with_logger


@with_logger
class TriangulationColoring:
    def __init__(self, tri, points) -> None:
        self.tri = tri
        self.points = points
        self.npts = len(self.points)
        self.conflict_graph = self._create_blocking_rank_2_graph()

    def _create_blocking_rank_2_graph(self):
        self.logger.info("Compute rank-2 subgraph of blocking hypergraph")
        G = Graph()

        for t in self.tri:
            G.add_node(tuple(t), pos=mean(self.points[t], axis=0))

        H = self._create_tri_graph()
        
        for t1, t2 in combinations(self.tri, 2):
            s1, s2 = set(t1), set(t2)
            if len(s1 & s2) >= 1:
                V1 = s1 - s2
                V2 = s2 - s1
                E = product(V1, V2)
                for e in E:
                    if H.has_edge(*e):
                        G.add_edge(tuple(t1), tuple(t2))
                        
            elif len(s1 & s2) == 0:
                for v1 in s1:
                    cnt = 0
                    for v2 in s2:
                        if H.has_edge(v1, v2):
                            cnt += 1
                    if cnt >= 2:
                        G.add_edge(tuple(t1), tuple(t2))
                for v1 in s2:
                    cnt = 0
                    for v2 in s1:
                        if H.has_edge(v1, v2):
                            cnt += 1
                    if cnt >= 2:
                        G.add_edge(tuple(t1), tuple(t2))
                
        return G
    
    def _create_tri_graph(self):
        H = Graph()

        for t in self.tri:
            for v in t:
                H.add_node(v, pos=self.points[v])
            for u, v in combinations(t, 2):
                H.add_edge(u,v)

        return H
    
    def _greedy_color_graph(self, verbose=False):
        self.logger.info("Color greedily the rank-2 subgraph of the blocking hypergraph")
        colorings = []
        strats = ["largest_first", "smallest_last", "random_sequential", "connected_sequential_bfs", "connected_sequential_dfs", "independent_set", "DSATUR"]
        interchange = [1,1,1,1,1,0,0]
        for i in range(7):
            c = greedy_color(self.conflict_graph, strategy = strats[i], interchange=interchange[i])
            colorings.append(c)
            self.logger.info("Greedy strategy <%s> found coloring with %d colors", strats[i], max(c.values())+1)
        self.coloring = min(colorings, key=lambda c: max(c.values())+1)
        self.ncolor = max(self.coloring.values())+1

    def validate_coloring(self):        
        induced = induced_triangles(self.tri)
        valid = True
        for u,v,w in self.tri + induced:
            color_u = {self.coloring[tuple(t)] for t in incident_triangles(self.tri, u) if t != (u,v,w)}
            color_v = {self.coloring[tuple(t)] for t in incident_triangles(self.tri, v) if t != (u,v,w)}
            color_w = {self.coloring[tuple(t)] for t in incident_triangles(self.tri, w) if t != (u,v,w)}
            if (u,v,w) in self.coloring:
                if len(color_u & color_v & color_w - {self.coloring[u,v,w]}) > 0:     
                    valid = False
            else:
                if len(color_u & color_v & color_w) > 0:
                    valid = False
        return valid
    
    
    def _SAT_color_graph(self, ncol):    
        def var(v, c, ncol=ncol):
            return int(ncol*v + c + 1)
        
        def var_inv(i):
            return (i-1)//ncol, (i-1)%ncol
        
        node_id = {}
        node_by_id = {}
        for i,v in enumerate(self.conflict_graph.nodes()):
            node_id[v] = i
            node_by_id[i] = v
        
        formula = CNF()
        
        for v in self.conflict_graph.nodes():
            formula.extend(CardEnc.equals(lits=[var(node_id[v], c) for c in range(ncol)], bound=1, encoding=EncType.pairwise))
        
        for u,v in self.conflict_graph.edges():
            for c in range(ncol):
                formula.append([-var(node_id[u],c), -var(node_id[v],c)])

        
        T = induced_triangles(self.tri)
        for u,v,w in self.tri + T:
            tu = filter(lambda t: t != [u,v,w], incident_triangles(self.tri, u))
            tv = filter(lambda t: t != [u,v,w], incident_triangles(self.tri, v))
            tw = filter(lambda t: t != [u,v,w], incident_triangles(self.tri, w))
            for i,j,k in filter(lambda prod: prod[0] != prod[1] != prod[2] != prod[0], product(tu, tv, tw)):
                i = tuple(i)
                j = tuple(j)
                k = tuple(k)
                if not (self.conflict_graph.has_edge(i,j) or self.conflict_graph.has_edge(i,k) or self.conflict_graph.has_edge(j,k)):
                    for c in range(ncol):
                        formula.extend(CardEnc.atmost(lits=[var(node_id[l], c) for l in [i,j,k]], bound=2, encoding=EncType.pairwise))
        
        with Glucose3(bootstrap_with=formula.clauses) as prob:
            feas = prob.solve()
            if feas:
                model = prob.get_model()

        if feas:
            color_new = {}
            for v in self.conflict_graph.nodes():
                for c in range(ncol):
                    i = var(node_id[v],c)
                    if model[i-1] > 0:
                        color_new[v] = c
            self.coloring = color_new
        else:
            self.logger.info("No coloring with %d colors", ncol)
        return feas

    def color(self, verbose=False):
        self._greedy_color_graph(verbose=verbose)
        self.logger.info("Validate coloring with %d colors on complete blocking hypergraph ", self.ncolor)
        if not self.validate_coloring():
            self.logger.warning("INVALID COLORING")
            self.logger.info("Try SAT with %d colors", self.ncolor)
            ncolor = self.ncolor
            self.greedy_success = False
            while not self._SAT_color_graph(ncolor):
                self.logger.info("Try SAT with %d colors", ncolor)
                ncolor += 1
            self.logger.info("Succesfully colored with %d colors", ncolor)            
            self.ncolor = ncolor
        else:
            self.logger.info("Coloring is valid")
            self.greedy_success = True

    def plot(self):
        fig = figure(figsize=(8,8))
        ax = fig.add_subplot(111, aspect="equal")
        for t in self.tri:
            ax.add_patch(Polygon(self.points[t], closed=True, fill=True, color=get_rgb(t, self.coloring, self.ncolor)))
        ax.triplot(self.points[:,0], self.points[:,1], self.tri, color="white", linewidth=0.9)
        show()

    def writeColoring(self, filename):
        coloring_idx = [self.coloring[tuple(t)] for t in self.tri]
        dic = {"coloring": coloring_idx,
               "ncolor": self.ncolor}
        with open(filename, "w+") as f:
            json.dump(dic, f, indent=4)

    def readColoring(self, filename):
        with open(filename) as f:
            data = json.load(f)
        self.ncolor = data["ncolor"]
        self.coloring = {}
        for i,c in enumerate(data["coloring"]):
            self.coloring[tuple(self.tri[i])] = c

    def writeNFR(self, filename):
        nnodes = self.ncolor + self.npts + 2
        source = 0
        sink = nnodes-1
        terminals = [i for i in range(self.ncolor+1, nnodes-1)]
        nodes = [i for i in range(nnodes)]
        vars = [f"L{i+1}" for i in range(self.ncolor)] + [f"x{i+1}" for i in range(self.npts)]
        edges = [{"source": source,
                  "target": i+1,
                  "variable" : i,
                  "coefficient": 1} for i in range(self.ncolor)]
        for t in self.tri:
            for p in t:
                edge = {"source": self.coloring[tuple(t)]+1,
                        "target": self.ncolor+1+p,
                        "variable": self.coloring[tuple(t)],
                        "coefficient": 1}
                edges.append(edge)

        edges += [{"source": i,
                        "target": sink,
                        "variable": i-1,
                        "coefficient": 1} for i in terminals]
        nfr = {"Source": source,
               "Sink": sink,
               "Terminals": terminals,
               "Nodes": nodes,
               "Variables": vars,
               "Edges": edges}
        with open(filename, "w+") as f:
            json.dump(nfr, f)
            
        
