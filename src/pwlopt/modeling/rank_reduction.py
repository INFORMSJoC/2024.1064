from itertools import combinations

from numpy import inf, ndarray, vstack

from ..utils import with_logger


class SimplicialPartition:
    def __init__(self, simplices: list[set], pts: ndarray):
        self.simplices = simplices
        self.pts = pts
        self.vertices = set()
        for s in self.simplices:
            for v in s:
                self.vertices.add(v)
        self.dim = len(max(self.simplices, key=len)) - 1

@with_logger
class ConflictHypergraph:
    def __init__(self, simplicial_partition: SimplicialPartition) -> None:
        self.simplicial_partition = simplicial_partition
        self._conflicts = None
        self._neighbors = None

    @property
    def dim(self):
        return self.simplicial_partition.dim

    @property
    def rank(self):
        return len(max(self.conflicts, key=len))

    @property
    def neighbors(self):
        """
            Compute neighboring vertices for every vertex if not stored
        """
        if self._neighbors is None:
            self._neighbors = {v: set() for v in self.vertices}

            for simplex in self.simplicial_partition.simplices:
                for u in simplex:
                    self._neighbors[u].update(v for v in simplex if v != u)
        return self._neighbors

    @property
    def simplices(self):
        return self.simplicial_partition.simplices

    @simplices.setter
    def simplices(self, other):
        self.simplicial_partition.simplices = other

    @property
    def pts(self):
        return self.simplicial_partition.pts

    @pts.setter
    def pts(self, other):
        self.simplicial_partition.pts = other

    @property
    def vertices(self):
        return self.simplicial_partition.vertices

    def is_feasible(self, subset: set):
        """
            Check if there exists a simplex that contains the subset
        """
        for s in self.simplices:
            if subset <= s:
                return True
        return False

    def is_minimal(self, conflict: set):
        """
            Check if there exists a subset of the conflict set that is also a conflict set.
        """
        for cs in self.conflicts:
            if cs <= conflict:
                return False
        return True

    @property
    def conflicts(self):
        """
            Check if a subset of the neighbors of a vertex induce a minimal conflict set,
            i.e., it is not subset of any of the simplices (= feasible), and none of its
            subsets are conflict sets.
        """
        if self._conflicts is None:
            self._conflicts = []
            for k in range(2, self.dim + 2):
                for v in self.vertices:
                    for subset in combinations(self.neighbors[v], k):
                        candidate = set(subset)
                        if not self.is_feasible(candidate) and self.is_minimal(candidate):
                            self._conflicts.append(candidate)
        return self._conflicts

    def split_edge(self, u, v):
        """
            Split an edge at the midpoint
        """
        new_vertex = len(self.vertices)
        new_pt = (self.pts[u] + self.pts[v]) * .5

        new_simplices = []

        for simplex in self.simplices:
            if u not in simplex or v not in simplex:
                new_simplices.append(simplex)
                continue

            remainder = simplex - {u, v}

            new_simplices.append(remainder | {u, new_vertex})
            new_simplices.append(remainder | {v, new_vertex})

        self.simplicial_partition.vertices.add(new_vertex)
        self.simplicial_partition.pts = vstack((self.pts, new_pt))
        self.simplices = new_simplices
        self.logger.debug("Add new vertex %d", new_vertex)
        self._neighbors = None
        self._conflicts = None

    def count_created_conflicts_for_splitting_with_cardinality(self, u, v, card):
        """
            Count the newly created conflicts by splitting edge (u,v) of cardinality <card>,
            using Propositions 2 and 3
        """
        new_conflicts = 0
        for subset in combinations(self.neighbors[u] & self.neighbors[v], card - 1):
            candidate = set(subset)
            if (candidate | {u}) in self.conflicts or (
                candidate | {v}
            ) in self.conflicts:
                new_conflicts += 1
            if candidate | {u, v} in self.conflicts:
                new_conflicts += 1
        return new_conflicts

    def count_resolved_conflicts_for_splitting_with_cardinality(self, u, v, card):
        """
            Count the conflicts of cardinality <card>, eliminated by splitting edge (u,v)
        """
        resolved_conflicts = 0
        for conflict in self.conflicts:
            if u in conflict and v in conflict and len(conflict) == card:
                resolved_conflicts += 1
        return resolved_conflicts

    def split_delta(self, u, v, card):
        """
            Compute the diff of the number of resolved and created conflicts
        """
        return self.count_resolved_conflicts_for_splitting_with_cardinality(
            u, v, card
        ) - self.count_created_conflicts_for_splitting_with_cardinality(u, v, card)


    def reduce_rank(self):
        """
            Main loop of  the rank reduction algorithm (Algorithm 2)
        """
        iter_cnt = 0
        while self.rank > 2:
            self.logger.info("Iteration %d: conflict hypergraph rank is %d", iter_cnt, self.rank)
            best_pair = None
            best_delta = -inf
            for u,v in combinations(self.vertices, 2):
                if (d := self.split_delta(u, v, self.rank)) > best_delta:
                    best_delta = d
                    best_pair = (u,v)
            self.logger.info("Split best edge %s, reduce rank-%d conflicts by %d", best_pair, self.rank, best_delta)
            u, v = best_pair
            self.split_edge(u, v)
            iter_cnt += 1
        self.logger.info("Conflict hypergraph rank is reduced to 2")
