# 2024.1064

## Cite
TBD

## Repository structure

```text
.
├── src/                    # Python source code
│   └── pwlopt/             # Python library
│       ├── fitting/        # Implementation of PWL function fitting algorithm
│       ├── hydropower/     # Implementation of short-term hydropower scheduling models
│       └── modeling/       # Implementation of modeling tools, e.g. biclique cover algorithm, rank reduction etc.
├── tests/                  # Tests for Delaunay-refinement
├── data/
│   ├── hydro_instances/    # Hydropower benchmark instances
│   ├── simplicizations/    # Random simplicial partitions for rank reduction experiments
│   └── triangulations/     # Triangulation instances
├── results/                # Generated results
│   ├── biclique_cover/     # Results for biclique cover algorithm
│   ├── fitting/            # Results for PWL fitting
│   ├── hydropower/         # Results for hydropower scheduling on various triangulation
│   ├── rank_reduction/     # Results of rank reduction algorithm on 3 and 4 dim. simplicial partitions
│   └── triangle_coloring/  # Blocking hypergraph coloring results
├── notebooks/              # Notebooks for example usage of algorithms
├── cpp/                    # C++ implementation of the maximal Poisson-disk sampling alg.
├── pyproject.toml          # Python project configuration
├── CMakeLists.txt          # CMake file for building sampling alg.
├── AUTHORS                 # Author list
├── LICENSE                 # License file
└── README.md
```

## Requirements
The required packages are listed in `pyproject.toml`. To build and solve the MILP models (the hydropower models and the biclique cover algorithm), a FICO Xpress license is required.

## Hydropower instances

The benchmark instances used in the computational experiments were obtained from the Library of Codes and Instances maintained by the Operations Research group at the University of Bologna:

https://site.unibo.it/operations-research/en/research/library-of-codes-and-instances-1

The instances originate from:

> **Borghetti et al. (2008)** *An MILP Approach for Short-Term Hydro Scheduling and Unit Commitment With Head-Dependent Reservoir*, IEEE Transactions on Power Systems, 23(3), 1115-1124.

The instance files found in `data/hydro_instances` were interpreted from `.lp` files in the original distribution.

For licensing and redistribution terms, see the original source.