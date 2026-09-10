from itertools import combinations
from logging import getLogger

import pytest
from matplotlib.pyplot import colormaps
from numpy import array


@pytest.mark.skip
def with_logger(cls):
    cls.logger = getLogger(f"{cls.__module__}.{cls.__name__}")
    return cls


def write_trg(trg, pts, filename):
    npts, ntrg = len(pts), len(trg)
    with open(filename, "w+") as f:
        f.write(f"{npts},{ntrg}\n")
        f.writelines(f"{x},{y}\n" for x,y in pts)
        f.writelines(f"{u},{v},{w}\n" for u,v,w in trg)

def read_trg(filename):
    with open(filename, "r") as f:
        lines = f.readlines()
    npts, _ntrg = int(lines[0].split(",")[0]), int(lines[0].split(",")[1])
    
    pts = []
    for l in lines[1:npts+1]:
        x,y = float(l.split(",")[0]),float(l.split(",")[1])
        pts.append([x,y])
    pts = array(pts)

    trg = []
    for l in lines[npts+1:]:
        u,v,w = int(l.split(",")[0]),int(l.split(",")[1]),int(l.split(",")[2])
        trg.append([u,v,w])
    
    return trg, pts


def get_rgb(t, color, ncolor):
    cmap = colormaps["plasma"]
    return cmap(color[tuple(t)] / (ncolor+1))


def induced_triangles(tri):
    res = []
    for i,j,k in combinations(tri, 3):
        # print(i,j,k)
        t1 = set(i)
        t2 = set(j)
        t3 = set(k)
        if len(t1 |t2 | t3) == 4:
            res.append((t1 | t2 | t3) - (t1 & t2 & t3))
    return res

def incident_triangles(T, node):
    return list(filter(lambda t: node in t, T))