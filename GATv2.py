import numpy as np
from torch import Tensor, tensor, sparse_coo_tensor, zeros, zeros_like, ones, randn, rand, diag, cat, cuda, _assert, sigmoid,tanh, Size, transpose
from torch.nn import Module, Sequential, Linear, Sigmoid, LeakyReLU
# from torch.linalg import matmul
from torch.nn.functional import linear, one_hot, softmax


class GATv2(Module):
    """graph attention layer v2, from Brody et al, ICLR 2022"""
    def __init__(self, nin:int, nout:int, nslope):
        super(GATv2, self).__init__()
        self.W = Linear(nin, nout, bias=True)
        self.relu = LeakyReLU(negative_slope=nslope)
        self.a = Linear(nout, 1, bias=False)
    def forward(o, coo):
        """:param coo: an iterable of (row, col, Tensor, Tensor) with the edge node indices and node embeddings"""
        nEdges = len(coo)
        iis = np.zeros(nEdges)
        jjs = np.zeros(nEdges)
        vvs = np.zeros(nEdges)
        for ilist, (i, j, hi, hj) in enumerate(coo):
            h = cat(hi, hj, dim=0) # concatenate node features
            hw = o.W(h) # linear
            score = o.a(o.relu(hw)) # linear . relu
            iis[ilist] = i
            jjs[ilist] = j
            vvs[ilist] = score
        scoreMtx = sparse_coo_tensor([iis, jjs], vvs)
        attn = softmax(scoreMtx, dim=1)  # attention weights alpha_i,j

class Graph:
    def __init__(self):
        self.vertices = {}
        self.edges = {}
    def __repr__(self):
        return f'Nodes : {str(self.vertices)}, edges : {str(self.edges)}'
    def lookupNode(self, k):
        try:
            v0 = self.vertices[k]
            return v0
        except KeyError as e:
            return None
    def lookupEdge(self, i):
        try:
            i2 = self.edges[i]
            return i2
        except KeyError as e:
            return None
    def node(self, i, vv):
        if self.lookupNode(i) is None:
            self.vertices[i] = vv
    def edge(self, i1, i2):
        """add an edge"""
        i2m = self.lookupEdge(i1)
        if i2m is None:
            self.edges[i1] = [i2]
        else:
            self.edges[i1] = [i2] + i2m
    def neighbors(self, refIx: int, transpose=False):
        nn = []
        print(str(self.edges.values()))
        print(str(self.edges.items()))
        if transpose:
            for j in self.edges.values():
                if j == refIx:
                    nn.append(j)
        else:
            for i in self.edges.items():
                if i == refIx:
                    nn.append(i)
        return nn

def graphFromList(ll):
    g = Graph()
    for i1, i2, v1, v2 in ll:
        g.node(i1, v1)
        g.node(i2, v2)
        g.edge(i1, i2)
    return g


        # if self.lookupNode(i1) is not None and self.lookupEdge(i2) is not None:
        #     en = self.lookupEdge(i1)
        #     if en is None or not(en == i1):
        #         self.e[i1] = i2



# class Graph:
#     def __init__(self):
#         self.d = {}
#     def __iter__(self):
#         for e in self.d.items():
#             yield e
#     def __repr__(self):
#         return f'{str(self.d)}'
#     def tuples(self):
#         for (i, j), v in self.d.items():
#             yield i, j, v
#     def insert(self, ij: (int, int), v):
#         i, j = ij
#         self.d[(i, j)] = v
#     def neighbors(self, refIx:int, transpose=False):
#         nn = []
#         for i, j in self.d.keys():
#             if transpose:
#                 if j == refIx:
#                     nn.append(i)
#             else:
#                 if i == refIx:
#                     nn.append(j)
#         return nn
#
# def graphFromList(ll):
#     g = Graph()
#     for k, v in ll:
#         #print(f'{k}, {v}')
#         g.insert(k, v)
#     return g


