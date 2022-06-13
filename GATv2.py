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
        self.d = {}
    def __iter__(self):
        for e in self.d.items():
            yield e
    def __repr__(self):
        return f'{str(self.d)}'
    def insert(self, ij: (int, int), v):
        i, j = ij
        self.d[(i, j)] = v
    def neighbors(self, refIx:int, transpose=False):
        nn = []
        for i, j in self.d.keys():
            if transpose:
                if j == refIx:
                    nn.append(i)
            else:
                if i == refIx:
                    nn.append(j)
        return nn


def graphFromList(ll):
    g = Graph()
    for k, v in ll:
        #print(f'{k}, {v}')
        g.insert(k, v)
    return g


