import numpy as np
from torch import Tensor, tensor, sparse_coo_tensor, zeros, zeros_like, ones, randn, rand, diag, cat, cuda, _assert, sigmoid,tanh, Size, transpose
from torch.nn import Module, Sequential, Linear, Sigmoid, LeakyReLU
# from torch.linalg import matmul
from torch.nn.functional import linear, one_hot, softmax


class GATv2(Module):
    def __init__(self, nin:int, nout:int, nslope):
        super(GATv2, self).__init__()
        self.W = Linear(nin, nout, bias=True)
        self.relu = LeakyReLU(negative_slope=nslope)
        self.a = Linear(nout, 1, bias=False)
    def forward(o, coo):
        """:param coo: a list of (row, col, Tensor, Tensor) with the edge node indices and node embeddings"""
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
    def values(self):
        for v in self.__iter__():
            yield v


