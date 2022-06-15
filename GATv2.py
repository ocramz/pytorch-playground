import numpy as np
from torch import Tensor, tensor, sparse_coo_tensor, zeros, zeros_like, ones, randn, rand, diag, cat, cuda, _assert, \
    sigmoid, tanh, Size, transpose, matmul
from torch.nn import Module, Sequential, Linear, Sigmoid, LeakyReLU
from torch.utils.data import Dataset, DataLoader
# from torch.linalg import matmul
from torch.nn.functional import linear, one_hot, softmax

from graph import Graph, graphFromList


class GATv2(Module):
    """graph attention layer v2, from Brody et al, ICLR 2022
    :param nin: input dimension
    :param nout: output dimension
    :param nslope: LeakyReLU negative slope parameter
    """
    def __init__(self, nin:int, nout:int, nslope):
        super(GATv2, self).__init__()
        self.W = Linear(nin, nout, bias=True)
        self.relu = LeakyReLU(negative_slope=nslope)
        self.a = Linear(nout, 1, bias=False)
    def score(o, hi, hj):
        """score an edge given its node embeddings"""
        h = cat(hi, hj, dim=0)  # concatenate node features
        hw = o.W(h)  # linear
        return o.a(o.relu(hw))  # linear . relu
    def forward(o, hi:Tensor, hjs:Tensor):
        """
        :param hi: (nin * 1) embedding of node i
        :param hjs: (nin * nni) embeddings of i's neighbors N(i)
        :return: hi', updated embedding of node i
        """
        n = hjs.size(1)  # |N(i)|
        eij = zeros(n)
        for j in range(n):
            hj = hjs[:, j]
            eij[j] = o.score(hi, hj)
        alphaij = softmax(eij) # attention scores
        hiPrime = sigmoid(matmul(alphaij, o.W(hjs)))
        return hiPrime

class GraphDataset(Dataset):
    def __init__(self, gr:Graph):
        self.gr = gr
    def __len__(self):
        return self.gr.numNodes()
    def __getitem__(self, i):
        hi = self.gr.lookupNode(i) # embedding of node i
        ni = self.gr.neighbors(i)  # N(i)
        hjs = zeros((len(hi), len(ni)))  # embeddings of neighbors
        for j in ni:
            hj = self.gr.lookupNode(j)
            hjs[:, j] = hj
        return hi, hjs



    # def forward(o, coo):
    #     """:param coo: an iterable of (row, col, Tensor, Tensor) with the edge node indices and node embeddings"""
    #     nEdges = len(coo)
    #     iis = np.zeros(nEdges)
    #     jjs = np.zeros(nEdges)
    #     vvs = np.zeros(nEdges)
    #     for ilist, (i, j, hi, hj) in enumerate(coo):
    #         eScore = o.score(hi, hj)
    #         iis[ilist] = i
    #         jjs[ilist] = j
    #         vvs[ilist] = eScore
    #     scoreMtx = sparse_coo_tensor([iis, jjs], vvs)
    #     attn = softmax(scoreMtx, dim=1)  # attention weights alpha_i,j





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


