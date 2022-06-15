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
    def __init__(self, din:int, dout:int, nslope):
        super(GATv2, self).__init__()
        self.W = Linear(din, dout, bias=True)
        self.relu = LeakyReLU(negative_slope=nslope)
        self.a = Linear(dout, 1, bias=False)
    def score(o, hi, hj):
        """score an edge given its node embeddings"""
        h = cat(hi, hj, dim=0)  # concatenate node features
        hw = o.W(h)  # linear
        return o.a(o.relu(hw))  # linear . relu
    def forward(o, gr: Graph):
        """updates embeddings of the _whole_ graph"""
        for i, hi in gr.nodes():
            ni = gr.neighbors(i)  # N(i)
            n = len(ni)
            hjs = zeros((len(hi), len(ni)))  # embeddings of neighbors
            eij = zeros(n)  # scores of neighboring edges
            for j in ni:
                hj = gr.lookupNode(j)
                hjs[:, j] = hj
                eij[j] = o.score(hi, hj)
            alphaij = softmax(eij)  # attention scores
            hiPrime = sigmoid(matmul(alphaij, o.W(hjs)))
            gr.node(i, hiPrime)  # update node embedding
        return gr
