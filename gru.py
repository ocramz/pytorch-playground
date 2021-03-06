from torch import Tensor, tensor, zeros, zeros_like, ones, randn, rand, diag, cuda, _assert, sigmoid,tanh, Size, transpose
from torch.nn import Module, Sequential, Linear, Sigmoid
# from torch.linalg import matmul
from torch.nn.functional import linear, one_hot, softmax
# from torch.utils.data import DataLoader
# from torchtext.data import get_tokenizer
# from torchtext.vocab import Vocab, vocab, build_vocab_from_iterator
# from collections import Counter, OrderedDict
# from string_helpers import mkVocab, embedString, Tokenize

class GRUClassifier(Module):
    """GRU followed by a linear classification layer"""
    def __init__(self, nh:int, cats:int):
        """
        :param nh: dim of state vector h
        :param cats: number of categories
        """
        super(GRUClassifier, self).__init__()
        self.gru = GRU(nh, cats)
        self.out = Linear(nh, cats, bias=False)
    def forward(o, x):
        y = o.gru(x)
        # y2 = softmax(o.out(y), dim=0)
        y2 = o.out(y)
        return y2


class GRU(Module):
    """gated recurrent unit"""
    def __init__(self, nh:int, cats:int):
        """
        :param nh: dim of state vector h
        :param cats: embedding dim of x_t (= # of categories)
        """
        super(GRU, self).__init__()
        self.nh = nh  # latent vector dimension
        self.Wh = Linear(cats, nh, bias=False)  # nh * d
        self.Wz = Linear(cats, nh, bias=False)
        self.Wr = Linear(cats, nh, bias=False)
        self.Uh = Linear(nh, nh, bias=False)  # nh * nh
        self.Uz = Linear(nh, nh, bias=True)
        self.Ur = Linear(nh, nh, bias=True)
    def forward(o, xbatch):
        nbatch = xbatch.size(0)
        t = zeros(nbatch, o.nh)
        for i, x in enumerate(xbatch):
            o.htilde = randn(o.nh)  # candidate state
            o.hprev = randn(o.nh)  # h_{t-1} is random at time 0
            o.h = randn(o.nh)  # h_{t} is random at time 0
            o.zt = randn(o.nh)  # update gate
            o.rt = randn(o.nh)  # reset gate
            for xt in x:
                o.rt = sigmoid(o.Wr(xt) + o.Ur(o.hprev))
                o.zt = sigmoid(o.Wz(xt) + o.Uz(o.hprev))
                o.htilde = tanh(o.Wh(xt) + o.Uh(o.hprev))
                o.h = hadamard(1 - o.zt, o.hprev) + hadamard(o.zt, o.htilde)
                o.hprev = o.h  # update h_(t-1)
            t[i, :] = o.h
        return t # h_fin

def hadamard(a: Tensor, b: Tensor):
    """Hadamard (componentwise) product of two vectors
    :returns Tensor"""
    _assert(a.size() == b.size(), f'arguments should have the same size, rather than {a.size()}, {b.size()}')
    assertIsVector(a)
    assertIsVector(b)
    return linear(a, diag(b))

def assertIsVector(x: Tensor):
    _assert(len(x.size()) == 1, f'{x} should be a vector')

# def ordHistogram(xs):
#     """histogram in decreasing count order"""
#     count = Counter(xs)
#     return OrderedDict(sorted(count.items(), key=lambda x: x[1], reverse=True))



if __name__ == '__main__':
    pass

    # fpath = 'data/alice'
    # tok = Tokenize(sep=',. ()\n_??????')
    # voc = mkVocab(fpath, tok)
    # # print(voc.lookup_token(0))
    # print(embedString(voc, 'Alice took a xyz and found it disagree', tok))
    # # for v in voc:
    # #     print(v)
