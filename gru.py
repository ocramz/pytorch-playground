from torch import Tensor, tensor, zeros, ones, randn, diag, cuda, _assert, Size
from torch.nn import Module, Sequential, Linear, Sigmoid
from torch.linalg import matmul
from torch.nn.functional import linear, sigmoid, tanh, one_hot
# from torch.utils.data import DataLoader
# from torchtext.data import get_tokenizer
from torchtext.vocab import Vocab, vocab, build_vocab_from_iterator
from collections import Counter, OrderedDict
# from string_helpers import mkVocab, embedString, Tokenize

class GRU(Module):
    def __init__(self, nh:int, d:int):
        super(GRU, self).__init__()
        self.htilde = randn(nh)  # candidate state
        self.hprev = randn(nh)  # h_{t-1} is random at time 0
        self.h = randn(nh)  # h_{t} is random at time 0
        self.zt = randn(nh)  # update gate
        self.rt = randn(nh)  # reset gate
        self.Wh = Linear(nh, d, bias=False)
        self.Wz = Linear(nh, d, bias=False)
        self.Wr = Linear(nh, d, bias=False)
        self.Uh = Linear(nh, d, bias=False)
        self.Uz = Linear(nh, d, bias=True)
        self.Ur = Linear(nh, d, bias=True)

    def forward(o, xs):
        for i, x in enumerate(xs):
            o.h = hadamard(1 - o.zt, o.hprev) + hadamard(o.zt, o.htilde)
            o.zt = sigmoid(o.Wz(x) + o.Uz(o.hprev))
            o.htilde = tanh(o.Wh(x) + o.Uh(o.hprev))
            o.rt = sigmoid(o.Wr(x) + o.Ur(o.hprev))
            # o.zt = sigmoid(matmul(o.Wz, x) + matmul(o.Uz, o.hprev))
            # o.htilde = tanh(matmul(o.Wh, x) + hadamard(o.rt, matmul(o.Uh, o.hprev)))
            # o.rt = sigmoid(matmul(o.Wr, x) + matmul(o.Ur, o.hprev))
            o.hprev = o.h  # update h_(t-1)



def hadamard(a: Tensor, b: Tensor):
    """Hadamard (componentwise) product of two vectors
    :returns Tensor"""
    _assert(a.size() == b.size(), 'arguments should have the same size')
    assertIsVector(a)
    assertIsVector(b)
    return linear(a, diag(b))

def assertIsVector(x: Tensor):
    _assert(len(x.size()) == 1, f'{x} should be a vector')

def ordHistogram(xs):
    """histogram in decreasing count order"""
    count = Counter(xs)
    return OrderedDict(sorted(count.items(), key=lambda x: x[1], reverse=True))

def oneHot(voc: Vocab, keys):
    """one-hot encoding of a vector given a vocabulary
    :returns Tensor"""
    return one_hot(tensor(voc(keys)), num_classes=len(voc))

if __name__ == '__main__':
    pass

    # fpath = 'data/alice'
    # tok = Tokenize(sep=',. ()\n_“”')
    # voc = mkVocab(fpath, tok)
    # # print(voc.lookup_token(0))
    # print(embedString(voc, 'Alice took a xyz and found it disagree', tok))
    # # for v in voc:
    # #     print(v)
