from torch import Tensor, tensor, zeros, ones, randn, diag, cuda, _assert, Size
from torch.nn import Module, Sequential, Linear, Sigmoid
from torch.linalg import matmul
from torch.nn.functional import linear, sigmoid, tanh, one_hot
from torch.utils.data import DataLoader
from torchtext.vocab import Vocab, vocab, build_vocab_from_iterator
from collections import Counter, OrderedDict
import io

# Get cpu or gpu device for training.
device = "cuda" if cuda.is_available() else "cpu"
print(f"Using {device}")


def ordHistogram(xs):
    """histogram in decreasing count order"""
    count = Counter(xs)
    return OrderedDict(sorted(count.items(), key=lambda x: x[1], reverse=True))

def oneHot(v: Vocab, keys):
    """one-hot encoding of a vector given a vocabulary
    :returns Tensor"""
    return one_hot(tensor(v(keys)), num_classes=len(voc))

def assertIsVector(x: Tensor):
    _assert(len(x.size()) == 1, f'{x} should be a vector')


def hadamard(a: Tensor, b: Tensor):
    """Hadamard (componentwise) product of two vectors
    :returns Tensor"""
    _assert(a.size() == b.size(), 'arguments should have the same size')
    assertIsVector(a)
    assertIsVector(b)
    return linear(a, diag(b))


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
            o.zt = sigmoid(matmul(o.Wz, x) + matmul(o.Uz, o.hprev))
            o.htilde = tanh(matmul(o.Wh, x) + hadamard(o.rt, matmul(o.Uh, o.hprev)))
            o.rt = sigmoid(matmul(o.Wr, x) + matmul(o.Ur, o.hprev))
            o.hprev = o.h  # update h_(t-1)

model = GRU(5, 10).to(device)
print(model)


def yield_tokens(file_path, sep=' '):
    with io.open(file_path, encoding='utf-8', mode='r') as f:
        for line in f:
            yield line.strip(sep).split(sep)


def embedString(voc:Vocab, s:str, sep=' '):
    """tokenize the string and look up the tokens in the Vocab (dictionary) object"""
    iis = s.strip(sep).split(sep)
    ils = voc.lookup_indices(iis)
    return tensor(ils)


def mkVocab(file_path):
    """build a Vocab out of a text file"""
    v = build_vocab_from_iterator(yield_tokens(file_path),
                                  specials=["<unk>"])
    v.set_default_index(0)
    return v


if __name__ == '__main__':
    fpath = 'data/alice'
    voc = mkVocab(fpath)
    print(voc.lookup_token(0))
    print(embedString(voc, 'Alice took a xyz'))
    # for v in voc:
    #     print(v)
