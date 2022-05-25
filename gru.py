from torch import Tensor, tensor, zeros, ones, randn, diag, cuda
from torch.nn import Module, Sequential, Linear, Sigmoid
from torch.linalg import matmul
from torch.nn.functional import linear, sigmoid
from torch.utils.data import DataLoader
from torchtext.vocab import Vocab, build_vocab_from_iterator
import io

# Get cpu or gpu device for training.
device = "cuda" if cuda.is_available() else "cpu"
print(f"Using {device}")

def hadamard(a, b):
    """Hadamard product"""
    return linear(a, diag(b))

class GRU(Module):
    def __init__(self, nh):
        super(GRU, self).__init__()
        self.h = randn(nh)  # initial state
        self.hprev = randn(nh)
        self.zt = randn(nh)
        self.Wz = Linear(nh, nh, bias=False)
        self.Uz = Linear(nh, nh, bias=True)
    def forward(self, xs):
        for i, x in enumerate(xs):
            if i == 0:
                self.h = hadamard(1 - self.zt, self.hprev) + hadamard(self.zt, self.h)
                self.hprev = self.h  # update h_(t-1)
                self.zt = sigmoid()
            else:
                pass
        pass


def yield_tokens(file_path, sep=' '):
     with io.open(file_path, encoding = 'utf-8', mode='r') as f:
         for line in f:
             yield line.strip(sep).split(sep)

def embedString(voc, s, sep=' '):
    """tokenize the string and look up the tokens in the Vocab (dictionary) object"""
    iis = s.strip(sep).split(sep)
    ils = voc.lookup_indices(iis)
    return tensor(ils)

def mkVocab(file_path) :
    """build a Vocab out of a text file"""
    v = build_vocab_from_iterator(yield_tokens(file_path),
                                     specials=["<unk>"])
    v.set_default_index(0)
    return v

if __name__ == '__main__':
    fpath = 'data/text0'
    voc = mkVocab(fpath)
    print(voc.lookup_token(0))
    print(embedString(voc, 'Alice took a xyz'))
    # for v in voc:
    #     print(v)