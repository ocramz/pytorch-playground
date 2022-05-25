from torch import Tensor, tensor, diag
from torch.nn import Module
from torch.linalg import matmul
from torch.nn.functional import linear, sigmoid
from torch.utils.data import DataLoader
from torchtext.vocab import Vocab, build_vocab_from_iterator
import io

def hadamard(a, b):
    """Hadamard product"""
    return linear(a, diag(b))

class GRU(Module):
    def __init__(self):
        super().__init__()
        pass
    def forward(self):
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