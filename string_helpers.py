from torch import Tensor, tensor, zeros, ones, randn, diag, cuda, Size
# from torchtext.data import get_tokenizer
from torchtext.vocab import Vocab, vocab, build_vocab_from_iterator
import io
import re


def ngrams(input_list, n):
    zs = zip(*[input_list[i:] for i in range(n)])
    return [''.join(list(z)) for z in zs]


class Tokenize:
    def __init__(self, sep=' ;:.,)(_\"\'‘”“\n'):
        self.sep = sep
    def getSep(self):
        """return the separators"""
        return list(self.sep)
    def split(self, s: str):
        sepPattern = f'[{self.sep}]+'
        toks = re.split(sepPattern, s)
        return [t for t in toks if t != '']


def yield_tokens(file_path, tok: Tokenize):
    with io.open(file_path, encoding='utf-8', mode='r') as f:
        for line in f:
            toksClean = tok.split(line)
            print(toksClean)
            if toksClean:
                yield toksClean


def embedString(voc: Vocab, s: str, tok:Tokenize):
    """tokenize the string and look up the tokens in the Vocab (dictionary) object"""
    # iis = s.strip(sep).split(sep)
    iis = tok.split(s)
    ils = voc.lookup_indices(iis)
    return tensor(ils)


def mkVocab(file_path, tok: Tokenize):
    """build a Vocab out of a text file"""
    v = build_vocab_from_iterator(yield_tokens(file_path, tok),
                                  specials=["<unk>"])
    v.set_default_index(0)
    return v
