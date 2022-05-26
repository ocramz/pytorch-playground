from torch import Tensor, tensor, zeros, ones, randn, diag, cuda, Size, from_numpy
import numpy as np
# from torchtext.data import get_tokenizer
from torchtext.vocab import Vocab, vocab, build_vocab_from_iterator
import io
import re
from itertools import chain
from collections import Counter, OrderedDict
from bimap import BiMap

def ngrams(input_list, n):
    zs = zip(*[input_list[i:] for i in range(n)])
    return [''.join(list(z)) for z in zs]


class Tokenize:
    def __init__(self, sep=' ;:.,)(_\"\'‘”“\n\t\r', ngramSize=2):
        self.sep = sep
        self.ngramSize = ngramSize
    def getSep(self):
        """
        :returns list of separators"""
        return list(self.sep)
    def split(self, s: str):
        """
        :rtype list of str"""
        sepPattern = f'[{self.sep}]+'
        toks = re.split(sepPattern, s)
        return [t for t in toks if t != '']
    def ngrams(self, s: str):
        """
        :returns iterator of string n-grams
        :rtype list of str"""
        toks = self.split(s)
        # print(toks)
        n = self.ngramSize
        # ngit = chain.from_iterable(ngrams(t, n) for t in toks)  # iterable
        # ngrs = list(ngit)
        ngrs = []
        for t in toks:
            # print(t, ngrs)
            ngrs = ngrs + ngrams(t, n)
        return ngrs
    def ngramCounts(self, s:str):
        ngrs = self.ngrams(s)
        return Counter(ngrs)


def t0():
    s = """Alice was beginning to get very tired of sitting by her sister on the
    bank, and of having nothing to do: once or twice she had peeped into
    the book her sister was reading, but it had no pictures or
    conversations in it, “and what is the use of a book,” thought Alice
    “without pictures or conversations?”
    """
    tok = Tokenize()
    ngrs = tok.ngrams(s)
    voc = BiMap(ngrs)
    # print(voc)
    testStr = 'books are nice and not boring'
    # print(tok.ngrams(testStr))
    return embedStringBM(voc, 'books are nice and not boring', tok)



def yield_tokens(file_path, tok: Tokenize):
    with io.open(file_path, encoding='utf-8', mode='r') as f:
        for line in f:
            toksClean = tok.split(line)
            print(toksClean)  # debug
            if toksClean:
                yield toksClean

def embedStringBM(voc: BiMap, s:str, tok:Tokenize):
    """:returns a Tensor"""
    iis = tok.ngrams(s)
    ils = voc.lookupVDs(iis)
    # print(list(ils))  # debug
    return from_numpy(np.fromiter(ils, dtype=int))

def embedString(voc: Vocab, s: str, tok: Tokenize):
    """tokenize the string and look up the tokens in the Vocab (dictionary) object"""
    iis = tok.split(s)
    ils = voc.lookup_indices(iis)
    return tensor(ils)

def mkVocab(file_path, tok: Tokenize):
    """build a Vocab out of a text file"""
    v = build_vocab_from_iterator(yield_tokens(file_path, tok),
                                  specials=["<unk>"])
    v.set_default_index(0)
    return v
