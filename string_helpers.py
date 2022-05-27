from torch import Tensor, tensor, zeros, ones, randn, diag, cuda, Size, from_numpy
from torch.utils.data import Dataset, DataLoader
import numpy as np
# from torchtext.data import get_tokenizer
from torchtext.vocab import Vocab, vocab, build_vocab_from_iterator
import io
from os import path
import re
from itertools import chain
from linecache import getline
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
        n = self.ngramSize
        ngit = chain.from_iterable(ngrams(t, n) for t in toks)  # iterable
        ngrs = list(ngit)
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
    So she was considering in her own mind (as well as she could, for the
hot day made her feel very sleepy and stupid), whether the pleasure of
making a daisy-chain would be worth the trouble of getting up and
picking the daisies, when suddenly a White Rabbit with pink eyes ran
close by her.
There was nothing so _very_ remarkable in that; nor did Alice think it
so _very_ much out of the way to hear the Rabbit say to itself, “Oh
dear! Oh dear! I shall be late!” (when she thought it over afterwards,
it occurred to her that she ought to have wondered at this, but at the
time it all seemed quite natural); but when the Rabbit actually _took a
watch out of its waistcoat-pocket_, and looked at it, and then hurried
on, Alice started to her feet, for it flashed across her mind that she
had never before seen a rabbit with either a waistcoat-pocket, or a
watch to take out of it, and burning with curiosity, she ran across the
field after it, and fortunately was just in time to see it pop down a
large rabbit-hole under the hedge.
    """
    tok = Tokenize(ngramSize=2)
    ngrs = tok.ngrams(s)
    voc = BiMap(ngrs)
    testStr = 'books are nice and not boring'
    return embedStringBM(voc, testStr, tok)

class TextDataset(Dataset):
    def __init__(self, fpath, xdim, strLen = 20, tok:Tokenize = Tokenize()):
        """
        :param fpath: path of text data file
        :param xdim: vector dimension
        :param strLen: max string length
        :param tok:  tokenizer
        """
        self.fpath = fpath
        self.tok = tok
        self.strLen = strLen
        self.xdim = xdim
        self.voc = ngramsFromTextFile(fpath, tok)
    def __len__(self):
        _, n = fileBounds(self.fpath)
        return n
    def __getitem__(self, ix):
        s = stringAtIx(self.fpath, ix, self.strLen)
        print(s)  # debug
        x, y = embedStringBM(self.voc, s, self.tok, self.xdim)
        return x, y





def stringAtIx(fpath, ix, k):
    """
    :param fpath: file path
    :param ix: linear index >= 0
    :param k: max string length
    :returns string of length (at most) k starting at index ix"""
    numLines, fsizeBytes = fileBounds(fpath)
    bytesPerLine = fsizeBytes // numLines
    m = ix // numLines + 1
    n = ix % bytesPerLine
    return stringFromTextFile(fpath, m, n, k)

def fileBounds(fpath):
    """:returns number of lines, file size in bytes"""
    with open(fpath) as f:
        numLines = len(f.readlines())
        fsizeBytes = path.getsize(fpath)
        return numLines, fsizeBytes

def stringFromTextFile(fpath, m, n, k):
    """load a text string of given length from a given row, colum of a file
    :param m: line #
    :param n: column #
    :param k: string max length
    """
    txt = getline(fpath, m)[n:]
    ll = len(txt)
    if ll < k:
        return txt
    elif ll > k:
        return txt[0:k]
    else:
        return txt

def ngramsFromTextFile(fpath, tok:Tokenize = Tokenize()):
    """build a vocabulary of n-grams from a text file
    :returns BiMap of text n-grams"""
    bm = BiMap()
    with io.open(fpath, encoding='utf-8', mode='r') as f:
        for line in f:
            ngrs = tok.ngrams(line)
            bm.fromIter(ngrs)
        return bm



def embedStringBM(voc: BiMap, s:str, tok:Tokenize, dim0: int=10):
    """:returns a Tensor of given dimension and label. Pads missing elements or removes extra ones"""
    iis = tok.ngrams(s)
    ils = voc.lookupVDs(iis)
    z = voc.defaultIx
    ilsl = list(ils)
    d = len(ilsl)  # length of n-gram list
    dim = dim0 + 1
    if d == dim:
        ilsPrep = ilsl
    elif d < dim:
        ilsPrep = ilsl + ([z] * (dim - d))
    else:
        ilsPrep = ilsl[0:dim]
    x = from_numpy(np.fromiter(ilsPrep[0:dim0], dtype=np.float32))
    y = ilsPrep[-1]  # label is the _last_ element
    return x, y

# # Vocab-based

# def embedString(voc: Vocab, s: str, tok: Tokenize):
#     """tokenize the string and look up the tokens in the Vocab (dictionary) object"""
#     iis = tok.split(s)
#     ils = voc.lookup_indices(iis)
#     return tensor(ils)
#
# def mkVocab(file_path, tok: Tokenize):
#     """build a Vocab out of a text file"""
#     v = build_vocab_from_iterator(yield_tokens(file_path, tok),
#                                   specials=["<unk>"])
#     v.set_default_index(0)
#     return v
#
# def yield_tokens(file_path, tok: Tokenize):
#     with io.open(file_path, encoding='utf-8', mode='r') as f:
#         for line in f:
#             toksClean = tok.split(line)
#             print(toksClean)  # debug
#             if toksClean:
#                 yield toksClean