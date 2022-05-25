import torch as nn
from torch.utils.data import DataLoader
from torchtext.vocab import Vocab, build_vocab_from_iterator
import io

def yield_tokens(file_path):
     with io.open(file_path, encoding = 'utf-8', mode='r') as f:
         for line in f:
             yield line.strip().split()

def mkVocab(file_path) :
    return build_vocab_from_iterator(yield_tokens(file_path), specials=["<unk>"])

if __name__ == '__main__':
    fpath = 'data/text0'
    voc = mkVocab(fpath)
    print(voc.lookup_token(2))
    # for v in voc:
    #     print(v)