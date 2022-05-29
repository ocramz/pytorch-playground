from torch import randn, device, cuda
from torch.nn import Module, GRU, Linear, ReLU

if cuda.is_available():
    device = device("cuda")
else:
    device = device("cpu")

class GRUClassifierOrig(Module):
    def __init__(self, indim:int, hdim:int, outdim:int, nlayers:int, drop_prob=0.2):
        """
        :param indim: input dimension
        :param hdim:  latent state dimension
        :param outdim: output dimension
        :param nlayers: number of layers
        :param drop_prob: dropout probability
        """
        super(GRUClassifierOrig, self).__init__()
        self.indim = indim
        self.hdim = hdim  # dimension of latent state
        self.nlayers = nlayers
        self.gru = GRU(indim, hdim, nlayers, batch_first=True, dropout=drop_prob)
        self.fc = Linear(hdim, outdim)
        # self.relu = ReLU()
    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = weight.new(self.nlayers, batch_size, self.hdim).zero_().to(device)
        return hidden
    def forward(self, xbatch):
        out, h = self.gru(xbatch)
        out2 = self.fc(out[:, -1])
        # out2 = self.relu(self.fc(out[:, -1]))
        return out2, h
