from torch import Tensor, cuda, argmax, squeeze, unsqueeze, no_grad
from torch.nn import MSELoss, CrossEntropyLoss
from torch.optim import SGD
from torch.utils.data import DataLoader
import wandb

from GATv2_1 import GATv2
from misc import Timer

# Get cpu or gpu device for training.
device = "cuda" if cuda.is_available() else "cpu"
print(f"Using {device}")
log_out = False  # enable logging
# # hparams
num_epochs = 1  # number of epochs
learning_rate = 0.01
# # params
din = 10
dout = 10
nslope = 0.02

# # # W&B
if log_out:
    wandb.init(
        project="GATv2_1",
        entity="unfoldml",
        config={
            "learning_rate": learning_rate,
            "epochs": num_epochs,
            })

# #
model = GATv2(din, dout, nslope)

if __name__ == '__main__':
    print(model)