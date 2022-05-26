from torch import cuda
from torch.nn import MSELoss, CrossEntropyLoss
from torch.optim import SGD
from torch.utils.data import DataLoader
from string_helpers import TextDataset
from gru import GRU

# Get cpu or gpu device for training.
device = "cuda" if cuda.is_available() else "cpu"
print(f"Using {device}")

# hparams
num_epochs = 60
learning_rate = 0.001

# dataset
fpath = 'data/alice'
xdim = 20  # vector dimension
strLen = 30
dataset = TextDataset(fpath, xdim, strLen)
training_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# model
model = GRU(5, xdim).to(device)
# print(model)

# loss
loss_fn = CrossEntropyLoss()
optim = SGD(model.parameters(), lr=learning_rate)

def train():
    for epoch in range(num_epochs):
        print(f'epoch {epoch}')
        model.train(True)
        avg_loss = train1()

def train1():
    running_loss = 0.
    for i, data in enumerate(training_loader):
        inputs, labels = data
        optim.zero_grad()  # zero out gradient
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()  # compute gradient of loss
        optim.step()
        running_loss += loss.item()
        if i % 1000 == 999:
            last_loss = running_loss / 1000  # loss per batch
    return last_loss


if __name__ == '__main__':
    train()

