from torch import cuda
from torch.nn import MSELoss, CrossEntropyLoss
from torch.optim import SGD
from torch.utils.data import DataLoader
from string_helpers import TextDataset
from gru import GRUClassifier

# Get cpu or gpu device for training.
device = "cuda" if cuda.is_available() else "cpu"
print(f"Using {device}")
# # hparams
num_epochs = 60
learning_rate = 0.001
bsize = 32  # batch size
# # dataset
fpath = 'data/alice'
xdim = 20  # vector dimension
strLen = 30
dataset = TextDataset(fpath, xdim, strLen)
cats = dataset.numClasses()
training_loader = DataLoader(dataset, batch_size=bsize, shuffle=True)

# # model
model = GRUClassifier(5, xdim, cats).to(device)
print(model)
# # loss
loss_fn = CrossEntropyLoss()
# # optimizer
optim = SGD(model.parameters(), lr=learning_rate)

def train():
    for epoch in range(num_epochs):
        print(f'epoch {epoch}')
        model.train(True)  # thaw weights
        avg_loss = train1()

def train1():
    running_loss = 0.
    for i, data in enumerate(training_loader):
        inputs, labels = data  # get data batch
        print(f'LABELS : {labels.size()}') # debug
        optim.zero_grad()  # zero out gradient
        outputs = model(inputs)  # eval model
        print(f'OUTPUTS : {outputs.size()}')
        loss = loss_fn(outputs, labels)  # compute loss
        loss.backward()  # compute gradient of loss
        optim.step()
        running_loss += loss.item()
        if i % 1000 == 999:
            last_loss = running_loss / 1000  # loss per batch
    return last_loss


if __name__ == '__main__':
    train()

