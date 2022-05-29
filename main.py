from torch import cuda
from torch.nn import MSELoss, CrossEntropyLoss
from torch.optim import SGD
from torch.utils.data import DataLoader
from string_helpers import TextDataset, Tokenize
from gru import GRUClassifier
from gru_orig import GRUClassifierOrig

# Get cpu or gpu device for training.
device = "cuda" if cuda.is_available() else "cpu"
print(f"Using {device}")
# # hparams
num_epochs = 60
learning_rate = 0.01
batch_size = 32  # batch size
# # dataset
fpath = 'data/alice'
xdim = 20  # vector dimension
hdim = 5  # dimension of latent vector h
nlayers = 5  # number of GRU layers
strLen = 30
tok = Tokenize('\n\r\t\"\'')
dataset = TextDataset(fpath, xdim, strLen, tok)
cats = dataset.numClasses()
training_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# # model
# model = GRUClassifier(hdim, cats).to(device)
model = GRUClassifierOrig(xdim, hdim, cats, nlayers).to(device)

print(model)
# # loss
loss_fn = CrossEntropyLoss()
# # optimizer
optim = SGD(model.parameters(), lr=learning_rate)

def train():
    model.train(True)  # thaw weights
    for epoch in range(num_epochs):
        print(f'epoch {epoch}')
        h = model.init_hidden(batch_size)
        avg_loss = train1(h)

def train1(h):
    running_loss = 0.
    for i, data in enumerate(training_loader):
        inputs, labels = data  # get data batch
        optim.zero_grad()  # zero out gradient

        # # # gru.py
        # outputs = model(inputs)  # eval model

        # # # gru_orig.py
        print(f'INPUTS : {inputs.size()}, LABELS : {labels.size()}')
        outputs, h = model(inputs.to(device).float(), h)

        # print(f'INPUTS : {inputs.size()}, LABELS : {labels.size()}, OUTPUTS : {outputs.size()}')
        loss = loss_fn(outputs, labels)  # compute loss

        loss.backward()  # compute gradient of loss
        optim.step()
        running_loss += loss.item()
        if i % 10 == 9:
            print(f'LOSS : {loss}')
            last_loss = running_loss / 1000  # loss per batch

    return last_loss


if __name__ == '__main__':
    train()

