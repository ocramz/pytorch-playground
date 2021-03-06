from torch import Tensor, cuda, argmax, squeeze, unsqueeze, no_grad
from torch.nn import MSELoss, CrossEntropyLoss
from torch.optim import SGD
from torch.utils.data import DataLoader
import wandb

from string_helpers import TextDataset, Tokenize, embedStringOH
from gru import GRUClassifier
from gru_orig import GRUClassifierOrig
from bimap import BiMap
from misc import Timer

# Get cpu or gpu device for training.
device = "cuda" if cuda.is_available() else "cpu"
print(f"Using {device}")
log_out = False  # enable logging
# # hparams
num_epochs = 1  # number of epochs
learning_rate = 0.01
batch_size = 3  # batch size
# # dataset
fpath = 'data/alice'
xdim = 20  # vector dimension
hdim = 5  # dimension of latent vector h
nlayers = 2  # number of GRU layers
strLen = 30
tok = Tokenize('\n\r\t\"\'')
dataset = TextDataset(fpath, xdim, strLen, tok)
cats = dataset.numClasses()
vocab = dataset.voc  # ngram vocabulary
training_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# # # W&B
if log_out:
    wandb.init(
        project="pytorch-playground",
        entity="unfoldml",
        config={
            "learning_rate": learning_rate,
            "epochs": num_epochs,
            "batch_size": batch_size,
            "dim_hidden" : hdim,
            "n_layers" : nlayers,
            "string_length" : strLen,
            "token_separators" : tok.getSep(),
            "n_categories": cats
            })

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
        # h = model.init_hidden(batch_size)
        avg_loss, dt = train1(epoch)
        if log_out:
            print(f'average loss : {avg_loss}')
            # # # W&B log
            wandb.log({"epoch": epoch, "loss": avg_loss, "batch_time": dt})
    model.train(False)  # freeze weights
    evaluate()  # eyeball results on trained model

def train1(epoch):
    """train a single epoch"""
    running_loss = 0.
    avg_dt = 0.
    for i, data in enumerate(training_loader):
        inputs, labels = data  # get data batch
        if epoch == 0 and i == 0 :
            print(f'INPUTS : {inputs.size()}, LABELS : {labels.size()}')
        optim.zero_grad()  # zero out gradient
        # t0 = time.time()  # start timer
        with Timer() as timer:
            # # # gru.py
            # outputs = model(inputs)  # eval model

            # # # gru_orig.py
            outputs, h = model(inputs.to(device).float())

            loss = loss_fn(outputs, labels)  # compute loss
            loss.backward()  # compute gradient of loss
            optim.step()
            # t1 = time.time()  # stop timer
            # dt = t1 - t0
            running_loss += loss.item()
            avg_dt += timer.dt
            if i == batch_size - 1:
                avg_batch_loss = running_loss / batch_size  # average loss per batch
                avg_dt = avg_dt / batch_size

    return avg_batch_loss, avg_dt

def evaluate():
    s = "flavour of cherry-tart, custard, pine-apple"
    x, y = embedStringOH(vocab, s, tok, xdim)
    x = unsqueeze(x, 0)  # model expects batch index as first dimension
    # print(f'INPUTS : {x.size()}, LABELS : {y.size()}')

    with no_grad():
        yhat, h = model(x)  # evaluate trained model
        label = decodeOneHot(y, vocab)
        label_hat = decodeOneHot(squeeze(yhat, dim=0), vocab)

        # ilabel = argmax(y)
        # ilabel_hat = argmax(squeeze(yhat, dim=0))
        # label = list(vocab.lookupKs(ilabel))
        # label_hat = list(vocab.lookupKs(ilabel_hat))

        # print(f'real : {y.size()}\npredicted : {yhat.size()}')
        print(f'real : {label}\npredicted : {label_hat}')
        # print(f'real : {label}, predicted : {label_hat}')

def decodeOneHot(y: Tensor, voc: BiMap):
    ilabel = argmax(y)
    print(y.size(), y, ilabel)
    # return list(voc.lookupKs(ilabel))



if __name__ == '__main__':
    train()

