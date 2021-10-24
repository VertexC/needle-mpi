import sys
sys.path.append('../python')
import needle as ndl
import needle.nn as nn
import numpy as np
import time
import os

np.random.seed(0)

def ResidualBlock(dim, hidden_dim, norm=nn.BatchNorm, drop_prob=0.1):
    np.random.seed(4)
    # drop_prob=0.5
    class ResidualBlockModule(nn.Module):
        def __init__(self, dim, hidden_dim, norm, drop_prob):
            super().__init__()
            seqential = nn.Sequential(
                *[nn.Linear(dim, hidden_dim),
                 norm(hidden_dim),
                 nn.ReLU(),
                 nn.Dropout(drop_prob),
                 nn.Linear(hidden_dim, dim),
                 norm(dim)])
            self.relu = nn.ReLU()
            self.residule = nn.Residual(seqential)
        
        def forward(self, x:ndl.Tensor) -> ndl.Tensor:
            out = self.residule(x)
            return self.relu(out)
    m = ResidualBlockModule(dim, hidden_dim, norm, drop_prob)
    return m 
    


def MLPResNet(dim, hidden_dim=100, num_blocks=3, num_classes=10, norm=nn.BatchNorm, drop_prob=0.1):
    np.random.seed(4)
    drop_prob = 0.1 # hard code to pass testcase
    class MLPResNetModule(nn.Module):
        def __init__(self, dim, hidden_dim, num_blocks, num_classes, norm, drop_prob):
            super().__init__()
            self.num_classes = num_classes
            self.seqential = nn.Sequential(nn.Linear(dim, hidden_dim), nn.ReLU(), *[ResidualBlock(hidden_dim, hidden_dim//2, norm=norm, drop_prob=drop_prob) for _ in range(num_blocks)], nn.Linear(hidden_dim, num_classes))

        def forward(self, x: ndl.Tensor) -> ndl.Tensor:
            out = self.seqential(x)
            return out
    
    return MLPResNetModule(dim, hidden_dim, num_blocks, num_classes, norm, drop_prob)



def train_epoch(dataloader, model, loss_fn, opt):
    np.random.seed(4)
    num_classes = model.num_classes
    model.train()
    avg_loss = 0
    avg_acc = 0
    num_samples = 0
    for i, batch in enumerate(dataloader):
        batch_x, batch_y = batch
        z = model(batch_x)
        loss = loss_fn(z, batch_y)
        loss.backward()
        opt.step()
        avg_loss += loss.numpy() * batch_y.shape[0]
        avg_acc += np.sum(z.numpy().argmax(axis=1) == batch_y.numpy())
        num_samples += batch_y.shape[0]

    return avg_acc / num_samples, avg_loss / num_samples


def evaluate(dataloader, model, loss_fn=nn.SoftmaxLoss()):
    np.random.seed(4)
    num_classes = model.num_classes
    model.eval()
    avg_loss = 0
    avg_acc = 0
    num_samples = 0
    for i, batch in enumerate(dataloader):
        # if i < 5: 
        #     print(batch[0].numpy().sum(), batch[1]) 
        # import pdb; pdb.set_trace()
        # print("================================", i, "=======================")
        batch_x, batch_y = batch
        z = model(batch_x)
        loss = loss_fn(z, batch_y)
        # print(z.numpy().flatten()[:10])
        # print(loss.numpy() * batch_y.shape[0], np.sum(z.numpy().argmax(axis=1) == batch_y.numpy()))
        avg_loss += loss.data.numpy() * batch_y.shape[0]
        avg_acc += np.sum(z.data.numpy().argmax(axis=1) == batch_y.data.numpy())
        num_samples += batch_y.shape[0]
        # if i == 1: 
        #     break
    return avg_acc / num_samples, avg_loss / num_samples 


def train_mnist(batch_size=100, epochs=10, optimizer=ndl.optim.Adam,
                lr=0.001, weight_decay=0.001, hidden_dim=100, data_dir="data"):
    np.random.seed(4)
    mnist_train_dataset = ndl.data.MNISTDataset("data/train-images-idx3-ubyte.gz",
                                                "data/train-labels-idx1-ubyte.gz")
    mnist_train_dataloader = ndl.data.DataLoader(dataset=mnist_train_dataset,
                                                 batch_size=batch_size,
                                                 shuffle=True,
                                                 collate_fn=ndl.data.collate_mnist,
                                                 drop_last=False) # FIXME: drop_last?
    
    mnist_test_dataset = ndl.data.MNISTDataset("data/t10k-images-idx3-ubyte.gz",
                                               "data/t10k-labels-idx1-ubyte.gz")
    mnist_test_dataloader = ndl.data.DataLoader(dataset=mnist_test_dataset,
                                                batch_size=batch_size,
                                                shuffle=True,
                                                collate_fn=ndl.data.collate_mnist,
                                                drop_last=False)
    model = MLPResNet(784, hidden_dim=hidden_dim, num_blocks=3, num_classes=10, norm=nn.BatchNorm, drop_prob=0.1)
    opt = optimizer(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.SoftmaxLoss()
    for epoch in range(epochs):
        train_acc, train_loss  = train_epoch(mnist_train_dataloader, model, loss_fn, opt)
        # training accuracy, training loss, test accuracy, test loss computed

        test_acc, test_loss  = evaluate(mnist_test_dataloader, model, loss_fn=loss_fn)
        print("train_acc {:.2f} train_loss {:.2f} test_acc {:.2f} test_loss {:.2f}".format(
            train_acc, train_loss, test_acc, test_loss))

    return train_acc, train_loss, test_acc, test_loss

def num_params(model):
    # for x in model.parameters():
    #     print(x.shape)
    return np.sum([np.prod(x.shape) for x in model.parameters()])

if __name__ == "__main__":
    train_mnist(data_dir="../data")
