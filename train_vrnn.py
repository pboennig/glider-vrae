import torch
import torch.nn as nn
import torch.utils
import torch.utils.data
from constants import *
from torch.autograd import Variable
import matplotlib.pyplot as plt 
from VariationalRecurrentNeuralNetwork.model import VRNN
from torch.utils.data import DataLoader, TensorDataset

"""implementation of the Variational Recurrent
Neural Network (VRNN) from https://arxiv.org/abs/1506.02216
using unimodal isotropic gaussian distributions for 
inference, prior, and generating models."""


def train(epoch):
    train_loss = 0
    for batch_idx, data in enumerate(train_loader):
        
        #transforming data
        #data = Variable(data)
        #to remove eventually
        data = data[0]
        data = Variable(data.squeeze().transpose(0, 1))
        data = (data - data.min().data) / (data.max().data - data.min().data)
        
        #forward + backward + optimize
        optimizer.zero_grad()
        kld_loss, nll_loss, _, _ = model(data)
        loss = kld_loss + nll_loss
        loss.backward()
        optimizer.step()

        #grad norm clipping, only in pytorch version >= 1.10
        nn.utils.clip_grad_norm(model.parameters(), clip)

        #printing
        if batch_idx % print_every == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\t KLD Loss: {:.6f} \t NLL Loss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                kld_loss.data / batch_size,
                nll_loss.data / batch_size))

            sample = model.sample(28)
            print(sample)

        train_loss += loss.data


    print('====> Epoch: {} Average loss: {:.4f}'.format(
        epoch, train_loss / len(train_loader.dataset)))


#manual seed
torch.manual_seed(seed)
plt.ion()

#init model + optimizer + datasets
X = torch.load(kDataFile)
train_loader = DataLoader(
    TensorDataset(X),
    batch_size=batch_size, shuffle=True)


model = VRNN(x_dim, h_dim, z_dim, n_layers)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(1, n_epochs + 1):
    
    #training + testing
    train(epoch)
    # test(epoch)

    #saving model
    if epoch % save_every == 1:
        fn = 'model_dir/vrnn_state_dict_'+str(epoch)+'.pth'
        torch.save(model.state_dict(), fn)
        print('Saved model to '+fn)