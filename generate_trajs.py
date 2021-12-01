'''
Vary over each dimension and generate samples, showing what each latent dimension represents.
'''
import torch
from torch.utils.data.dataset import TensorDataset
from timeseries_clustering_vae.vrae.vrae import VRAE
import numpy as np
from constants import *

batch_size = 10 # so that we get embeddings for all datapoints

X = torch.load(kDataFile)
observed_Z = np.load(kRawEmbeddingsFile) # fit on the 72 input sequences

num_sequences, sequence_length, number_of_features = X.shape
dload = './model_dir' # where to store downloads
vrae = VRAE(sequence_length=sequence_length,
            number_of_features = number_of_features,
            hidden_size = hidden_size, 
            hidden_layer_depth = hidden_layer_depth,
            latent_length = latent_length,
            batch_size = batch_size,
            learning_rate = learning_rate,
            n_epochs = n_epochs,
            dropout_rate = dropout_rate,
            optimizer = optimizer, 
            cuda = cuda,
            print_every=print_every, 
            clip=clip, 
            max_grad_norm=max_grad_norm,
            loss = loss,
            block = block,
            dload = dload)
vrae.load(f'model_dir/{kModelFile}')
print("Observed z decoded:", vrae.decoder(torch.tensor(observed_Z)[:10]))

output = torch.zeros(latent_length, sequence_length, batch_size, 2)
z = torch.zeros(latent_length, batch_size, latent_length)
for j in range(latent_length):
    z[j,:,j] = torch.linspace(-5, 5, batch_size)
    output[j] = vrae.decoder(z[j])
output = output.swapaxes(1, 2)
np.save(kGenSeqFile, output.detach().numpy())

# empirical mean and stdevs to generate sequences from
obs_mean_Z = torch.Tensor(observed_Z.mean(axis=0, keepdims=True))
obs_std_Z = torch.Tensor(observed_Z.std(axis=0, keepdims=True))

# generate batch_size sequences drawn from normal fit on embeddings
'''
shape = (batch_size, latent_length)
mean = vrae.lmbd.latent_mean.repeat(batch_size, 1)
std = vrae.lmbd.latent_logvar.repeat(batch_size, 1).exp().sqrt()
print(mean, std)
start = torch.normal(mean=mean, std=std)
gen = vrae.decoder(start)
gen = gen.swapaxes(0, 1)
np.save(kRandomSeqFile, gen.detach().numpy())
'''

X = torch.load(kDataFile)
gen = vrae.reconstruct(TensorDataset(X))
print(gen)
np.save(kReconstructedSeqFile, gen)