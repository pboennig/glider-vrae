'''
Vary over each dimension and generate samples, showing what each latent dimension represents.
'''
import torch
from torch.utils.data.dataset import TensorDataset
import scale_data
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

def variation_sweep():
    output = torch.zeros(latent_length, sequence_length, batch_size, 2)
    obs_mean_Z = torch.Tensor(observed_Z.mean(axis=0))
    mu = obs_mean_Z.detach().numpy()
    z = obs_mean_Z.unsqueeze(1).unsqueeze(2).expand(-1, batch_size, latent_length).clone()
    for j in range(latent_length):
        z[j,:,j] = torch.linspace(mu[j] - kVariationSweep, mu[j] + kVariationSweep, batch_size)
        output[j] = vrae.decoder(z[j])
    output = output.swapaxes(1, 2)
    np.save(kGenSeqFile, scale_data.unscale_data(output.detach().numpy()))

# empirical mean and stdevs to generate sequences from
def sample():
    obs_mean_Z = torch.Tensor(observed_Z.mean(axis=0)).unsqueeze(1).repeat(1, batch_size)
    obs_std_Z = torch.Tensor(observed_Z.std(axis=0)).unsqueeze(1).repeat(1, batch_size)
    obs_std_Z = 5 * torch.ones((latent_length, batch_size))

    start = torch.normal(mean=obs_mean_Z, std=obs_std_Z).swapaxes(0, 1)
    gen = vrae.decoder(start)
    gen = gen.swapaxes(0, 1)
    np.save(kRandomSeqFile, scale_data.unscale_data(gen.detach().numpy()))

def recon():
    X = scale_data.scale_data(torch.load(kDataFile))
    gen = vrae.reconstruct(TensorDataset(X)).swapaxes(0,1)
    print(gen)
    np.save(kReconstructedSeqFile, scale_data.unscale_data(gen))

#sample()
#variation_sweep()
recon()