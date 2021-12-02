'''
Load processed data, fit vrae, and save output.
'''
# Follows https://github.com/tejaslodaya/timeseries-clustering-vae/blob/7454fb68662680473fc242014dec7a291b7a4c7a/Timeseries_clustering.ipynb
from timeseries_clustering_vae.vrae.vrae import VRAE
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from constants import *
import scale_data


# load data
X = scale_data.scale_data(torch.load(kDataFile))
print(X.shape)
num_sequences, sequence_length, number_of_features = X.shape

# Initialize model
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

training_traj = vrae.fit(TensorDataset(X))
np.save(kTrainingTraj, np.array(training_traj))
vrae.save(kModelFile)
