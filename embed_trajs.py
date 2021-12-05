'''
Load model, embed data into latent z space, and save 2-D representation after PCA.
'''
import torch
from timeseries_clustering_vae.vrae.vrae import VRAE
from torch.utils.data import DataLoader, TensorDataset
from sklearn  import preprocessing
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
from constants import *
import scale_data

X = torch.load(kDataFile)
batch_size = X.shape[0] # so that we get embeddings for all datapoints, overwrite constant in constants.py

# load data
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

vrae.load(f'model_dir/{kModelFile}')
z = vrae.transform(TensorDataset(scale_data.scale_data(X)))
np.save(kRawEmbeddingsFile, z)

pca = PCA(n_components=2)
z_embedded = pca.fit_transform(z)
print(z_embedded)
np.save(kEmbeddingsFile, z_embedded)
