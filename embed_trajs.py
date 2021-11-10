import torch
from timeseries_clustering_vae.vrae.vrae import VRAE
from torch.utils.data import DataLoader, TensorDataset
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

kModelFile = 'vrae.pt'
kDataFile = 'data/processed/x_without_artifact.pt'

X = torch.load(kDataFile)
# Hyperparameters
hidden_size = 90
hidden_layer_depth = 1
latent_length = 20
batch_size = X.shape[0] # so that we get embeddings for all datapoints
learning_rate = 0.0005
n_epochs = 40
dropout_rate = 0.2
optimizer = 'Adam' # options: ADAM, SGD
cuda = True # options: True, False
print_every=30
clip = True # options: True, False
max_grad_norm=5
loss = 'MSELoss' # options: SmoothL1Loss, MSELoss
block = 'LSTM' # options: LSTM, GRU

# load data
X = torch.load(kDataFile)
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
z = vrae.transform(TensorDataset(X))
pca = PCA(n_components=2)

z_embedded = pca.fit_transform(z)
np.save("data/pca_z.npy", z_embedded)
