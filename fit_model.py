# Follows https://github.com/tejaslodaya/timeseries-clustering-vae/blob/7454fb68662680473fc242014dec7a291b7a4c7a/Timeseries_clustering.ipynb
from timeseries_clustering_vae.vrae.vrae import VRAE
import torch
from torch.utils.data import DataLoader, TensorDataset

kDataFile = 'data/processed/x.pt'
kModelFile = 'vrae.pt'

# Hyperparameters
hidden_size = 90
hidden_layer_depth = 1
latent_length = 20
batch_size = 32
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

vrae.fit(TensorDataset(X))
vrae.save(kModelFile)
