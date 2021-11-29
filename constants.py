# Hyperparameters
hidden_size = 90
hidden_layer_depth = 1
latent_length = 20
batch_size = 32
learning_rate = 0.0005
n_epochs = 400
dropout_rate = 0.2
optimizer = 'Adam' # options: ADAM, SGD
cuda = True # options: True, False
print_every=30
clip = True # options: True, False
max_grad_norm=5
loss = 'MSELoss' # options: SmoothL1Loss, MSELoss
block = 'LSTM' # options: LSTM, GRU

# Misc constants
kBounding = .25 # how much to bound around trajectory

# Filenames w.r.t root
kRawDataFile = 'data/processed/x.pt'
kDataFile = 'data/processed/x_without_artifact.pt'
kModelFile = 'vrae.pt'
kDataFile = 'data/processed/x_without_artifact.pt'
kEmbeddingsFile = 'data/pca_z.npy'
kGenSeqFile = 'data/variation_of_dims.npy'
kRandomSeqFile = 'data/random_sequences.npy'
kRawEmbeddingsFile = 'data/raw_z.npy'
