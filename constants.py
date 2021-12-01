# Hyperparameters
hidden_size = 200
hidden_layer_depth = 3
latent_length = 50
batch_size = 32
learning_rate = 0.01
n_epochs = 40
dropout_rate = 0.2
optimizer = 'Adam' # options: ADAM, SGD
cuda = True # options: True, False
print_every=30
clip = True # options: True, False
max_grad_norm=5
loss = 'SmoothL1Loss' # options: SmoothL1Loss, MSELoss
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
kReconstructedSeqFile = 'data/reconstruct.npy'
