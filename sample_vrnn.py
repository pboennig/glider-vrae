import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from VariationalRecurrentNeuralNetwork.model import VRNN
from constants import *

state_dict = torch.load(f'saves/vrnn_state_dict_41.pth')
model = VRNN(x_dim, h_dim, z_dim, n_layers)
model.load_state_dict(state_dict)

sample = model.sample(x_dim*6)
print(sample.shape)
plt.plot(sample)
plt.show()