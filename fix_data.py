'''
Clean up data by removing problematic index.
'''
import torch
from torch.utils.data import DataLoader, TensorDataset
from constants import *

kProblematicIndex = 60 # this trajectory doesn't make any sense; the glider probably didn't go over England


X = torch.load(kRawDataFile)

# kind of messy, but this is a one time op so it's fine
new_X = torch.empty(X.shape[0] - 1, *X.shape[1:])
new_X[:kProblematicIndex] = X[:kProblematicIndex]
new_X[kProblematicIndex:] = X[kProblematicIndex+1:]
x_split = torch.split(new_X, 10, dim=1)
x_split = torch.cat(x_split, dim=0)
torch.save(new_X[:,::10], kDataFile)
