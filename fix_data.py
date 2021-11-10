import torch
from torch.utils.data import DataLoader, TensorDataset

kDataFile = 'data/processed/x.pt'
kNewDataFile = 'data/processed/x_without_artifact.pt'
kProblematicIndex = 60 # this trajectory doesn't make any sense; the glider probably didn't go over England


X = torch.load(kDataFile)

# kind of messy, but this is a one time op so it's fine
new_X = torch.empty(X.shape[0] - 1, *X.shape[1:])
new_X[:kProblematicIndex] = X[:kProblematicIndex]
new_X[kProblematicIndex:] = X[kProblematicIndex+1:]
torch.save(new_X, kNewDataFile)
