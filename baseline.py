import os
import torch
from sklearn.decomposition import PCA
from src.plotting import *

IN_DIR = './data/processed'

def main():
	filename = 'x_without_artifact.pt'
	filepath = os.path.join(IN_DIR, filename)

	num_examples = 72
	X = torch.load(filepath).numpy()
	X = X.reshape(num_examples, -1)
	print(X.shape)

	pca = PCA(n_components=2)
	X_new = pca.fit_transform(X)

	plot_z(X_new, './plots/baseline_pca.png')
	plot_z_highlight_i(X_new, './plots/baseline_pca_49_highlighted.png', i=49)


if __name__ == '__main__':
	main()