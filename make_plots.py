from src.plotting import *
import numpy as np
import torch

A = torch.load(kDataFile).numpy()
raw_z = np.load(kRawEmbeddingsFile)
z_embedded = np.load(kEmbeddingsFile)
gen_seq = np.load(kGenSeqFile)
rand_seq = np.load(kRandomSeqFile)
recon_seq = np.load(kReconstructedSeqFile)
training_traj = np.load(kTrainingTraj)

'''
plot_trajectories(A, 'plots/traj_map.png') 
trajectory_grid(A, 'plots/traj_grid.png')
'''
plot_z(z_embedded, 'plots/vrae_pca.png')
#plot_traj_variation(gen_seq, 'plots/var_dim', np.linspace(-5, 5, num=10))
plot_trajectories(rand_seq, 'plots/ensemble.png', title=f'{rand_seq.shape[0]} randomly generated trajectories')
plot_trajectories(recon_seq, 'plots/recon_ensemble.png', title=f'Reconstructed original sequences')
plot_training_traj(training_traj, 'plots/loss_trajectory.png')
'''
for i in range(A.shape[0]):
    plot_z_highlight_i(z_embedded, f'plots/paired_pca_traj/{i}_pca_highlighted.png', i=i)
    plot_traj_highlight_i(A, f'plots/paired_pca_traj/{i}_traj_highlighted.png', i=i)
'''
