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
auto_reg = np.load(kAutoRegSeq)

plot_trajectories(A, 'plots/traj_map.png', title="Original 72 sequences") 
trajectory_grid(A, 'plots/traj_grid.png')
plot_z(z_embedded, 'plots/vrae_pca.png')
plot_z_highlight_i(z_embedded, f'plots/paired_pca_traj/49_pca_highlighted.png')
plot_traj_highlight_i(A, f'plots/paired_pca_traj/49_traj_highlighted.png')
#plot_traj_variation(gen_seq, 'plots/var_dim', np.linspace(-5, 5, num=10))
plot_trajectories(A[[31]], 'plots/31_orig.png') 
plot_trajectories(auto_reg[np.newaxis], 'plots/31_autoreg.png')
plot_trajectories(recon_seq[[31]], 'plots/31_vrae.png')
print(A[31].shape, auto_reg.shape, recon_seq[31].shape)
plot_comparison(np.stack([A[31], auto_reg]), 'plots/comp.png')
plot_trajectories(recon_seq[[31]], 'plots/31_vrae.png', title="...but VRAE doesn't")

plot_trajectories(rand_seq, 'plots/ensemble.png', title=f'{rand_seq.shape[0]} randomly generated trajectories')
plot_trajectories(recon_seq, 'plots/recon_ensemble.png', title=f'Reconstructed original sequences')
plot_training_traj(training_traj, 'plots/loss_trajectory.png')
