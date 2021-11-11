import torch
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np

kDataFile = '../data/processed/x_without_artifact.pt'
kEmbeddingsFile = '../data/pca_z.npy'
kBounding = .25 # how much to bound around trajectory
A = torch.load(kDataFile).numpy()
z_embedded = np.load(kEmbeddingsFile)

def plot_trajectories(A):
    """
    Plot trajectories on one map.
    traj_array: (num_traj, len_traj, 2), assume that 2nd dimension is (lon, lat)
    """
    lon = A[:,:,0]
    lat = A[:,:,1]

    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([np.min(lat)-kBounding, np.max(lat)+kBounding, np.min(lon)-kBounding, np.max(lon)+kBounding], crs=ccrs.PlateCarree())
    ax.coastlines(resolution='10m')
    ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)

    for i in range(A.shape[0]):
        plt.plot(A[i,:,1], A[i,:,0])
    plt.savefig('../plots/traj_map.png', dpi=200)
    plt.clf()

def trajectory_grid(A):
    assert(A.shape[0] == 72)
    fig, axs = plt.subplots(8, 9)
    fig.set_size_inches(18.5, 10.5)
    fig.suptitle("All 72 trajectories, indpendent of map")
    for i in range(8):
        for j in range(9):
            idx = i*9 + j
            axs[i, j].plot(A[idx,:,1], A[idx,:,0])
            axs[i, j].set_xticks([])
            axs[i,j].set_yticks([])
    plt.savefig("../plots/traj_grid.png")
    plt.clf()
    fig.set_size_inches(8, 6)


def plot_z(z_embedded):
    plt.scatter(x=z_embedded[:,0], y=z_embedded[:,1])
    plt.title("VRAE latent embeddings in PCA space")
    plt.xlabel("PC 1")
    plt.ylabel("PC 2")
    plt.savefig("../plots/vrae_pca.png")
    plt.clf()

def plot_z_highlight_i(z_embedded, i=49):
    plt.scatter(x=z_embedded[:,0], y=z_embedded[:,1], marker='o', color='grey')
    plt.scatter(x=z_embedded[i,0], y=z_embedded[i,1], marker='o', color='red')
    plt.xlabel("PC 1")
    plt.ylabel("PC 2")
    plt.savefig("../plots/vrae_pca_49_highlighted.png", dpi=200)
    plt.clf()

def plot_traj_highlight_i(A, i=49):
    lon = A[:,:,0]
    lat = A[:,:,1]

    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([np.min(lat)-kBounding, np.max(lat)+kBounding, np.min(lon)-kBounding, np.max(lon)+kBounding], crs=ccrs.PlateCarree())
    ax.coastlines(resolution='10m')
    ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)

    for j in range(A.shape[0]):
        if j != i:
            plt.plot(A[j,:,1], A[j,:,0], color='grey')
    
    plt.plot(A[i,:,1], A[i,:,0], color='red')
    plt.savefig("../plots/vrae_traj_49_highlighted.png", dpi=200)
    plt.clf()

plot_trajectories(A) 
trajectory_grid(A)
plot_z(z_embedded)
plot_z_highlight_i(z_embedded)
plot_traj_highlight_i(A)
