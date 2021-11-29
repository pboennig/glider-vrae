import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np
from matplotlib import cm
from matplotlib import colors
from constants import *

def plot_trajectories(A, fn, title = None):
    """
    Plot trajectories on one map.
    traj_array: (num_traj, len_traj, 2), assume that 2nd dimension is (lon, lat)
    """
    print("plotting trajectory map...")
    lon = A[:,:,0]
    lat = A[:,:,1]

    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([np.min(lat)-kBounding, np.max(lat)+kBounding, np.min(lon)-kBounding, np.max(lon)+kBounding], crs=ccrs.PlateCarree())
    ax.coastlines(resolution='10m')
    ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
    
    if title is not None:
       plt.title(title) 


    for i in range(A.shape[0]):
        plt.plot(A[i,:,1], A[i,:,0])
    plt.savefig(fn, dpi=200)
    plt.clf()

def trajectory_grid(A, fn):
    """
    Plot all 72 trajectories in a grid to show different shapes. The map is not drawn to make it cleaner.
    """
    print("plotting traj grid...")
    assert(A.shape[0] == 72)
    fig, axs = plt.subplots(8, 9)
    fig.set_size_inches(18.5, 10.5)
    fig.suptitle("All 72 trajectories, independent of map")
    for i in range(8):
        for j in range(9):
            idx = i*9 + j
            axs[i, j].plot(A[idx,:,1], A[idx,:,0])
            axs[i, j].set_xticks([])
            axs[i,j].set_yticks([])
    plt.savefig(fn)
    plt.clf()
    fig.set_size_inches(8, 6)


def plot_z(z_embedded, fn):
    """
    Given the 2-D PCA latent space embeddings, plot a scatter plot.
    """
    print("plotting pca scatter...")
    plt.scatter(x=z_embedded[:,0], y=z_embedded[:,1])
    plt.title("VRAE latent embeddings in PCA space")
    plt.xlabel("PC 1")
    plt.ylabel("PC 2")
    plt.savefig(fn)
    plt.clf()

def plot_z_highlight_i(z_embedded, fn, i=49):
    """
    Highlight one z in PCA scatter plot.
    """
    print("plotting pca scatter with highlight...")
    plt.scatter(x=z_embedded[:,0], y=z_embedded[:,1], marker='o', color='grey')
    plt.scatter(x=z_embedded[i,0], y=z_embedded[i,1], marker='o', color='red')
    plt.xlabel("PC 1")
    plt.ylabel("PC 2")
    plt.savefig(fn, dpi=200)
    plt.clf()

def plot_traj_highlight_i(A, fn, i=49):
    """
    Plot all trajectories on map and highlight i.
    """
    print("plotting trajectories with highlight...")
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
    plt.savefig(fn, dpi=200)
    plt.clf()

def plot_traj_variation(X, dir, values):
    """
    X: (latent_dim, num_seq, seq_length)
    values: the different values of each latent_dim, typically a linspace array.

    For each latent_dim, create a plot showing how varying that latent variable changes the generated sequence, coloring by the value.
    """
    print("plotting variation over latent dims...")
    for j in range(X.shape[0]):
        A = X[j]
        lon = A[:,:,0]
        lat = A[:,:,1]

        ax = plt.axes(projection=ccrs.PlateCarree())
        ax.set_extent([np.min(lat)-kBounding, np.max(lat)+kBounding, np.min(lon)-kBounding, np.max(lon)+kBounding], crs=ccrs.PlateCarree())
        ax.coastlines(resolution='10m')
        ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)

        norm = colors.Normalize(vmin=np.min(values), vmax=np.max(values))
        c_m = cm.cool

        # create a ScalarMappable and initialize a data structure
        s_m = cm.ScalarMappable(cmap=c_m, norm=norm)
        s_m.set_array([])

        for i in range(A.shape[0]):
            plt.plot(A[i,:,1], A[i,:,0], color=s_m.to_rgba(values[i]))
        plt.colorbar(s_m)
        plt.title(f'Variation in dimension {j}')
        plt.savefig(f'{dir}/{j}.png', dpi=200)
        plt.clf()

def box_embed(Z, dir):
    for j in range(Z.shape[1]):
        plt.title(f'Values of dimension {j}')
        plt.boxplot(Z[:,j])
        plt.savefig(f'{dir}/{j}.png', dpi=200)
        plt.clf()