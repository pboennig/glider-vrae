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
    print(f"plotting trajectory map for {fn}...", end="")
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
    print("done!")


def plot_comparison(A, fn):
    print(A.shape)
    print(f"plotting orig/autoreg/vrae comp for {fn}...", end="")
    lon = A[:2,:,0]
    lat = A[:2,:,1]

    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([np.min(lat)-kBounding, np.max(lat)+kBounding, np.min(lon)-kBounding, np.max(lon)+kBounding], crs=ccrs.PlateCarree())
    ax.coastlines(resolution='10m')
    ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
    
    plt.plot(A[0,:,1], A[0,:,0], color='r', label='original')
    plt.plot(A[1,:,1], A[1,:,0], color='b', label='autoregressive')
    plt.title("Autoregressive model gives reasonable sample...")
    plt.legend(loc="upper left")
    plt.savefig(fn, dpi=200)
    print("done!")
    plt.clf()


def trajectory_grid(A, fn):
    """
    Plot all 72 trajectories in a grid to show different shapes. The map is not drawn to make it cleaner.
    """
    print("plotting traj grid...", end="")
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
    print("done!")


def plot_z(z_embedded, fn):
    """
    Given the 2-D PCA latent space embeddings, plot a scatter plot.
    """
    print("plotting pca scatter...", end="")
    plt.scatter(x=z_embedded[:,0], y=z_embedded[:,1])
    plt.title("VRAE latent embeddings in PCA space")
    plt.xlabel("PC 1")
    plt.ylabel("PC 2")
    plt.savefig(fn)
    plt.clf()
    print("done!")

def plot_z_highlight_i(z_embedded, fn, i=49):
    """
    Highlight one z in PCA scatter plot.
    """
    print(f"plotting pca scatter with highlight i = {i}...", end="")
    plt.scatter(x=z_embedded[:,0], y=z_embedded[:,1], marker='o', color='grey')
    plt.scatter(x=z_embedded[i,0], y=z_embedded[i,1], marker='o', color='red')
    plt.xlabel("PC 1")
    plt.ylabel("PC 2")
    plt.savefig(fn, dpi=200)
    plt.clf()
    print("done!")

def plot_traj_highlight_i(A, fn, i=49):
    """
    Plot all trajectories on map and highlight i.
    """
    print(f"plotting trajectories with highlight i ={i}...", end="")
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
    print("done!")

def plot_traj_variation(X, dir, raw_z):
    """
    X: (latent_dim, num_seq, seq_length)
    values: the different values of each latent_dim, typically a linspace array.

    For each latent_dim, create a plot showing how varying that latent variable changes the generated sequence, coloring by the value.
    """
    print("plotting variation over latent dims...",end="")
    vals = raw_z.mean(axis=0)
    for j in range(X.shape[0]):
        A = X[j]
        lon = A[:,:,0]
        lat = A[:,:,1]

        ax = plt.axes(projection=ccrs.PlateCarree())
        ax.set_extent([np.min(lat)-kBounding, np.max(lat)+kBounding, np.min(lon)-kBounding, np.max(lon)+kBounding], crs=ccrs.PlateCarree())
        ax.coastlines(resolution='10m')
        ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)

        values = np.linspace(vals[j] - kVariationSweep, vals[j] + kVariationSweep, 10)
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
    print("done!")

def box_embed(Z, dir):
    for j in range(Z.shape[1]):
        plt.title(f'Values of dimension {j}')
        plt.boxplot(Z[:,j])
        plt.savefig(f'{dir}/{j}.png', dpi=200)
        plt.clf()

def plot_training_traj(traj, fname, recon_color='xkcd:coral', kl_color='xkcd:olive'):
    print("Plotting loss versus epoch...", end="")
    recon = traj[:,0]
    kl = traj[:,1]
    x = np.arange(1, kl.shape[0]+1, 10)
    fig, ax1 = plt.subplots()
    ax1.set_ylabel('reconstruction loss', color=recon_color)
    ax1.set_yscale('log')
    ax1.set_xlabel('epoch')
    ax1.plot(recon, color=recon_color)
    ax1.tick_params(axis='y', labelcolor=recon_color)
    
    ax2 = ax1.twinx()
    ax2.set_ylabel('KL divergence', color=kl_color)
    ax2.plot(kl, color=kl_color)
    ax2.tick_params(axis='y', labelcolor=kl_color)

    plt.savefig(fname, dpi=200)
    plt.clf()
    print("done!")
