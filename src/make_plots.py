import torch
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np

kDataFile = '../data/processed/x.pt'
kBounding = .25 # how much to bound around trajectory
X = torch.load(kDataFile)

def plot_trajectories(traj_tensor):
    """
    Plot trajectories on a map.
    traj_array: (num_traj, len_traj, 2), assume that 2nd dimension is (lon, lat)
    """
    A = np.delete(traj_tensor.numpy(), 60, axis=0) # the 60th trajectory is junk data
    lon = A[:,:,0]
    lat = A[:,:,1]
    print(np.mean(lon), np.mean(lat))

    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([np.min(lat)-kBounding, np.max(lat)+kBounding, np.min(lon)-kBounding, np.max(lon)+kBounding], crs=ccrs.PlateCarree())
    ax.coastlines(resolution='10m')
    ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)

    for i in range(A.shape[0]):
        plt.plot(A[i,:,1], A[i,:,0])
    plt.title("Trajectory dataset")
    plt.show()

plot_trajectories(X)
