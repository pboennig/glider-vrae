import argparse
import pandas as pd
from shapely.geometry import Point, Polygon
import geopandas as gpd
from geopandas import GeoSeries
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np

def plot_trajectory(trajectory):
    ax.set_extent([min(lon)-kBounding, max(lon)+kBounding, min(lat)-kBounding, max(lat)+kBounding], crs=ccrs.PlateCarree())
    ax.coastlines(resolution='10m')
    ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)

    plt.plot(lon, lat)
    plt.show()

