import argparse
import pandas as pd
from shapely.geometry import Point, Polygon
import geopandas as gpd
from geopandas import GeoSeries
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

kBounding = .25 # how much to bound around trajectory


parser = argparse.ArgumentParser(description='Plot a trajectory stored in a CSV.')
parser.add_argument('file', type=argparse.FileType('r'))
args = parser.parse_args()


df = pd.read_csv(args.file)[1:].drop_duplicates(subset=['longitude','latitude'])
lon = pd.to_numeric(df["longitude"])
lat = pd.to_numeric(df["latitude"])
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_extent([min(lon)-kBounding, max(lon)+kBounding, min(lat)-kBounding, max(lat)+kBounding], crs=ccrs.PlateCarree())
ax.coastlines(resolution='10m')
ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)

plt.plot(lon, lat)
plt.show()
