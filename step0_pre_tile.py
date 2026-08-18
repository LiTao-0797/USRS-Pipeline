# -*- coding: utf-8 -*-
```
Note, this is a demo code meant to generate MLS point cloud tiles (used for registration with UAV point clouds). 
However, in the EVO dataset (step1~step4), 
the tiles have already been prepared in advance by the dataset providers (including the "tiles.csv" file).
```

import os
import sys
import numpy as np
import open3d as o3d
from pathlib import Path
import csv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)
sys.path.append(BASE_DIR)

from utils.cloud_io import CloudIO

raw_ply_dir = os.path.join(BASE_DIR, 'raw_ply_files')
os.makedirs(raw_ply_dir, exist_ok=True)

tiled_ply_dir=os.path.join(BASE_DIR, 'tiled_ply_files')
os.makedirs(tiled_ply_dir, exist_ok=True)

class Config:
    input_cloud_path = os.path.join(raw_ply_dir, '5080_54435_new.ply')
    output_folder = tiled_ply_dir
    tile_size = 20
    # Set to None to enable automatic large coordinate detection
    offset = None
    #Typically, when coordinates reach 10**5 or 10**6 (e.g., standard UTM coordinates), 
    #the mantissa precision of float32 begins to degrade.

# Initial loading
cio = CloudIO(offset=None)
print(f"Reading point cloud: {Config.input_cloud_path}")

raw_cloud = cio.load_cloud(Config.input_cloud_path)


# Initialize grid
pos = raw_cloud.point["positions"].numpy()
x_min, y_min = np.min(pos[:, 0]), np.min(pos[:, 1])
x_max, y_max = np.max(pos[:, 0]), np.max(pos[:, 1])

num_cols = int(np.ceil((x_max - x_min) / Config.tile_size))
num_rows = int(np.ceil((y_max - y_min) / Config.tile_size))
print(f"Planned grid: {num_rows} rows x {num_cols} columns")

# Perform tiling
tile_info = []
large_z = 1e10

for r in range(num_rows):
    for c in range(num_cols):
        counter = r * num_cols + c    #Tile counter, starting from 0
        t_x_min = x_min + c * Config.tile_size    #Minimum X coordinate for the current tile
        t_y_min = y_min + r * Config.tile_size    #Minimum Y coordinate for the current tile
        
        # Cropping
        min_b = o3d.core.Tensor([t_x_min, t_y_min, -large_z], o3d.core.float32)
        #Minimum coordinates of the crop box
        max_b = o3d.core.Tensor([t_x_min + Config.tile_size, t_y_min + Config.tile_size, large_z], o3d.core.float32)
        #Maximum coordinates of the crop box
        crop_box = o3d.t.geometry.AxisAlignedBoundingBox(min_b, max_b)    #Construct the crop box
        tile_cloud = raw_cloud.crop(crop_box)    #Perform cropping
        
        # Record CSV information (record all tiles)
        tile_info.append([counter, r, c, t_x_min, t_y_min, Config.tile_size, Config.tile_size])
        
        # Save PLY only if there are points in the tile
        if len(tile_cloud.point.positions) > 0:
            save_path = os.path.join(Config.output_folder, f"tile_{counter}.ply")
            cio.save_cloud(tile_cloud, save_path, local_coordinates=False)
            print(f"保存: tile_{counter}.ply")

# Save CSV
csv_path = os.path.join(Config.output_folder, "tiles.csv")
with open(csv_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["#counter", "row", "col", "x_min", "y_min", "size_x", "size_y"])
    writer.writerows(tile_info)

print("\nTask completed!")
