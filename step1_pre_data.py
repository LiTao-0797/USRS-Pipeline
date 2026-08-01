# -*- coding: utf-8 -*-

import os
import sys
import numpy as np
import shutil

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from utils.data_prepare_utils import process_and_save_cloud


class Config:
    # Raw input paths
    dataset_root = os.path.join(BASE_DIR, "datasets\\evo_example_dataset")
    uav_cloud_path = os.path.join(dataset_root, "uav_cloud.ply")
    mls_tiles_folder = os.path.join(dataset_root, "tiles")
    tiles_csv_path = os.path.join(mls_tiles_folder, "tiles.csv")
    
    #  Output paths after preprocessing
    output_root = os.path.join(BASE_DIR, "evo_example_dataset_processed")
    output_uav_path = os.path.join(output_root, "uav_cloud.ply")
    output_tiles_folder = os.path.join(output_root, "tiles")
    output_tiles_csv_path = os.path.join(output_tiles_folder, "tiles.csv")
    
    # EVO-specific offset
    offset = np.array([-399000.0, -6786000.0, 0.0])
    
    # Verticality calculation parameters
    verticality_radius = 0.5


print("=" * 60)
print("Starting preprocessing for EVO dataset verticality calculation...")
print("=" * 60)

# Create output directory structure
os.makedirs(Config.output_root, exist_ok=True)
os.makedirs(Config.output_tiles_folder, exist_ok=True)

# Process UAV global reference point cloud
print("\n--- Processing UAV global reference cloud ---")
process_and_save_cloud(
    Config.uav_cloud_path, 
    Config.output_uav_path, 
    Config.offset, 
    Config.verticality_radius
)

#  Batch process MLS local tile point clouds
print("\n--- Processing MLS local tiles ---")
if not os.path.exists(Config.mls_tiles_folder):
    print(f"[Error] MLS tile folder not found: {Config.mls_tiles_folder}")
    sys.exit(1)
    
tile_files = [f for f in os.listdir(Config.mls_tiles_folder) if f.endswith(".ply") and f.startswith("tile")]
print(f"Found {len(tile_files)} tiles to process...")

for tile_file in sorted(tile_files):
    tile_input = os.path.join(Config.mls_tiles_folder, tile_file)
    tile_output = os.path.join(Config.output_tiles_folder, tile_file)
    process_and_save_cloud(
        tile_input, 
        tile_output, 
        Config.offset, 
        Config.verticality_radius
    )
    
# 3. Fully copy tiles.csv to ensure the input data structure for downstream scripts (e.g., step3) is complete
print("\n--- Synchronizing structural metadata ---")
if os.path.exists(Config.tiles_csv_path):
    print(f"Copying key tile configuration file: {os.path.basename(Config.tiles_csv_path)}")
    shutil.copyfile(Config.tiles_csv_path, Config.output_tiles_csv_path)
    print("[Synchronization complete]")
else:
    print(f"[Warning] {os.path.basename(Config.tiles_csv_path)} not found in the source path.")
    
print("\n" + "=" * 60)
print("EVO verticality preprocessing task completed!")
print(f"Final results saved to: {Config.output_root}")
print("=" * 60)