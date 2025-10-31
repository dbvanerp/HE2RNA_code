"""
HE2RNA: Divide whole-slide images in tiles and extract ResNet features
Copyright (C) 2020  Owkin Inc.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("Enabled memory growth for GPU.")
    except RuntimeError as e:
        print(e)

import os
import numpy as np
import pickle as pkl
import argparse
import openslide
import openslide.deepzoom
import colorcorrect
from joblib import Parallel, delayed
from PIL import Image
from colorcorrect.util import from_pil, to_pil
from colorcorrect import algorithm as cca
from tqdm import tqdm
from keras.applications.resnet50 import ResNet50
from keras.models import Model
from keras.applications.resnet50 import preprocess_input
import random
import uuid

def _l0_downsample_for_dz_level(zoom, dz_level):
    # Deep Zoom levels are power-of-two downsamples of level-0 dimensions
    dz_levels = zoom.level_count
    return 2 ** (dz_levels - dz_level - 1)

def _pick_dz_level_for_target_mpp(slide, zoom, target_mpp):
    # Use MPP X; for anisotropic pixels, this can be adapted as needed
    mpp_x = slide.properties.get(openslide.PROPERTY_NAME_MPP_X)
    if mpp_x is None:
        raise ValueError("Slide is missing MPP metadata; cannot target 0.5 mpp")
    mpp0 = float(mpp_x)

    best_level = 0
    best_err = float('inf')
    for dz_level in range(zoom.level_count):
        ds = _l0_downsample_for_dz_level(zoom, dz_level)
        mpp = mpp0 * ds
        err = abs(mpp - target_mpp)
        if err < best_err:
            best_err = err
            best_level = dz_level
    return best_level

def extract_tile_features(dz_level, coord, zoom, tile_size, clam_l0_mode=False):
    if clam_l0_mode:
        x_l0, y_l0 = int(coord[1]), int(coord[2])

        ds = _l0_downsample_for_dz_level(zoom, dz_level)

        x_z = x_l0 / ds
        y_z = y_l0 / ds

        tile_col = int(x_z // tile_size)
        tile_row = int(y_z // tile_size)

        # Clamp within valid address range
        max_cols, max_rows = zoom.level_tiles[dz_level]
        tile_col = max(0, min(tile_col, max_cols - 1))
        tile_row = max(0, min(tile_row, max_rows - 1))

        tile = np.array(zoom.get_tile(dz_level, (tile_col, tile_row)))
    else:
        tile = np.array(zoom.get_tile(dz_level, (coord[1], coord[2])))
    tile_img = Image.fromarray(tile)
    tile_img = to_pil(cca.stretch(from_pil(tile_img)))
    tile = np.array(tile_img)
    # Add a 1 in 100 chance to save an example tile image (for debug/visualization)
    if random.randint(1, 1000) == 1:
        example_dir = os.path.join(os.getcwd(), "tile_examples")
        os.makedirs(example_dir, exist_ok=True)
        # Use uuid4 to guarantee no collisions, save as PNG
        filename = f"tile_example_{uuid.uuid4().hex}.png"
        tile_img.save(os.path.join(example_dir, filename))
    return tile

def save_numpy_features(path2slides, folder, slidename, coords, path, tile_size, clam_l0_mode=False):
    model = ResNet50(weights='imagenet', include_top=True)
    model = Model(inputs=model.inputs, outputs=model.get_layer('avg_pool').output)

    slide = openslide.OpenSlide(os.path.join(path2slides, folder, slidename))
    zoom = openslide.deepzoom.DeepZoomGenerator(slide, tile_size=tile_size, overlap=0)

    # Choose Deep Zoom level that best matches 0.5 mpp
    target_mpp = 0.5
    if clam_l0_mode == False:
        dz_level = int(coords[0][0])
        print("Using provided dz_level", dz_level)
    else:
        dz_level = _pick_dz_level_for_target_mpp(slide, zoom, target_mpp)
        print("Calculated dz_level for target mpp:", dz_level)
    tiles = np.array([extract_tile_features(dz_level, coord, zoom, tile_size, clam_l0_mode=clam_l0_mode) for coord in tqdm(coords)])
    tiles = preprocess_input(tiles)
    X = model.predict(tiles, batch_size=32)
    X = np.concatenate([coords, X], axis=1)
    np.save(os.path.join(path, '0.50_mpp', slidename.split('.')[0] + '.npy'), X)

def process_all_slides(path2slides, tile_coords, path, tile_size, clam_l0_mode=False):

    subfolder = {}
    slide_dirs = [d for d in os.listdir(path2slides) if os.path.isdir(os.path.join(path2slides, d))]

    slidenames = []
    subfolders = []

    for d in slide_dirs:
        for f in os.listdir(os.path.join(path2slides, d)):
            if f.endswith('.svs') or f.endswith('.tif') and 'mask' not in f:
                slidenames.append(f)
                subfolders.append(d)

    if not os.path.exists(path):
        os.mkdir(path)
    if not os.path.exists(os.path.join(path, '0.50_mpp')):
        os.mkdir(os.path.join(path, '0.50_mpp'))

    for folder, slidename in zip(subfolders, slidenames):
        if slidename in tile_coords.keys():
            save_numpy_features(path2slides, folder, slidename, tile_coords[slidename], path, tile_size, clam_l0_mode=clam_l0_mode)
        else:
            print(f'Warning: tile coordinates not found for file {slidename}, skipping it')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_to_slides", help="path to folder containing subfolders with whole-slide images",
                        default='data/PESO/')
    parser.add_argument("--path_to_save_features", help="path to save features as npy files",
                        default='data/PESO_tiles')
    parser.add_argument("--tile_coordinates", help="path to pkl file containing tile coordinates",
                        default='tile_coordinates/tile_coordinates_PESO.pkl')
    parser.add_argument("--tile_size", help="size of the tile",
                        default=224)
    parser.add_argument("--clam_l0_mode", help="use clam_l0_mode",
                        default=False)
    args = parser.parse_args()
    path2slides = args.path_to_slides
    path = args.path_to_save_features
    tile_coords = args.tile_coordinates
    tile_size = int(args.tile_size)
    clam_l0_mode = args.clam_l0_mode
    clam_l0_mode = True if clam_l0_mode == 'True' else False
    process_all_slides(path2slides, pkl.load(open(tile_coords, 'rb')), path, tile_size, clam_l0_mode)
    
if __name__ == '__main__':

     main()