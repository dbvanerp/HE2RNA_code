"""
HE2RNA: Apply super-tile preprocessing to ResNet features of tiles extracted from whole-slide images
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

import os
import argparse
import numpy as np
import h5py
import matplotlib.pyplot as plt
from tqdm import tqdm
from kmeans_pytorch import kmeans
import torch
import pandas as pd
from TCGADatasetClass import TCGADataset

def parse_project_filter(project_filter_arg):
    """Parse project filter from comma-separated list or single-column CSV."""
    if project_filter_arg is None or project_filter_arg == "":
        return None

    if os.path.exists(project_filter_arg):
        project_df = pd.read_csv(project_filter_arg)
        if project_df.shape[1] == 0:
            raise ValueError(f"Empty project filter file: {project_filter_arg}")
        return project_df.iloc[:, 0].dropna().astype(str).tolist()

    return [p.strip() for p in project_filter_arg.split(",") if p.strip()]


def plot_cluster(slide_name, coords, cluster_ids, centroids, plot_dir):
    plt.figure(figsize=(4, 3), dpi=160)
    plt.scatter(coords[:, 0], coords[:, 1], c=cluster_ids, cmap='cool', s=8)
    plt.scatter(centroids[:, 0], centroids[:, 1], c='white', alpha=0.6, edgecolors='black', linewidths=2, s=24)

    all_x = np.concatenate([coords[:, 0], centroids[:, 0]])
    all_y = np.concatenate([coords[:, 1], centroids[:, 1]])
    x_pad = (all_x.max() - all_x.min()) * 0.05 if all_x.max() > all_x.min() else 1.0
    y_pad = (all_y.max() - all_y.min()) * 0.05 if all_y.max() > all_y.min() else 1.0
    plt.xlim(all_x.min() - x_pad, all_x.max() + x_pad)
    plt.ylim(all_y.min() - y_pad, all_y.max() + y_pad)
    plt.gca().invert_yaxis()

    plt.title(
        f"Kmeans clustering results for {slide_name} (n_clusters={len(np.unique(cluster_ids))})",
        fontsize=8,
        pad=10,
        loc='center'
    )
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    os.makedirs(plot_dir, exist_ok=True)
    output_path = os.path.join(plot_dir, f"cluster_plot_{slide_name}.png")
    plt.savefig(output_path)
    print(f"Saved cluster plot to {output_path}")
    plt.close()


def cluster_dataset(
    dataset,
    n_tiles=100,
    path_to_data='data/TCGA_slic_100.h5',
    input_tiles=8000,
    max_slides=None,
    enable_plot=False,
    plot_every=200,
    plot_dir='/home/dvanerp/temp_clusterplots_uni',
):
    """Perform KMeans on each tiles to create 'supertiles'. Supertile
    features are obtained by averaging resnet features.

    Args
        dataset (TCGADataset): dataset used to align and index sample files
        n_tiles (int): number of supertiles to generate
        path_to_data (str): path to hdf5 file to save the clustered dataset
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device for KMeans: {device}")

    sample_ids = list(dataset.sample_ids)
    if max_slides is not None:
        sample_ids = sample_ids[:max_slides]
    if len(sample_ids) == 0:
        raise ValueError("No aligned samples found in dataset.")

    first_sample = sample_ids[0]
    first_data = np.load(dataset.feature_files[first_sample], mmap_mode='r')
    if first_data.ndim != 2:
        raise ValueError(f"Expected 2D array for sample {first_sample}, got shape {first_data.shape}")
    tile_width = first_data.shape[1]
    slide_name_len = max(len(sid) for sid in sample_ids)

    with h5py.File(path_to_data, 'w') as file:
        file.create_dataset('X', (len(sample_ids), n_tiles, tile_width), dtype=np.float32)
        file.create_dataset('cluster_attribution', (len(sample_ids), input_tiles), dtype=np.int64)
        dt = h5py.string_dtype(encoding='utf-8', length=slide_name_len)
        file.create_dataset('slide_name', (len(sample_ids),), dtype=dt)

        for n, sample_id in enumerate(tqdm(sample_ids)):
            try:
                slide_name = sample_id
                file['slide_name'][n] = slide_name
                x = np.load(dataset.feature_files[sample_id])

                if x.ndim != 2:
                    raise ValueError(f"Expected 2D matrix, got shape {x.shape}")
                if x.shape[1] != tile_width:
                    raise ValueError(
                        f"Feature width mismatch for {sample_id}: expected {tile_width}, got {x.shape[1]}"
                    )
                if x.shape[1] < 4:
                    raise ValueError(f"Expected at least 4 columns [zoom, x, y, features...], got {x.shape[1]}")

                # Remove padded rows: keep tiles with non-zero coords or non-zero feature vectors.
                has_valid_coords = np.any(x[:, 1:3] != 0, axis=1)
                has_valid_features = np.any(x[:, 3:] != 0, axis=1)
                valid_mask = has_valid_coords | has_valid_features
                x_valid = x[valid_mask]

                if x_valid.shape[0] == 0:
                    raise ValueError("No valid tiles found after removing padding.")

                zoom_value = x_valid[0, 0]
                coords = torch.from_numpy(x_valid[:, 1:3].astype(np.float32)).to(device)
                vals = torch.from_numpy(x_valid[:, 3:].astype(np.float32)).to(device)

                cluster_ids, centroids = kmeans(
                    X=coords,
                    num_clusters=min(n_tiles, coords.shape[0]),
                    distance='euclidean',
                    tol=1e-4,
                    iter_limit=1000,
                    device=device,
                )

                if enable_plot and np.random.randint(0, max(1, plot_every)) == 0:
                    plot_cluster(
                        slide_name=slide_name,
                        coords=coords.cpu().numpy(),
                        cluster_ids=cluster_ids.cpu().numpy(),
                        centroids=centroids.cpu().numpy(),
                        plot_dir=plot_dir,
                    )

                new_coords = []
                new_vals = []
                for cl in torch.unique(cluster_ids):
                    mask_cl = cluster_ids == cl
                    new_coords.append(coords[mask_cl].mean(dim=0).cpu().numpy())
                    new_vals.append(vals[mask_cl].mean(dim=0).cpu().numpy())

                num_clusters = len(new_coords)
                zoom_col = np.full((num_clusters, 1), zoom_value, dtype=np.float32)
                x_clustered = np.concatenate([zoom_col, new_coords, new_vals], axis=1).astype(np.float32)

                if x_clustered.shape[0] < n_tiles:
                    padding = np.zeros((n_tiles - x_clustered.shape[0], x_clustered.shape[1]), dtype=np.float32)
                    x_clustered = np.concatenate([x_clustered, padding], axis=0)
                elif x_clustered.shape[0] > n_tiles:
                    print(f"Warning: {sample_id} has {x_clustered.shape[0]} clusters, but only {n_tiles} are needed")
                    x_clustered = x_clustered[:n_tiles]

                file['X'][n] = x_clustered

                cluster_ids_np = cluster_ids.cpu().numpy()
                cluster_attr = np.zeros(input_tiles, dtype=np.int64)
                num_valid_tiles = min(cluster_ids_np.shape[0], input_tiles)
                cluster_attr[:num_valid_tiles] = cluster_ids_np[:num_valid_tiles]
                file['cluster_attribution'][n] = cluster_attr

            except Exception as e:
                raise RuntimeError(
                    f"Failed while processing sample '{sample_id}' "
                    f"from '{dataset.feature_files.get(sample_id, 'UNKNOWN_PATH')}': {e}"
                ) from e
        
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path_to_transcriptome",
        help="Path to transcriptome CSV (used as targets_csv for sample alignment).",
        default='/home/dvanerp/pepsi/data/raw/tcga_rna/tcga_all_transcriptomes_tpm_10782.csv'
    )
    parser.add_argument(
        "--keyfile_path",
        help="Path to keyfile CSV used by TCGADataset.",
        default='/home/dvanerp/pepsi/data/raw/manifests/new_keyfiles/new_master_keyfile.csv'
    )
    parser.add_argument(
        "--features_dir",
        help="Path to raw UNI2 TCGA .npy features.",
        default='/home/dvanerp/pepsi/data/processed/UNI2-TCGA-data/TCGA_subsampled_8000'
    )
    parser.add_argument(
        "--project_filter",
        help="Optional comma-separated project list or path to single-column CSV.",
        default=None
    )
    parser.add_argument(
        "--path_to_save_processed_data",
        help="Path where supertile-preprocessed data should be saved.",
        default='data/TCGA_slic_100.h5'
    )
    parser.add_argument("--n_tiles", help="Number of supertiles", default=100, type=int)
    parser.add_argument("--input_tiles", help="Input tile count used for cluster_attribution width", default=8000, type=int)
    parser.add_argument("--max_slides", help="Optional max number of slides to process (debug/sanity runs).", default=None, type=int)
    parser.add_argument("--plot_clusters", help="Enable occasional cluster plots.", action='store_true')
    parser.add_argument("--plot_every", help="Plot one out of every N slides when plotting is enabled.", default=200, type=int)
    parser.add_argument("--plot_dir", help="Directory to store cluster plots.", default='/home/dvanerp/temp_clusterplots_uni')
    args = parser.parse_args()
    output_dir = os.path.dirname(args.path_to_save_processed_data)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    print(f"Saving processed data to {args.path_to_save_processed_data}")
    project_filter = parse_project_filter(args.project_filter)

    # I do not care about the genes here, so I set them to just one for fast loading
    dataset = TCGADataset(
        features_dir=args.features_dir,
        targets_csv=args.path_to_transcriptome,
        keyfile_path=args.keyfile_path,
        genes="ENSG00000000003.15",
        max_tiles=args.input_tiles,
        project_filter=project_filter,
    )
    cluster_dataset(
        dataset=dataset,
        n_tiles=args.n_tiles,
        path_to_data=args.path_to_save_processed_data,
        input_tiles=args.input_tiles,
        max_slides=args.max_slides,
        enable_plot=args.plot_clusters,
        plot_every=args.plot_every,
        plot_dir=args.plot_dir,
    )


if __name__ == '__main__':

    main()
