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
from torch.utils.data import DataLoader
from tqdm import tqdm
# from libKMCUDA import kmeans_cuda
from kmeans_pytorch import kmeans
from wsi_data import TCGAFolder, ToTensor
from transcriptome_data import TranscriptomeDataset
import torch
import uuid

def plot_cluster(slide_name, coords, cluster_ids, centroids):
    # plot
    def _to_numpy(arr):
        if isinstance(arr, torch.Tensor):
            return arr.detach().cpu().numpy()
        return arr

    coords = _to_numpy(coords)
    cluster_ids = _to_numpy(cluster_ids)
    centroids = _to_numpy(centroids)

    plt.figure(figsize=(4, 3), dpi=160)
    # Make scatter points smaller by setting s
    plt.scatter(coords[:, 0], coords[:, 1], c=cluster_ids, cmap='cool', s=8)
    plt.scatter(centroids[:, 0], centroids[:, 1], c='white', alpha=0.6, edgecolors='black', linewidths=2, s=24)

    # Dynamically set axis limits to contain all data points with some padding
    all_x = np.concatenate([coords[:, 0], centroids[:, 0]])
    all_y = np.concatenate([coords[:, 1], centroids[:, 1]])
    x_pad = (all_x.max() - all_x.min()) * 0.05 if all_x.max() > all_x.min() else 1.0
    y_pad = (all_y.max() - all_y.min()) * 0.05 if all_y.max() > all_y.min() else 1.0
    plt.xlim(all_x.min() - x_pad, all_x.max() + x_pad)
    plt.ylim(all_y.min() - y_pad, all_y.max() + y_pad)
    plt.gca().invert_yaxis()

    # Reduce title font size and adjust its position to prevent it from falling out of bounds
    plt.title(
        f"Kmeans clustering results for {slide_name} (n_clusters={len(np.unique(cluster_ids))})",
        fontsize=8,
        pad=10,
        loc='center'
    )
    plt.tight_layout(rect=[0, 0, 1, 0.93])  # leave a bit more space above the plot for the title
    plt.savefig(f"/home/dvanerp/temp_clusterplots/cluster_plot_{slide_name}.png")
    print(f"Saved cluster plot to /home/dvanerp/temp_clusterplots/cluster_plot_{slide_name}.png")
    plt.close()


def cluster_dataset(dataset, n_tiles=100,
                    path_to_data='data/TCGA_slic_100.h5'):
    """Perform KMeans on each tiles to create 'supertiles'. Supertile
    features are obtained by averaging resnet features.

    Args
        dataset (torch.utils.data.Dataset)
        n_tiles (int): number of supertiles to generate
        path_to_data (str): path to hdf5 file to save the clustered dataset
    """

    file = h5py.File(path_to_data, 'w')
    file.create_dataset('X', (len(dataset), n_tiles, 2051))
    file.create_dataset('cluster_attribution', (len(dataset), 8000))

    dataloader = DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=16,)

    n = 0


    for x, y in tqdm(dataloader):
        # x comes as a tuple from DataLoader; first element is your array

        path, _ = dataset.samples[n]
        slide_filename = os.path.basename(path)
        slide_name = os.path.splitext(slide_filename)[0]

        x = x[0].numpy().T

        # Remove padding
        mask = x[:, 0] > 0
        x = x[mask]
        zoom_value = x[0, 0]

        # Split coordinates and values
        coords = torch.tensor(x[:, 1:3], dtype=torch.float32).cuda()
        vals   = torch.tensor(x[:, 3:], dtype=torch.float32).cuda()
        
        # GPU k-means
        cluster_ids, centroids = kmeans(
            X=coords,
            num_clusters=min(n_tiles, coords.shape[0]),
            distance='euclidean',
            tol = 1e-4,
            iter_limit = 1000,
            device=torch.device('cuda')
        )
        
        # Randomly activate plot_cluster for 1 in 200 slides
        if np.random.randint(0, 200) == 0:
            plot_cluster(slide_name, coords, cluster_ids, centroids)

        # Aggregate features per cluster
        new_x = []
        new_c = []
        for cl in torch.unique(cluster_ids):
            mask_cl = cluster_ids == cl
            new_c.append(coords[mask_cl].mean(dim=0).cpu().numpy())
            new_x.append(vals[mask_cl].mean(dim=0).cpu().numpy())
        
        num_clusters = len(new_c) 

        # Create a new column for the zoom level
        zoom_col = np.full((num_clusters, 1), zoom_value, dtype=np.float32)

        x_clustered = np.concatenate([zoom_col, new_c, new_x], axis=1)  # shape: (num_clusters, 2051)

        # Pad if fewer than n_tiles
        num_clusters = x_clustered.shape[0]
        if num_clusters < n_tiles:
            padding = np.zeros((n_tiles - num_clusters, x_clustered.shape[1]), dtype=np.float32)
            x_clustered = np.concatenate([x_clustered, padding], axis=0)

        # Save to HDF5
        file['X'][n] = x_clustered

        cluster_ids_np = cluster_ids.cpu().numpy()  # Shape is (7997,)

        # Get the number of original tiles you're saving
        num_original_tiles = cluster_ids_np.shape[0]  # This is 7997

        # Create the destination array with the correct HDF5 shape (8000)
        # (You used 8000 when you created the dataset, not n_tiles)
        cluster_attr = np.zeros(8000, dtype=np.int64) 

        # Copy your (7997,) data into the (8000,) array
        cluster_attr[:num_original_tiles] = cluster_ids_np 

        # Save the full (8000,) array
        file['cluster_attribution'][n] = cluster_attr

        n += 1

    file.close()
        
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_to_transcriptome", help="path to transcriptome data saved as a csv file",
                        default='/home/dvanerp/pepsi/data/raw/tcga_rna/all_transcriptomes_incl_missing_projects.csv')
    parser.add_argument("--path_to_save_processed_data", help="path where supertile-preprocessed data should be saved",
                        default='data/TCGA_slic_100.h5')
    parser.add_argument("--n_tiles", help="number of supertiles",
                        default=100, type=int)
    args = parser.parse_args()
    os.makedirs(os.path.dirname(args.path_to_save_processed_data), exist_ok=True)
    print(f"Saving processed data to {args.path_to_save_processed_data}")
    rna_data = TranscriptomeDataset.from_saved_file(args.path_to_transcriptome, genes=[])
    histo_data = TCGAFolder.match_transcriptome_data(rna_data)
    histo_data.transform = ToTensor()
    
    cluster_dataset(histo_data, n_tiles=args.n_tiles, path_to_data=args.path_to_save_processed_data)


if __name__ == '__main__':

    main()
