"""
HE2RNA: Extract prediction of gene expression per tile and compare to ground truth
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
import openslide
import openslide.deepzoom
import pickle as pkl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import torch
from tqdm import tqdm
from torch import nn
from concurrent.futures import ThreadPoolExecutor
from sklearn.metrics import roc_curve, roc_auc_score
from scipy.stats import pearsonr, spearmanr


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def compute_heatmap(path_to_model, path_to_tile_features):

    X_he = np.load(path_to_tile_features)
    coords = X_he[:, :3]
        
    all_scores = []
    x = torch.Tensor(X_he[np.newaxis].transpose(1, 2, 0)).to(DEVICE)
    clusters = np.arange(X_he.shape[0])
    if os.path.isdir(path_to_model):
        models = [torch.load(f'{path_to_model}/model_' +
                             str(k) + '/model.pt', map_location=DEVICE) for k in range(5)]
    else:
        models = [torch.load(path_to_model, map_location=DEVICE)]

    for model in tqdm(models):
        all_scores.append(model.conv(x).detach().cpu().numpy())
        # all_scores = np.exp(all_scores) - 1 # back log transform, not in OG code

    print(len(all_scores))
    # Average over genes and cross-val folds
    tile_scores = np.mean(all_scores, axis=(0, 2))[:, 0]
    
    return coords, tile_scores


def display_heatmap(path_to_slide, coords, tile_scores, path=None,
                   vmin=None, vmax=None, vmin_pct=10, vmax_pct=99,
                   heatmap_only_path=None):
    """
    Display a heatmap overlay and optionally save a publication-quality heatmap+scale.

    Args:
        path_to_slide: Path to slide (.svs).
        coords: Tile coordinates, shape (N, 3).
        tile_scores: Per-tile scores, shape (N,).
        path: Path to save the two-panel display (optional).
        vmin: Minimum value for color scaling (optional).
        vmax: Maximum value for color scaling (optional).
        vmin_pct: Percentile for autogenerating vmin if vmin not supplied.
        vmax_pct: Percentile for autogenerating vmax if vmax not supplied.
        heatmap_only_path: If given, output a high-quality, heatmap+scale-only PNG.
    """
    slide_he = openslide.OpenSlide(path_to_slide)
    print(f'Dimensions of the slide: {slide_he.dimensions}')

    fig, (ax1, ax2) = plt.subplots(1, 2, gridspec_kw={'wspace': 0, 'hspace': 0})
    fig.set_size_inches((15, 10))
    zoom_he = openslide.deepzoom.DeepZoomGenerator(slide_he, tile_size=224, overlap=0)
    im = np.array(slide_he.get_thumbnail((2000, 2000)))
    ax1.imshow(im)
    ax1.set_xticks([])
    ax1.set_yticks([])

    n_tiles = zoom_he.level_tiles[int(coords[0, 0])]
    grid = (np.array(im.shape[:2]) / n_tiles[::-1])

    score = tile_scores
    # Determine display range for better contrast
    if vmin is None or vmax is None:
        # Percentile-based range on non-zero scores (avoid background)
        nonzero = score[score != 0]
        if nonzero.size == 0:
            nonzero = score
        vmin_val = np.percentile(nonzero, vmin_pct) if vmin is None else vmin
        vmax_val = np.percentile(nonzero, vmax_pct) if vmax is None else vmax
    else:
        vmin_val, vmax_val = vmin, vmax
    score = np.clip(score, vmin_val, vmax_val)

    mask = np.zeros_like(im[:, :, 0]).astype(float)
    for s, coord in zip(score, coords):
        # Remove unnecessary padding by not adding manual offsets
        x = int(coord[2])
        y = int(coord[1])
        mask[int(x * grid[0]): int((x + 1) * grid[0]),
             int(y * grid[0]): int((y + 1) * grid[0])] = s

    ims = ax2.imshow(mask, cmap='inferno', vmin=vmin_val, vmax=vmax_val)
    ax2.set_xticks([])
    ax2.set_yticks([])
    cbar = plt.colorbar(ims, ax=ax2)
    cbar.ax.tick_params(labelsize=16)

    if path is not None:
        plt.savefig(path, bbox_inches='tight', dpi=300)
        heatmap_only_path = path.replace('.png', '_heatmap_only.png')

    plt.close(fig)

    # --- Save heatmap+scale only version, high quality ---
    if heatmap_only_path is not None:
        # Make the background white by setting facecolor='white'
        fig2, ax = plt.subplots(1, 1, facecolor='white')
        fig2.set_size_inches((8, 8))
        ims2 = ax.imshow(mask, cmap='inferno', vmin=vmin_val, vmax=vmax_val)
        ax.set_xticks([])
        ax.set_yticks([])
        cbar2 = plt.colorbar(ims2, ax=ax, shrink=0.9, pad=0.02)
        cbar2.ax.tick_params(labelsize=18)
        fig2.tight_layout(pad=0.2)
        fig2.savefig(heatmap_only_path, bbox_inches='tight', dpi=300, facecolor='white')
        plt.close(fig2)
    else:
        if path is None:
            # Only show if not saving to file
            plt.figure()
            plt.imshow(mask, cmap='inferno', vmin=vmin_val, vmax=vmax_val)
            plt.xticks([])
            plt.yticks([])
            plt.colorbar()
            plt.show()
def compute_aucs_CRC(path_to_model, path_to_tiles):
    scores = []
    cats = ['LYM', 'ADI', 'STR', 'NORM', 'TUM', 'DEB', 'MUS', 'MUC', 'BACK']

    if os.path.isdir(path_to_model):
        models = [torch.load(f'{path_to_model}/model_' + str(k) +
                         '/model.pt', map_location=DEVICE) for k in range(5)]
    else:
        models = [torch.load(path_to_model, map_location=DEVICE)]

    for cat in tqdm(cats):
        all_scores = []
        X_he = np.load(os.path.join(path_to_tiles, f'{cat}.npy'))
        x = torch.Tensor(X_he.transpose(1, 2, 0)).to(DEVICE)
        clusters = np.arange(X_he.shape[0])

        for model in models:
            all_scores.append(model.conv(x).detach().cpu().numpy())

        all_scores = np.mean(all_scores, axis=(0, 1, 2))
        scores.append(all_scores)
    labels = np.concatenate([np.ones_like(scores[0]), np.zeros_like(np.concatenate(scores[1:]))])
    auc_lym_vs_all = roc_auc_score(labels, np.concatenate(scores))
    print(f'AUC for lymphocytes vs all other classes: {auc_lym_vs_all:.4f}')
    dic = {}
    for i in range(1, 8):
        labels = np.concatenate([np.ones_like(scores[0]), np.zeros_like(scores[i])])
        auc = roc_auc_score(labels, np.concatenate([scores[0], scores[i]]))
        print(f'AUC for lymphocytes vs class {cats[i]}: {auc:.4f}')
        dic[f'AUC LYM vs {cats[i]}'] = auc
    return auc_lym_vs_all, dic


def post_processing(seg):
    return (seg[:, :, 0] > 1).mean()


def compute_correlation_PESO(path_to_model, path_to_tiles, path_to_masks, corr='pearson', per_slide=False):
    scores = []
    gts = []
    per_slide_corrs = []
    files = os.listdir(path_to_tiles)
    ns = np.unique([file.split('_')[1] for file in files])

    if os.path.isdir(path_to_model):
        models = [torch.load(f'{path_to_model}/model_' + str(k) +
                         '/model.pt', map_location=DEVICE) for k in range(5)]
    else:
        models = [torch.load(path_to_model, map_location=DEVICE)]

    for n in tqdm(ns):

        X_he = np.load(os.path.join(path_to_tiles, f'pds_{n}_HE.npy'))
        coords = X_he[:, :3]
        mask_ = openslide.OpenSlide(os.path.join(path_to_masks, f'pds_{n}_HE_training_mask.tif'))

        zoom_mask = openslide.deepzoom.DeepZoomGenerator(mask_, tile_size=224, overlap=0)

        tile_scores = []
        x = torch.Tensor(X_he[np.newaxis].transpose(1, 2, 0)).to(DEVICE)
        clusters = np.arange(X_he.shape[0])

        for model in models:
            tile_scores.append(model.conv(x).detach().cpu().numpy())
        tile_scores = np.mean(tile_scores, axis=(0, 2, 3))
        scores.append(tile_scores)

        gt = []
        def get_gt_tile(coord):
            img_mask = np.array(
                zoom_mask.get_tile(int(coord[0]), (int(coord[1]), int(coord[2]))))

            return post_processing(img_mask)

        with ThreadPoolExecutor() as executor:
            gt = list(tqdm(executor.map(get_gt_tile, coords), total=len(coords), leave=False))

        gt = np.array(gt)
        print(gt.shape, gt[:5], tile_scores.shape, tile_scores[:5])
        print(f"GT mean: {np.mean(gt):.4f}, std: {np.std(gt):.4f}")
        print(f"tile_scores mean: {np.mean(tile_scores):.4f}, std: {np.std(tile_scores):.4f}")
        gts.append(gt)
        if per_slide:
            if corr == 'pearson':
                per_slide_corrs.append((n, pearsonr(gt, tile_scores)))
            elif corr == 'spearman':
                per_slide_corrs.append((n, spearmanr(gt, tile_scores)))
    
    gts = np.concatenate(gts)
    scores = np.concatenate(scores)
    if corr == 'pearson':
        overall = pearsonr(gts, scores)
    elif corr == 'spearman':
        overall = spearmanr(gts, scores)
    return overall if not per_slide else (overall, per_slide_corrs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", help="dataset on which to carry spatialization experiment, CRC or PESO")
    parser.add_argument("--path_to_model", help="path to the folder containing the models trained by cross-validation",
                        default='epithelium_selection')
    parser.add_argument("--path_to_tiles", help="path to folder containing .npy files of tile features")
    parser.add_argument("--path_to_slide", help="path to a single slide for heatmap generation")
    parser.add_argument("--slides", nargs='+', help="paths to slides for batch heatmap generation")
    parser.add_argument("--slides_dir", help="directory containing slides for batch heatmap generation")
    parser.add_argument("--path_to_masks", help="path to folder containing training masks from PESO")
    parser.add_argument("--corr", help="type of correlation to compute, pearson or spearman", default='pearson')
    parser.add_argument("--per_slide_corr", action='store_true', default=True,
                        help="also report correlation per slide in PESO experiment")
    parser.add_argument("--heatmap_vmin", type=float, default=None,
                        help="absolute minimum value for heatmap color scale")
    parser.add_argument("--heatmap_vmax", type=float, default=None,
                        help="absolute maximum value for heatmap color scale")
    parser.add_argument("--heatmap_vmin_pct", type=float, default=10,
                        help="percentile for min when vmin not set (default 10)")
    parser.add_argument("--heatmap_vmax_pct", type=float, default=99,
                        help="percentile for max when vmax not set (default 99)")
    parser.add_argument("--output_path", help="path to save the heatmap")
    args = parser.parse_args()

    print(f"Using device: {DEVICE}")
    if args.output_path is not None:
        os.makedirs(args.output_path, exist_ok=True)

    if args.experiment == 'CRC':
        compute_aucs_CRC(args.path_to_model, args.path_to_tiles)
    elif args.experiment == 'PESO':
        corr_val = compute_correlation_PESO(
            args.path_to_model, args.path_to_tiles, args.path_to_masks, args.corr, args.per_slide_corr
        )
        if args.per_slide_corr:
            overall, per_slide = corr_val
            print(f"{args.corr} correlation (overall model vs ground truth tiles): {overall}")
            for slide_id, c in per_slide:
                print(f"Slide {slide_id}: {args.corr} correlation {c}")
        else:
            print(f"{args.corr} correlation (model vs ground truth tiles): {corr_val}")
    elif args.experiment == 'PESO_heatmap':
        if args.slides_dir:
            if args.output_path is None:
                raise ValueError("output_path is required when generating multiple heatmaps.")
            os.makedirs(args.output_path, exist_ok=True)
            slide_paths = [os.path.join(args.slides_dir, f) for f in os.listdir(args.slides_dir)]
            slide_paths = [p for p in slide_paths if os.path.isfile(p)]
            if len(slide_paths) == 0:
                raise ValueError(f"No slides found in {args.slides_dir}")
            for slide_path in slide_paths:
                if True:
                    slide_name = os.path.splitext(os.path.basename(slide_path))[0]
                    tiles_path = args.path_to_tiles
                    if os.path.isdir(tiles_path):
                        tile_features_path = os.path.join(tiles_path, f"{slide_name}.npy")
                    else:
                        tile_features_path = tiles_path

                    if not os.path.exists(tile_features_path):
                        print(f"Tile features not found for slide {slide_name} at {tile_features_path}, skipping.")
                        continue

                    coords, tile_scores = compute_heatmap(args.path_to_model, tile_features_path)
                    output_file = os.path.join(args.output_path, f"{slide_name}_heatmap.png")
                    print(f"Saving heatmap for {slide_name} to {output_file}")
                    
                    display_heatmap(
                        slide_path, coords, tile_scores, output_file,
                        vmin=args.heatmap_vmin, vmax=args.heatmap_vmax,
                        vmin_pct=args.heatmap_vmin_pct, vmax_pct=args.heatmap_vmax_pct
                    )
                else: 
                    print(f"Skipping heatmap for {slide_path} because it does not contain '60' in the path")
        elif args.slides:
            if args.output_path is None:
                raise ValueError("output_path is required when generating multiple heatmaps.")
            os.makedirs(args.output_path, exist_ok=True)
            for slide_path in args.slides:
                slide_name = os.path.splitext(os.path.basename(slide_path))[0]
                tiles_path = args.path_to_tiles
                if os.path.isdir(tiles_path):
                    tile_features_path = os.path.join(tiles_path, f"{slide_name}.npy")
                else:
                    tile_features_path = tiles_path

                if not os.path.exists(tile_features_path):
                    print(f"Tile features not found for slide {slide_name} at {tile_features_path}, skipping.")
                    continue

                coords, tile_scores = compute_heatmap(args.path_to_model, tile_features_path)
                output_file = os.path.join(args.output_path, f"{slide_name}_heatmap.png")
                print(f"Saving heatmap for {slide_name} to {output_file}")
                display_heatmap(
                    slide_path, coords, tile_scores, output_file,
                    vmin=args.heatmap_vmin, vmax=args.heatmap_vmax,
                    vmin_pct=args.heatmap_vmin_pct, vmax_pct=args.heatmap_vmax_pct
                )
        else:
            coords, tile_scores = compute_heatmap(args.path_to_model, args.path_to_tiles)
            display_heatmap(
                args.path_to_slide, coords, tile_scores, args.output_path,
                vmin=args.heatmap_vmin, vmax=args.heatmap_vmax,
                vmin_pct=args.heatmap_vmin_pct, vmax_pct=args.heatmap_vmax_pct
            )
    else:
        print("unrecognized experiment")

if __name__ == '__main__':

    main()