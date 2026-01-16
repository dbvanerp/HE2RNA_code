'''
Building a dataset containing features and targets from the CSCC dataset.
Meant to replace the current dataset builder (transcriptome_data.py and wsi_data.py) in the HE2RNA code.
'''
import os
import h5py
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Optional
from utils import summarize_class
from torch.utils.data import DataLoader

class CSCCDataset(Dataset):
    """
    PyTorch Dataset for CSCC multi-instance learning.
    
    Features: H5 files per patient (filename = study_number.h5)
    Targets: Bulk RNA-seq from CSV
    Linking: keyfile maps skylinedx_id -> study_number
    
    Args:
        features_dir: Directory with H5 files (one per patient)
        targets_csv: CSV with RNA counts (ID column = skylinedx_id)
        keyfile_path: CSV mapping skylinedx_id_rsm2 -> study_number
        genes: Optional list of gene columns to use (None = all)
        max_tiles: Max tiles per patient for padding
        log_transform: Whether to log10(1+x) transform targets
    """
    
    def __init__(
        self,
        features_dir: str,
        targets_csv: str,
        keyfile_path: str,
        genes: Optional[List[str]] = None,
        max_tiles: int = 8000,
        log_transform: bool = True
    ):
        self.features_dir = Path(features_dir)
        self.max_tiles = max_tiles
        self.log_transform = log_transform
        
        # Load keyfile for ID mapping
        keyfile = pd.read_csv(keyfile_path)
        self.id_to_study = dict(zip(keyfile['skylinedx_id_rsm2'], keyfile['study_number']))
        
        # Load features index (study_number -> h5_path)
        self.feature_files = self._index_features()
        
        # Load targets and map to study_numbers
        self.targets_df, self.target_study_numbers = self._load_targets(targets_csv)
        
        # Find intersection: patients with both features AND targets
        self.study_numbers = self._align_patients()
        
        # Select genes
        if genes is None:
            # Auto-detect gene columns (adjust pattern as needed)
            self.genes = [c for c in self.targets_df.columns if c not in ['ID', 'study_number']]
        else:
            self.genes = genes
        
        # Build aligned target matrix
        self.targets = self._build_target_matrix()
        
        # Infer feature dimension
        self.feature_dim = self._get_feature_dim()
        
        print(f"CSCCDataset initialized: {len(self)} patients, {len(self.genes)} genes, {self.feature_dim}D features")
    
    def _index_features(self) -> dict:
        """Map study_number -> h5 file path."""
        feature_files = {}
        for f in os.listdir(self.features_dir):
            if f.endswith('.h5'):
                study_number = os.path.splitext(f)[0]
                feature_files[study_number] = self.features_dir / f
        print(f"Found {len(feature_files)} H5 feature files")
        return feature_files

    def _load_targets(self, targets_csv: str):
        """
        Load and transpose targets CSV so that columns are genes and rows are patients (samples),
        then map IDs to study_numbers.
        """
        df = pd.read_csv(targets_csv, index_col=0)  # Expect first col to be IDs, rows are samples, columns are genes
        # If columns are IDs and rows are genes, need to transpose so rows are IDs.
        if 'ID' not in df.columns:
            # likely the IDs are the index, and columns are genes
            df = df.transpose()
            df.index.name = 'ID'
            df = df.reset_index()
        # Now check if we still don't have 'ID' as a column, try to recover
        if 'ID' not in df.columns:
            # Try use index as 'ID'
            df['ID'] = df.index
        
        # Map skyline IDs to study numbers
        study_numbers = []
        valid_rows = []
        for idx, sid in enumerate(df['ID']):
            if sid in self.id_to_study:
                study_numbers.append(self.id_to_study[sid])
                valid_rows.append(idx)
            else:
                print(f"Warning: ID {sid} not found in keyfile, skipping")
        
        df = df.iloc[valid_rows].reset_index(drop=True)
        df['study_number'] = study_numbers

        print(f"Loaded {len(df)} targets with valid study_numbers after transpose fix")
        return df, study_numbers

    def _load_features(self, study_number: str) -> torch.Tensor:
        """Load and pad features for a patient. If there are more than max_tiles, pick a random subset (seed=42)."""
        h5_path = self.feature_files[study_number]
        with h5py.File(h5_path, 'r') as f:
            data = f['features'][:]
        
        n_tiles = data.shape[0]
        
        if n_tiles > self.max_tiles:
            rng = np.random.default_rng(seed=42)
            selected_indices = rng.choice(n_tiles, self.max_tiles, replace=False)
            selected_indices = np.sort(selected_indices)  # optional: sort for consistency
            data = data[selected_indices, :]
        elif n_tiles < self.max_tiles:
            padded = np.zeros((self.max_tiles, self.feature_dim), dtype=np.float32)
            padded[:n_tiles] = data
            data = padded
        # If n_tiles == self.max_tiles do nothing

        return torch.from_numpy(data.astype(np.float32))
        
        for idx, sid in enumerate(df['ID']):
            if sid in self.id_to_study:
                study_numbers.append(self.id_to_study[sid])
                valid_rows.append(idx)
            else:
                print(f"Warning: ID {sid} not found in keyfile, skipping")
        
        df = df.iloc[valid_rows].reset_index(drop=True)
        df['study_number'] = study_numbers
        
        print(f"Loaded {len(df)} targets with valid study_numbers")
        return df, study_numbers
    
    def _align_patients(self) -> List[str]:
        """Find patients with both features and targets."""
        has_features = set(self.feature_files.keys())
        has_targets = set(self.target_study_numbers)
        
        valid = has_features & has_targets
        missing_features = has_targets - has_features
        missing_targets = has_features - has_targets
        
        if missing_features:
            print(f"Warning: {len(missing_features)} patients have targets but no features")
            print(f"  Missing feature examples: {list(missing_features)[:5]}...")
        if missing_targets:
            print(f"Warning: {len(missing_targets)} patients have features but no targets")
            print(f"  Missing target examples: {list(missing_targets)[:5]}...")
        # Return in consistent order (sorted for reproducibility)
        study_numbers = sorted(list(valid))
        print(f"Aligned {len(study_numbers)} patients with both features and targets")
        return study_numbers
    
    def _build_target_matrix(self) -> np.ndarray:
        """Build (n_patients, n_genes) target matrix aligned to study_numbers."""
        # Index targets_df by study_number for fast lookup
        self.targets_df = self.targets_df.set_index('study_number')
        
        targets = self.targets_df.loc[self.study_numbers, self.genes].values.astype(np.float32)
        
        if self.log_transform:
            targets = np.log10(1 + targets)
        
        return targets
    
    def _get_feature_dim(self) -> int:
        """Infer feature dimension from first H5 file."""
        first_study = self.study_numbers[0]
        with h5py.File(self.feature_files[first_study], 'r') as f:
            return f['features'].shape[1]
    
    def _load_features(self, study_number: str) -> torch.Tensor:
        """Load and pad features for a patient."""
        h5_path = self.feature_files[study_number]
        
        with h5py.File(h5_path, 'r') as f:
            data = f['features'][:self.max_tiles]
        
        n_tiles = data.shape[0]
        
        # Pad if needed
        if n_tiles < self.max_tiles:
            padded = np.zeros((self.max_tiles, self.feature_dim), dtype=np.float32)
            padded[:n_tiles] = data
            data = padded
        
        return torch.from_numpy(data.astype(np.float32))
    
    def __len__(self) -> int:
        return len(self.study_numbers)
    
    def __getitem__(self, idx: int):
        study_number = self.study_numbers[idx]
        
        # Features: (max_tiles, feature_dim)
        features = self._load_features(study_number)
        
        # Targets: (n_genes,)
        targets = torch.from_numpy(self.targets[idx])
        return features, targets, study_number
    
    # === Utility properties for model training ===
    
    @property
    def patients(self) -> np.ndarray:
        """For compatibility with patient_kfold and other split functions."""
        return np.array(self.study_numbers)
    
    @property
    def dim(self) -> int:
        """Feature dimension for model initialization."""
        return self.feature_dim


# === Optional: collate function if you need metadata ===

def cscc_collate_with_ids(batch):
    """Collate that also returns study IDs (if needed for analysis)."""
    features = torch.stack([b[0] for b in batch])
    targets = torch.stack([b[1] for b in batch])
    study_numbers = [b[2] for b in batch]
    return features, targets, study_numbers


if __name__ == "__main__":
    features_dir = "/gpfs/work4/0/prjs1086/derm_shared/cscc/processed/he_packed_patch_feat_uni/rsm2/h5_files"
    targets_csv = "/gpfs/work4/0/prjs1086/derm_shared/cscc/processed/rna_counts/txi.gene.counts.csv"
    keyfile_path = "/gpfs/work4/0/prjs1086/derm_shared/cscc/doc/keyfiles/20260113_project_keyfile_anonymized_rsm2.csv"
    
    dataset = CSCCDataset(
        features_dir=features_dir,
        targets_csv=targets_csv,
        keyfile_path=keyfile_path,
        max_tiles=8000
    )
    

    # Compatible with your existing patient_kfold
    print(f"Patients array: {dataset.patients[:16]}...")
    summarize_class(dataset)

    loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=4, collate_fn=cscc_collate_with_ids)

    for features, targets, study_numbers in loader:
        # features: (B, max_tiles, feature_dim)
        # targets: (B, n_genes)
        # study_number: (B,)
        print(f"Sample features shape: {features.shape}")
        print(f"Sample targets shape: {targets.shape}")
        print(f"Sample study_number shape: {len(study_numbers)}")
        print(f"Study numbers:\n {study_numbers}")
        break