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
from typing import List, Optional, Tuple
from utils import summarize_class
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit, ShuffleSplit, train_test_split

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
        log_transform: bool = True,
        project_column: str = 'metastasis'
    ):
        self.features_dir = Path(features_dir)
        self.max_tiles = max_tiles
        self.log_transform = log_transform
        self.project_column = project_column

        # Load keyfile for ID mapping
        self.id_to_study = self._load_keyfile(keyfile_path)
        
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
        
        self.metadata = self._load_metadata(keyfile_path)

        print(f"CSCCDataset initialized: {len(self)} patients, {len(self.genes)} genes, {self.feature_dim}D features")
    
    def _load_keyfile(self, keyfile_path: str) -> dict:
        """Load keyfile and filter by QC status."""
        keyfile = pd.read_csv(keyfile_path)
        
        if 'rna_qc_status' in keyfile.columns:
            initial_count = len(keyfile)
            # Normalize to lowercase and strip whitespace to be safe
            # Handle potential NaN values by converting to string first
            keyfile = keyfile[keyfile['rna_qc_status'].astype(str).str.lower().str.strip() != 'fail']
            filtered_count = len(keyfile)
            print(f"Filtered out {initial_count - filtered_count} patients due to QC failure. Remaining: {filtered_count}")
        else:
            print("Warning: 'rna_qc_status' column not found in keyfile. Skipping QC filtering.")
            
        return dict(zip(keyfile['skylinedx_id_rsm2'], keyfile['study_number']))

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
        missing_ids = []
        
        for idx, sid in enumerate(df['ID']):
            if sid in self.id_to_study:
                study_numbers.append(self.id_to_study[sid])
                valid_rows.append(idx)
            else:
                missing_ids.append(sid)
        
        if missing_ids:
            print(f"Warning: {len(missing_ids)} IDs from targets CSV not found in keyfile (or filtered by QC).")
            print(f"  First 5 missing: {missing_ids[:5]}...")
        
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
            print(f"  Missing target examples: {list(missing_targets)[:10]}...")
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
    
    def _load_features_nosubsample(self, study_number: str) -> torch.Tensor:
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
    
    def _load_metadata(self, keyfile_path: str) -> pd.DataFrame:
        """Load metadata from keyfile."""
        keyfile = pd.read_csv(keyfile_path, index_col="study_number")
        return keyfile
    
    def _normalize_tissue_type(self, tissue_type: str) -> str:
        """
        Normalize tissue_type for stratification. 
        Only Biopsy and Excision are kept; everything else becomes 'other'.
        """
        if pd.isna(tissue_type):
            return 'other'
        tissue_lower = str(tissue_type).lower().strip()
        if tissue_lower in ['biopsy']:
            return 'Biopsy'
        elif tissue_lower in ['excision']:
            return 'Excision'
        else:
            return 'other'
    
    def _build_sample_groups(self) -> Tuple[List[List[str]], List[str]]:
        """
        Group samples by rsm1_matching_set_id to prevent pair leakage.
        Samples with NaN or unique IDs become singleton groups.
        
        Returns:
            groups: List of lists, each inner list contains study_numbers in that group
            group_labels: Normalized tissue_type for each group (Biopsy/Excision/other)
        """
        groups = []
        group_labels = []
        
        # Get pair IDs and tissue types for all study_numbers in the dataset
        pair_ids = self.metadata.loc[self.study_numbers, 'rsm1_matching_set_id']
        tissue_types = self.metadata.loc[self.study_numbers, 'tissue_type']
        
        # Track which study_numbers have been assigned to a group
        assigned = set()
        
        # Build a mapping from pair_id to study_numbers
        pair_to_studies = {}
        for study_num in self.study_numbers:
            pair_id = pair_ids.loc[study_num]
            # Check if pair_id is NaN or missing
            if pd.isna(pair_id):
                # Singleton group - no pair
                groups.append([study_num])
                raw_tissue = tissue_types.loc[study_num]
                group_labels.append(self._normalize_tissue_type(raw_tissue))
                assigned.add(study_num)
            else:
                # Has a pair_id - group together
                if pair_id not in pair_to_studies:
                    pair_to_studies[pair_id] = []
                pair_to_studies[pair_id].append(study_num)
        
        # Add paired groups (those with the same pair_id)
        for pair_id, study_list in pair_to_studies.items():
            groups.append(study_list)
            # Use first member's tissue_type for the group label (normalized)
            raw_tissue = tissue_types.loc[study_list[0]]
            group_labels.append(self._normalize_tissue_type(raw_tissue))
        
        # Print distribution of stratification labels
        from collections import Counter
        label_counts = Counter(group_labels)
        
        print(f"Built {len(groups)} sample groups from {len(self.study_numbers)} samples")
        print(f"  Singleton groups: {sum(1 for g in groups if len(g) == 1)}")
        print(f"  Paired groups: {sum(1 for g in groups if len(g) > 1)}")
        print(f"  Stratification labels: {dict(label_counts)}")
        
        return groups, group_labels
    
    def _groups_to_indices(self, group_indices: np.ndarray, groups: List[List[str]]) -> np.ndarray:
        """
        Convert group indices to dataset indices.
        
        Args:
            group_indices: Indices into the groups list
            groups: List of study_number lists (from _build_sample_groups)
            
        Returns:
            Array of dataset indices
        """
        # Flatten selected groups to study_numbers
        selected_study_numbers = []
        for gi in group_indices:
            selected_study_numbers.extend(groups[gi])
        
        # Map study_numbers to dataset indices
        indices = np.arange(len(self))
        return indices[np.isin(self.patients, selected_study_numbers)]
    
    def _safe_stratified_split(
        self, 
        indices: np.ndarray, 
        labels: np.ndarray, 
        test_size: float, 
        random_state: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Attempt stratified split, fall back to regular split if not possible.
        
        This handles cases where some classes have too few samples for stratification.
        
        Args:
            indices: Array of indices to split
            labels: Corresponding labels for stratification
            test_size: Fraction for test/validation set
            random_state: Random seed
            
        Returns:
            train_indices, test_indices (relative to input indices array)
        """
        # Check if stratification is possible: each class needs at least 2 samples
        unique, counts = np.unique(labels, return_counts=True)
        min_count = counts.min()
        
        if min_count >= 2:
            # Stratified split is possible
            try:
                sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
                train_idx, test_idx = next(sss.split(indices, labels))
                return train_idx, test_idx
            except ValueError as e:
                print(f"  Warning: Stratified split failed ({e}), falling back to regular split")
        else:
            print(f"  Warning: Class '{unique[counts.argmin()]}' has only {min_count} sample(s), using regular split")
        
        # Fall back to regular (non-stratified) split
        ss = ShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        train_idx, test_idx = next(ss.split(indices))
        return train_idx, test_idx
    
    def stratified_split(
        self, 
        test_size: float = 0.1, 
        valid_size: float = 0.1, 
        random_state: int = 42
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Single 80/10/10 stratified split with pair protection.
        
        Pairs (same rsm1_matching_set_id) stay together in the same split.
        Stratifies by tissue_type to maintain ratios across splits.
        
        Args:
            test_size: Fraction for test set (default 0.1)
            valid_size: Fraction for validation set (default 0.1)
            random_state: Random seed for reproducibility
            
        Returns:
            train_idx, valid_idx, test_idx as numpy arrays of dataset indices
        """
        groups, group_labels = self._build_sample_groups()
        group_labels = np.array(group_labels)
        n_groups = len(groups)
        group_indices = np.arange(n_groups)
        
        # First split: separate test set (with fallback)
        trainval_rel_idx, test_rel_idx = self._safe_stratified_split(
            group_indices, group_labels, test_size, random_state)
        trainval_group_idx = group_indices[trainval_rel_idx]
        test_group_idx = group_indices[test_rel_idx]
        
        # Second split: separate validation from training
        # Adjust valid_size to be relative to the trainval set
        adjusted_valid_size = valid_size / (1 - test_size)
        trainval_labels = group_labels[trainval_group_idx]
        train_rel_idx, valid_rel_idx = self._safe_stratified_split(
            trainval_group_idx, trainval_labels, adjusted_valid_size, random_state)
        
        # Map back to absolute group indices
        train_group_idx = trainval_group_idx[train_rel_idx]
        valid_group_idx = trainval_group_idx[valid_rel_idx]
        
        # Convert to dataset indices
        train_idx = self._groups_to_indices(train_group_idx, groups)
        valid_idx = self._groups_to_indices(valid_group_idx, groups)
        test_idx = self._groups_to_indices(test_group_idx, groups)
        
        print(f"Stratified split: {len(train_idx)} train, {len(valid_idx)} valid, {len(test_idx)} test")
        self._print_split_stats(train_idx, valid_idx, test_idx)
        
        return train_idx, valid_idx, test_idx
    
    def _print_split_stats(self, train_idx: np.ndarray, valid_idx: np.ndarray, test_idx: np.ndarray):
        """Print tissue_type distribution for each split."""
        tissue_types = self.metadata.loc[self.study_numbers, 'tissue_type']
        
        for name, idx in [('Train', train_idx), ('Valid', valid_idx), ('Test', test_idx)]:
            split_study_nums = [self.study_numbers[i] for i in idx]
            split_tissues = tissue_types.loc[split_study_nums]
            counts = split_tissues.value_counts()
            pcts = (counts / len(split_tissues) * 100).round(1)
            print(f"  {name} tissue_type distribution: {dict(zip(counts.index, pcts.values))}%")
    
    def stratified_kfold(
        self,
        n_splits: int = 5,
        valid_size: float = 0.1,
        random_state: int = 42
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
        """
        K-fold cross-validation with stratification and pair protection.
        
        Pairs (same rsm1_matching_set_id) stay together in the same split.
        Stratifies by tissue_type to maintain ratios across folds.
        
        Args:
            n_splits: Number of folds (default 5)
            valid_size: Fraction of training set for validation (default 0.1)
            random_state: Random seed for reproducibility
            
        Returns:
            train_idx: List of n_splits arrays of training indices
            valid_idx: List of n_splits arrays of validation indices
            test_idx: List of n_splits arrays of test indices
        """
        groups, group_labels = self._build_sample_groups()
        group_labels = np.array(group_labels)
        n_groups = len(groups)
        group_indices = np.arange(n_groups)
        
        # Use StratifiedKFold on groups
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        
        train_idx = []
        valid_idx = []
        test_idx = []
        
        for k, (trainval_group_idx, test_group_idx) in enumerate(skf.split(group_indices, group_labels)):
            # Test set for this fold
            fold_test_idx = self._groups_to_indices(test_group_idx, groups)
            test_idx.append(fold_test_idx)
            
            # Split trainval into train and valid (stratified with fallback)
            if valid_size > 0:
                trainval_labels = group_labels[trainval_group_idx]
                train_rel_idx, valid_rel_idx = self._safe_stratified_split(
                    trainval_group_idx, trainval_labels, valid_size, random_state)
                
                train_group_idx = trainval_group_idx[train_rel_idx]
                valid_group_idx = trainval_group_idx[valid_rel_idx]
                
                fold_train_idx = self._groups_to_indices(train_group_idx, groups)
                fold_valid_idx = self._groups_to_indices(valid_group_idx, groups)
            else:
                fold_train_idx = self._groups_to_indices(trainval_group_idx, groups)
                fold_valid_idx = np.array([], dtype=int)
            
            train_idx.append(fold_train_idx)
            valid_idx.append(fold_valid_idx)
            
            print(f"Fold {k}: {len(fold_train_idx)} train, {len(fold_valid_idx)} valid, {len(fold_test_idx)} test")
        
        # Print overall stats for first fold as example
        print("\nFold 0 tissue_type distribution:")
        self._print_split_stats(train_idx[0], valid_idx[0], test_idx[0])
        
        return train_idx, valid_idx, test_idx

    def __len__(self) -> int:
        return len(self.study_numbers)
    
    def __getitem__(self, idx: int):
        study_number = self.study_numbers[idx]
        
        # Features: (max_tiles, feature_dim)
        features = self._load_features(study_number)
        
        # Targets: (n_genes,)
        targets = torch.from_numpy(self.targets[idx])
        
        # For now return without study_number for compatibility 
        return features, targets
    
    # === Utility properties for model training ===
    
    @property
    def projects(self) -> pd.Series:
        """Return the projects in the dataset as a pandas Series, matched to study_number order (study_number is the index)."""
        return self.metadata.loc[self.study_numbers, self.project_column]
    
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
        max_tiles=8000,
        genes=["ENSG00000000003.15"]
    )
    
    train_idx, valid_idx, test_idx = dataset.stratified_kfold()
    # Compatible with your existing patient_kfold
    print(f"Patients array: {dataset.patients[:16]}...")
    summarize_class(dataset)
    print(f"Projects: {dataset.projects}")

    loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=4, collate_fn=cscc_collate_with_ids)
    seen = set()
    for features, targets, study_numbers in loader:
        # features: (B, max_tiles, feature_dim)
        # targets: (B, n_genes)
        # study_number: (B,)
        # print(f"Sample features shape: {features.shape}")
        # print(f"Sample targets shape: {targets.shape}")
        # print(f"Sample study_number shape: {len(study_numbers)}")
        # print(f"Study numbers:\n {study_numbers}")
        # print(f"Patient 0: {study_numbers[0]};\n  features mean: {features[0].mean()};\n features std: {features[0].std()};\n  targets mean: {targets[0].mean()};\n targets std: {targets[0].std()}")
        
        for study_number in study_numbers:
            if study_number in seen:
                print(f"Warning: Duplicate study number: {study_number}")
            seen.add(study_number)
    print(f"Seen {len(seen)} study numbers")