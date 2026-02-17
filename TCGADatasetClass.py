'''
Building a dataset containing features and targets from the TCGA dataset, based on the CSCCDatasetClass.py file.
Meant to replace the current dataset builder (transcriptome_data.py and wsi_data.py) in the HE2RNA code.
'''
import os
import h5py
import torch
from torch.utils.data import Dataset, Subset
from tqdm import tqdm
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple
from utils import summarize_class
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit, ShuffleSplit, train_test_split

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

def _get_memory_usage():
    """Get current memory usage in GB. Returns (used_gb, available_gb, total_gb)."""
    if HAS_PSUTIL:
        mem = psutil.virtual_memory()
        return mem.used / (1024**3), mem.available / (1024**3), mem.total / (1024**3)
    else:
        # Fallback: read from /proc/meminfo (Linux)
        try:
            with open('/proc/meminfo', 'r') as f:
                meminfo = f.read()
            lines = meminfo.split('\n')
            mem_total = int([l for l in lines if 'MemTotal' in l][0].split()[1]) / (1024**2)  # GB
            mem_available = int([l for l in lines if 'MemAvailable' in l][0].split()[1]) / (1024**2)  # GB
            mem_used = mem_total - mem_available
            return mem_used, mem_available, mem_total
        except:
            return None, None, None


class TCGADataset(Dataset):
    """
    PyTorch Dataset for TCGA Multi-instance learning.
    
    Features: H5 files per patient (filename = sample_id.h5)
    Targets: Bulk RNA-seq from CSV
    Linking: keyfile contains info like Project.ID, Sample.Type, Sample.ID, etc.

    Args:
        features_dir: Directory with H5 files (one per patient)
        targets_csv: CSV with RNA counts (ID column = Sample.ID)
        keyfile_path: CSV containing info like Project.ID, Sample.Type, Sample.ID, etc.
        genes: Optional list of gene columns to use (None = all)
        max_tiles: Max tiles per patient for padding
        log_transform: Whether to log10(1+x) transform targets
    """
    
    def __init__(
        self,
        targets_csv: str,
        keyfile_path: str,
        features_dir: Optional[str] = None,
        features_aggregated: Optional[str] = None, 
        genes: Optional[List[str]] = None,
        max_tiles: int = 8000,
        log_transform: bool = True,
        project_filter: list[str] = None
    ):
        print(f"\n{'-'*15} Initializing TCGADataset {'-'*15}\n")

        
        self.features_dir = features_dir
        self.features_aggregated_h5 = features_aggregated
        self.max_tiles = max_tiles
        self.log_transform = log_transform
        self.project_filter = project_filter
        self.genes = genes

        # Load keyfile and filter by sample type and project if applicable
        self.filtered_keyfile = self._load_keyfile(keyfile_path, self.project_filter)
        

        if self.features_dir is not None:
            # Load features index (sample_id -> h5_path) for lazy loading from disk
            self.feature_files = self._index_features()
            self.aggregated = False
        elif self.features_aggregated_h5 is not None:
            self.aggregated = True
            self.aggregated_h5 = h5py.File(self.features_aggregated_h5, 'r')
            # Build mapping: slide_name -> index in the dataset
            slide_names = self.aggregated_h5['slide_name'][:]
            # Convert bytes to strings if needed and create index mapping
            if isinstance(slide_names[0], bytes):
                self.feature_files = {k.decode(): idx for idx, k in enumerate(slide_names)}
            else:
                self.feature_files = {str(k): idx for idx, k in enumerate(slide_names)}
            print(f"Indexed {len(self.feature_files)} samples from aggregated H5 file")
        else:
            raise ValueError("Either features_dir or features_aggregated must be provided")

        # Load targets and map to sample_ids
        self.targets_df, self.target_sample_ids = self._load_targets(targets_csv)
        
        # Find intersection: patients with both features AND targets
        self.sample_ids = self._align_patients()
        
        # TODO Don't think i use this
        self.sample_ids_truncated = [sid[:15] for sid in self.sample_ids] 

        # For aggregated case, load features into memory to avoid I/O bottlenecks
        if self.aggregated:
            self._load_aggregated_features_to_memory()

        # Select genes
        if genes is None:
            # Auto-detect gene columns and exclude metadata columns like File.ID/Case.ID/Project.ID.
            self.genes = [c for c in self.targets_df.columns if str(c).startswith('ENSG')]
            print(f"No gene filter provided. Auto-detected {len(self.genes)} ENSG gene columns")
        else:
            self.genes = self._filter_genes_by_list(genes)
        
        # Build aligned target matrix
        self.targets = self._build_target_matrix()

        # Build projects series aligned to sample_ids (one per full sample_id)
        project_rows = []
        truncated_to_target_idx = {trunc_id: idx for idx, trunc_id in enumerate(self.targets_df.index)}
        for full_sid in self.sample_ids:
            trunc_sid = full_sid[:15]
            if trunc_sid in truncated_to_target_idx:
                project_rows.append(truncated_to_target_idx[trunc_sid])
            else:
                raise ValueError(f"Truncated sample ID {trunc_sid} (from {full_sid}) not found in targets_df")
        
        self.projects = self.targets_df.iloc[project_rows]['Project.ID']
        # Infer feature dimension
        self.feature_dim = self._get_feature_dim()
        
        print(f"\n{'-'*15} TCGADataset initialized: {len(self)} slides, {len(self.genes)} genes, {self.feature_dim}D features {'-'*15}\n")
        print(f"Full summary of the dataset:")
        summarize_class(self)


    def _load_keyfile(self, keyfile_path: str, project_filter: list[str] = None) -> set:
        """Load keyfile and filter by Sample.Type and project if applicable."""
        print(f"\nFiltering patients by project and sample type")
        keyfile = pd.read_csv(keyfile_path)

        if project_filter is not None:
            initial_count = len(keyfile)
            keyfile = keyfile[keyfile['Project.ID'].isin(project_filter)]
            filtered_count = len(keyfile)
            print(f"  Filtered out {initial_count - filtered_count} patients that are not in the project filter. Remaining: {filtered_count}")
        else:
            print("  No project filter provided. Using all projects.")
            
        if 'Sample.Type' in keyfile.columns:
            initial_count = len(keyfile)
            # Handle potential NaN values by converting to string first
            keyfile = keyfile[keyfile['Sample.Type'] == 'Primary Tumor']
            filtered_count = len(keyfile)
            print(f"  Filtered out {initial_count - filtered_count} patients that are not Primary Tumor. Remaining: {filtered_count}")
        else:
            print("  Warning: 'Sample.Type' column not found in keyfile. Skipping Sample.Type filtering.")
            
        # Convert to set for O(1) lookup performance
        return set(keyfile['Sample.ID'].values)

    def _index_features(self) -> dict:
        """Map sample_id -> npy file path."""
        zoom_folder = "0.50_mpp"
        feature_files = {}

        print(f"\nIndexing features from {self.features_dir}")

        if self.project_filter is None:
            # Get all projects in the features directory
            projects = os.listdir(self.features_dir)
        else:
            projects = self.project_filter

        for project in projects:
            project_dir = os.path.join(self.features_dir, project)
            if not os.path.isdir(project_dir):
                continue

            # Support both layouts:
            # 1) <features_dir>/<project>/0.50_mpp/*.npy
            # 2) <features_dir>/<project>/*.npy
            zoom_dir = os.path.join(project_dir, zoom_folder)
            search_dir = zoom_dir if os.path.isdir(zoom_dir) else project_dir

            for f in os.listdir(search_dir):
                if f.endswith('.npy'):
                    sample_id = os.path.splitext(f)[0]
                    feature_files[sample_id] = os.path.join(search_dir, f)
        print(f"  Found {len(feature_files)} .npy feature files")
        return feature_files

    def _load_targets(self, targets_csv: str):
        """
        Load and transpose targets CSV so that columns are genes and rows are patients (samples),
        then map IDs to sample_ids.
        """
        print(f"\nLoading targets from {os.path.basename(targets_csv)}\nMight take a while...")

        if self.genes is None or '.' not in self.genes[0]:
            # No gene filter provided, or genes are not in the format ENSG00000000003.15. Use all genes (filter later)
            df = pd.read_csv(targets_csv, index_col="Sample.ID")
        else:
            # Genes are provided in the format ENSG00000000003.15. Use only the genes in the list to save memory
            print(f"  Loading only the provided genes to save memory: {len(self.genes)} genes")
            try:
                usecols = self.genes + ['File.ID', 'Sample.ID', 'Case.ID', 'Project.ID']
                df = pd.read_csv(targets_csv, index_col="Sample.ID", usecols=usecols)
            except:
                print(f"  Error: Could not load targets CSV with provided genes using usecols (likely versioning mismatches). \n  Trying again with all genes, filtering later.")
                df = pd.read_csv(targets_csv, index_col="Sample.ID")
        # Target sample IDs are of form TCGA-02-0003-01A, len 16
        # self.filtered_keyfile is a set of sample IDs after filtering (for O(1) lookup)
        
        # Filter to only include samples in filtered_keyfile (O(n) instead of O(n*m))
        valid_mask = df.index.isin(self.filtered_keyfile)
        missing_count = (~valid_mask).sum()
        
        if missing_count > 0:
            print(f"Warning: {missing_count} IDs from targets CSV not found in keyfile (or filtered by QC).")
            # Get missing IDs before filtering
            missing_ids = df.index[~valid_mask].tolist()
            print(f"  First 5 missing: {missing_ids[:5]}...")
        
        # Filter the dataframe
        df = df[valid_mask].copy()
        sample_ids = df.index.tolist()
        
        # Truncate index to remove A/Z suffix
        df.index = df.index.str[:-1]
        df.index.name = 'sample_id'

        print(f"  Loaded {len(df)} targets with valid (filtered) sample_ids")
        return df, sample_ids

    def _load_aggregated_features_to_memory(self):
        """Load all aggregated features into memory with pre-processing (padding/random selection) for fast access during training."""
        print(f"\nLoading aggregated features into memory...")
        print(f"  Loading {len(self.sample_ids)} samples from H5 file")
        
        # Check initial memory
        used_gb, available_gb, total_gb = _get_memory_usage()
        if used_gb is not None:
            print(f"  Initial memory: {used_gb:.2f} GB used, {available_gb:.2f} GB available, {total_gb:.2f} GB total")
        
        # First, get feature_dim from first sample
        first_sample_id = self.sample_ids[0]
        first_h5_idx = self.feature_files[first_sample_id]
        first_data = self.aggregated_h5['X'][first_h5_idx, :]
        first_data = first_data[:, 3:]  # Remove first 3 columns
        feature_dim = first_data.shape[1]
        
        # Estimate memory per sample (max_tiles x feature_dim x 4 bytes for float32)
        estimated_mb_per_sample = (self.max_tiles * feature_dim * 4) / (1024 * 1024)
        estimated_total_gb = (estimated_mb_per_sample * len(self.sample_ids)) / 1024
        print(f"  Estimated memory per sample: {estimated_mb_per_sample:.2f} MB")
        print(f"  Estimated total cache size: {estimated_total_gb:.2f} GB")
        
        self.aggregated_features_cache = {}
        rng = np.random.default_rng(seed=42)
        
        # Memory check intervals: every 1000 samples or every 10GB estimated
        check_interval = max(100, min(1000, len(self.sample_ids) // 20))  # Check ~20 times during loading
        last_check_idx = 0
        last_check_memory = used_gb
        
        # Load and pre-process features for all aligned sample_ids
        for idx, sample_id in enumerate(tqdm(self.sample_ids, desc="  Loading & processing features")):
            if sample_id in self.feature_files:
                h5_idx = self.feature_files[sample_id]
                # Load from H5
                data = self.aggregated_h5['X'][h5_idx, :]
                data = data[:, 3:]  # Remove first 3 columns
                n_tiles = data.shape[0]
                
                # Apply padding and/or random selection
                if n_tiles > self.max_tiles:
                    # Random selection
                    selected_indices = rng.choice(n_tiles, self.max_tiles, replace=False)
                    selected_indices = np.sort(selected_indices)  # Sort for consistency
                    data = data[selected_indices, :]
                elif n_tiles < self.max_tiles:
                    # Padding
                    padded = np.zeros((self.max_tiles, feature_dim), dtype=np.float32)
                    padded[:n_tiles] = data
                    data = padded
                # If n_tiles == self.max_tiles, data is already correct size
                
                # Convert to torch tensor and store in cache
                self.aggregated_features_cache[sample_id] = torch.from_numpy(data.astype(np.float32))
            
            # Periodic memory check
            if (idx + 1) % check_interval == 0 or idx == len(self.sample_ids) - 1:
                used_gb, available_gb, total_gb = _get_memory_usage()
                if used_gb is not None:
                    samples_loaded = idx + 1
                    memory_delta = used_gb - last_check_memory if last_check_memory is not None else None
                    cache_memory_mb = sum(
                        tensor.element_size() * tensor.nelement() / (1024 * 1024) 
                        for tensor in self.aggregated_features_cache.values()
                    )
                    cache_memory_gb = cache_memory_mb / 1024
                    
                    print(f"  Progress: {samples_loaded}/{len(self.sample_ids)} samples loaded")
                    print(f"    Cache size: {cache_memory_gb:.2f} GB")
                    print(f"    System memory: {used_gb:.2f} GB used, {available_gb:.2f} GB available")
                    if memory_delta is not None:
                        print(f"    Memory increase since last check: {memory_delta:.2f} GB")
                    
                    # Warn if available memory is getting low
                    if available_gb < 10:
                        print(f"    WARNING: Only {available_gb:.2f} GB available! Risk of OOM.")
                    elif available_gb < 20:
                        print(f"    WARNING: Low available memory: {available_gb:.2f} GB")
                    
                    last_check_memory = used_gb
                    last_check_idx = idx
        
        # Close H5 file since we've loaded everything into memory
        self.aggregated_h5.close()
        self.aggregated_h5 = None
        
        # Final memory check
        used_gb, available_gb, total_gb = _get_memory_usage()
        if used_gb is not None:
            print(f"  Final memory: {used_gb:.2f} GB used, {available_gb:.2f} GB available")
        
        # Calculate memory usage
        total_memory_mb = sum(
            tensor.element_size() * tensor.nelement() / (1024 * 1024) 
            for tensor in self.aggregated_features_cache.values()
        )
        print(f"  Loaded {len(self.aggregated_features_cache)} samples into memory (pre-processed)")
        print(f"  Total cache memory usage: {total_memory_mb:.2f} MB ({total_memory_mb/1024:.2f} GB)")
    
    def _load_features(self, sample_id: str) -> torch.Tensor:
        """Load and pad features for a patient. If there are more than max_tiles, pick a random subset (seed=42).
        
        Returns tensor of shape (feature_dim, max_tiles) to match model expectations.
        Model expects (batch, features, tiles) format.
        """
        # self.feature_files is a dict of sample_id -> h5_path OR sample_id -> index in aggregated_h5
        if self.aggregated:
            # Return pre-processed tensor directly from cache (no processing needed)
            # Cache stores (max_tiles, feature_dim), need to transpose to (feature_dim, max_tiles)
            return self.aggregated_features_cache[sample_id].transpose(0, 1)
        else:
            # Non-aggregated case: load from .npy file and process
            npy_path = self.feature_files[sample_id]
            data = np.load(npy_path)
            data = data[:, 3:]
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

            # Transpose from (tiles, features) to (features, tiles) to match model expectations
            return torch.from_numpy(data.astype(np.float32).transpose(1, 0))
        

    
    def _align_patients(self) -> List[str]:
        """Find patients with both features and targets."""
        print(f"\nAligning features and targets")
        # Efficient version: Use mappings for versioned sample IDs
        # Create dictionaries mapping base ID -> set of full IDs for both features and targets
        def build_base_dict(ids):
            base_dict = {}
            for sid in ids:
                base = sid[:15]
                if base not in base_dict:
                    base_dict[base] = set()
                base_dict[base].add(sid)
            return base_dict

        features_by_base = build_base_dict(self.feature_files.keys())
        targets_by_base = build_base_dict(self.target_sample_ids)

        # valid are all feature IDs for which there is any matching base ID in targets
        valid = set()
        for base in features_by_base.keys() & targets_by_base.keys():
            valid.update(features_by_base[base])

        # missing features: target IDs for which no feature is available for the same base
        missing_features = set()
        missing_targets = set()
        for base in targets_by_base.keys() - features_by_base.keys():
            missing_features.update(targets_by_base[base])
        for base in features_by_base.keys() - targets_by_base.keys():
            missing_targets.update(features_by_base[base])

        if missing_features:
            print(f"  Warning: {len(missing_features)} patients have targets but no features")
            print(f"    Missing feature examples: {list(missing_features)[:5]}...")
                
        if missing_targets:
            print(f"  Warning: {len(missing_targets)} patients have features but no targets after filtering")
            print(f"    Missing target examples: {list(missing_targets)[:10]}...")
        # Return in consistent order (sorted for reproducibility)
        sample_ids = sorted(list(valid))
        if len(sample_ids) == 0:
            raise ValueError("No feature samples found with targets after filtering")
        else:
            print(f"  Aligned {len(sample_ids)} feature samples with targets")
        
        return sample_ids
    
    def _filter_genes_by_list(self, genes: List[str]) -> List[str]:
        """
        Filter gene columns in targets_csv to match provided gene list.
        
        Handles version numbers by splitting on '.' and comparing base IDs.
        For example, ENSG00000000003.15 will match ENSG00000000003.
        
        Args:
            genes: List of gene identifiers (with or without version numbers)
            
        Returns:
            List of matching column names from targets_csv
        """
        print(f"\nFiltering genes")
        # Get available gene columns (exclude ID and sample_id)
        available_columns = [c for c in self.targets_df.columns if c.startswith('ENSG')]
        
        # Create a set of provided genes (normalized: split on '.' to remove version)
        provided_genes_set = set()
        for gene in genes:
            # Handle both string gene names and potential version numbers
            base_gene = str(gene).split('.')[0]
            provided_genes_set.add(base_gene)
        print(f"  Attempting to match {len(provided_genes_set)} provided genes to {len(available_columns)} gene columns in targets_csv")

        # Match columns: check if column name (without version) matches any provided gene
        matched_genes = []
        for col in available_columns:
            base_col = str(col).split('.')[0]
            if base_col in provided_genes_set:
                matched_genes.append(col)
        
        # Report matching results
        if len(matched_genes) == 0:
            print(f"  Warning: No genes from provided list matched columns in targets_csv")
            print(f"    Provided genes (first 5): {list(genes)[:5]}")
            print(f"    Available columns (first 5): {available_columns[:5]}")
        else:
            print(f"  Matched {len(matched_genes)} out of {len(genes)} provided genes to columns in targets_csv")
            if len(matched_genes) < len(genes):
                # Find unmatched genes by comparing base names
                matched_base_names = {str(col).split('.')[0] for col in matched_genes}
                unmatched = [g for g in genes if str(g).split('.')[0] not in matched_base_names]
                print(f"    {len(unmatched)} genes not found in targets_csv (first 5): {unmatched[:5]}")
        
        return matched_genes
    
    def _build_target_matrix(self) -> np.ndarray:
        """Build (n_patients, n_genes) target matrix aligned to sample_ids."""
        print(f"\nBuilding target matrix")
        # Map each full sample_id to its truncated version, then look up target rows
        # Since multiple full IDs can map to the same truncated ID, we need to ensure
        # we get exactly one row per full sample_id
        
        # Create mapping: truncated_id -> target row index in targets_df
        truncated_to_target_idx = {trunc_id: idx for idx, trunc_id in enumerate(self.targets_df.index)}
        
        # Build target matrix: one row per full sample_id
        target_rows = []
        for full_sid in self.sample_ids:
            trunc_sid = full_sid[:15]
            if trunc_sid in truncated_to_target_idx:
                target_rows.append(truncated_to_target_idx[trunc_sid])
            else:
                raise ValueError(f"Truncated sample ID {trunc_sid} (from {full_sid}) not found in targets_df")
        
        # Extract target values using the row indices.
        # Guard against accidental metadata/string columns by coercing and validating.
        target_frame = self.targets_df.iloc[target_rows][self.genes]
        target_frame_numeric = target_frame.apply(pd.to_numeric, errors='coerce')
        if target_frame_numeric.isna().any().any():
            bad_cols = target_frame_numeric.columns[target_frame_numeric.isna().any()].tolist()
            bad_preview = target_frame[bad_cols].head(1).to_dict(orient='records')
            raise ValueError(
                "Non-numeric values found in target columns. "
                f"Likely non-gene columns were selected. Bad columns: {bad_cols[:5]}; "
                f"preview: {bad_preview}"
            )
        targets = target_frame_numeric.values.astype(np.float32)
        
        if self.log_transform:
            targets = np.log10(1 + targets)
        
        return targets
    
    def _get_feature_dim(self) -> int:
        """Infer feature dimension from first sample."""
        first_study = self.sample_ids[0]
        if self.aggregated:
            # Check if cache exists (loaded after alignment)
            if hasattr(self, 'aggregated_features_cache') and first_study in self.aggregated_features_cache:
                # Cache is loaded, get feature_dim from cached tensor (already processed)
                cached_tensor = self.aggregated_features_cache[first_study]
                return cached_tensor.shape[1]
            elif self.aggregated_h5 is not None:
                # H5 file still open, read from it (shouldn't happen in normal flow)
                data = self.aggregated_h5['X'][self.feature_files[first_study], :]
                data = data[:, 3:]
                return data.shape[1]
            else:
                # Should not happen, but fallback to cache if H5 is closed
                cached_tensor = self.aggregated_features_cache[first_study]
                return cached_tensor.shape[1]
        else:
            data = np.load(self.feature_files[first_study])
            data = data[:, 3:]
            return data.shape[1]
    
    def _build_sample_groups(self) -> Tuple[List[List[str]], List[str]]:
        """
        Group samples by truncated sample ID to prevent pair leakage.
        Pairs are defined as samples with the same truncated ID (first 15 chars).
        
        Returns:
            groups: List of lists, each inner list contains sample_ids in that group
            group_labels: Project.ID for each group (for stratification)
        """
        from collections import Counter
        
        groups = []
        group_labels = []
        
        # Build mapping: truncated_id -> list of full sample_ids
        truncated_to_full = {}
        for full_sid in self.sample_ids:
            trunc_sid = full_sid[:15]
            if trunc_sid not in truncated_to_full:
                truncated_to_full[trunc_sid] = []
            truncated_to_full[trunc_sid].append(full_sid)
        
        # Create groups and get project labels
        # self.projects is a Series indexed by iloc position, so we need to map
        sample_id_to_idx = {sid: idx for idx, sid in enumerate(self.sample_ids)}
        
        for trunc_sid, full_ids in truncated_to_full.items():
            groups.append(full_ids)
            # Get project from first member of the group
            first_idx = sample_id_to_idx[full_ids[0]]
            group_labels.append(self.projects.iloc[first_idx])
        
        # Print distribution of stratification labels
        label_counts = Counter(group_labels)
        n_singleton = sum(1 for g in groups if len(g) == 1)
        n_paired = sum(1 for g in groups if len(g) > 1)
        avg_paired_size = sum(len(g) for g in groups if len(g) > 1) / n_paired if n_paired > 0 else 0
        
        print(f"\nBuilding sample groups for splitting")
        print(f"  Built {len(groups)} sample groups from {len(self.sample_ids)} samples")
        print(f"  Singleton groups: {n_singleton}")
        if n_paired > 0:
            print(f"  Paired groups (same truncated ID): {n_paired} with avg size {avg_paired_size:.2f}")
        else:
            print(f"  Paired groups (same truncated ID): {n_paired}")
        print(f"  Stratification by Project.ID: {dict(label_counts)}")
        
        return groups, group_labels
    
    def _groups_to_indices(self, group_indices: np.ndarray, groups: List[List[str]]) -> np.ndarray:
        """
        Convert group indices to dataset indices.
        
        Args:
            group_indices: Indices into the groups list
            groups: List of sample_id lists (from _build_sample_groups)
            
        Returns:
            Array of dataset indices
        """
        # Flatten selected groups to sample_ids
        selected_sample_ids = set()
        for gi in group_indices:
            selected_sample_ids.update(groups[gi])
        
        # Map sample_ids to dataset indices
        sample_id_to_idx = {sid: idx for idx, sid in enumerate(self.sample_ids)}
        return np.array([sample_id_to_idx[sid] for sid in selected_sample_ids if sid in sample_id_to_idx])
    
    def _safe_stratified_split(
        self, 
        indices: np.ndarray, 
        labels: np.ndarray, 
        test_size: float, 
        random_state: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Attempt stratified split, fall back to regular split if not possible.
        
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
            try:
                sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
                train_idx, test_idx = next(sss.split(indices, labels))
                return train_idx, test_idx
            except ValueError as e:
                print(f"    Warning: Stratified split failed ({e}), falling back to regular split")
        else:
            print(f"    Warning: Class '{unique[counts.argmin()]}' has only {min_count} sample(s), using regular split")
        
        # Fall back to regular (non-stratified) split
        ss = ShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        train_idx, test_idx = next(ss.split(indices))
        return train_idx, test_idx
    
    def _print_split_stats(self, train_idx: np.ndarray, valid_idx: np.ndarray, test_idx: np.ndarray):
        """Print Project.ID distribution for each split."""
        for name, idx in [('Train', train_idx), ('Valid', valid_idx), ('Test', test_idx)]:
            if len(idx) == 0:
                print(f"    {name}: empty")
                continue
            split_projects = self.projects.iloc[idx]
            counts = split_projects.value_counts()
            print(f"    {name} Project.ID split distribution: {dict(counts)}")
    
    def stratified_split(
        self, 
        test_size: float = 0.1, 
        valid_size: float = 0.1, 
        random_state: int = 42
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Single stratified split with pair protection.
        
        Pairs (same truncated sample ID) stay together in the same split.
        Stratifies by Project.ID to maintain ratios across splits.
        
        Args:
            test_size: Fraction for test set (default 0.1)
            valid_size: Fraction for validation set (default 0.1)
            random_state: Random seed for reproducibility
            
        Returns:
            train_idx, valid_idx, test_idx as numpy arrays of dataset indices
        """
        print(f"\nCreating stratified split (test={test_size}, valid={valid_size})")
        groups, group_labels = self._build_sample_groups()
        group_labels = np.array(group_labels)
        n_groups = len(groups)
        group_indices = np.arange(n_groups)
        
        # First split: separate test set
        trainval_rel_idx, test_rel_idx = self._safe_stratified_split(
            group_indices, group_labels, test_size, random_state)
        trainval_group_idx = group_indices[trainval_rel_idx]
        test_group_idx = group_indices[test_rel_idx]
        
        # Second split: separate validation from training
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
        
        print(f"  Split result: {len(train_idx)} train, {len(valid_idx)} valid, {len(test_idx)} test")
        self._print_split_stats(train_idx, valid_idx, test_idx)
        
        return train_idx, valid_idx, test_idx
    
    def stratified_kfold(
        self,
        n_splits: int = 5,
        valid_size: float = 0.1,
        random_state: int = 42
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
        """
        K-fold cross-validation with stratification and pair protection.
        
        Pairs (same truncated sample ID) stay together in the same split.
        Stratifies by Project.ID to maintain ratios across folds.
        
        Args:
            n_splits: Number of folds (default 5)
            valid_size: Fraction of training set for validation (default 0.1)
            random_state: Random seed for reproducibility
            
        Returns:
            train_idx: List of n_splits arrays of training indices
            valid_idx: List of n_splits arrays of validation indices
            test_idx: List of n_splits arrays of test indices
        """
        print(f"\nCreating {n_splits}-fold stratified cross-validation (valid_size={valid_size})")
        groups, group_labels = self._build_sample_groups()
        group_labels = np.array(group_labels)
        n_groups = len(groups)
        group_indices = np.arange(n_groups)
        
        # Use StratifiedKFold on groups
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        
        train_idx = []
        valid_idx = []
        test_idx = []
        
        print(f"\n  Fold splits:")
        for k, (trainval_group_idx, test_group_idx) in enumerate(skf.split(group_indices, group_labels)):
            # Test set for this fold
            fold_test_idx = self._groups_to_indices(test_group_idx, groups)
            test_idx.append(fold_test_idx)
            
            # Split trainval into train and valid
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
            
            print(f"    Fold {k}: {len(fold_train_idx)} train, {len(fold_valid_idx)} valid, {len(fold_test_idx)} test")
        
        # Print stats for first fold as example
        print(f"\n  Fold 0 Project.ID distribution:")
        self._print_split_stats(train_idx[0], valid_idx[0], test_idx[0])
        
        return train_idx, valid_idx, test_idx

    def __len__(self) -> int:
        return len(self.sample_ids)
    
    def __getitem__(self, idx: int):
        
        sample_id = self.sample_ids[idx]

        features = self._load_features(sample_id)
        

        targets = torch.from_numpy(self.targets[idx])

        return features, targets
    
    # === Utility properties for model training ===
    

    @property
    def patients(self) -> np.ndarray:
        """For compatibility with patient_kfold and other split functions."""
        return np.array(self.sample_ids)
    
    @property
    def dim(self) -> int:
        """Feature dimension for model initialization."""
        return self.feature_dim


# === Optional: collate function if you need metadata ===

def tcga_collate_with_ids(batch):
    """Collate that also returns sample IDs (if needed for analysis)."""
    features = torch.stack([b[0] for b in batch])
    targets = torch.stack([b[1] for b in batch])
    sample_ids = [b[2] for b in batch]
    return features, targets, sample_ids


def match_patient_single(dataset, split) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Match a single train/valid/test split to dataset indices.
    
    Args:
        dataset: TCGADataset instance
        split: List/tuple of 3 arrays [train_patients, valid_patients, test_patients]
               Each array contains patient IDs (can be 12-char, 15-char, or full-length)
        
    Returns:
        train_idx, valid_idx, test_idx as numpy arrays of dataset indices
    """
    train_patients, valid_patients, test_patients = split
    
    print(f"\nMatching single split to dataset")
    print(f"  Split sizes: {len(train_patients)} train, {len(valid_patients)} valid, {len(test_patients)} test")
    
    # Detect format by checking length of first patient ID
    sample_patient_ids = []
    if len(train_patients) > 0:
        sample_patient_ids.append(str(train_patients[0]))
    if len(valid_patients) > 0:
        sample_patient_ids.append(str(valid_patients[0]))
    if len(test_patients) > 0:
        sample_patient_ids.append(str(test_patients[0]))
    
    if not sample_patient_ids:
        raise ValueError("No patient IDs found in split")
    
    # Determine truncation length based on patient ID format
    sample_length = len(sample_patient_ids[0])
    if sample_length == 12:
        trunc_len = 12
        print(f"  Detected old format: 12-char truncated patient IDs")
    elif sample_length >= 15:
        trunc_len = 15
        print(f"  Detected new format: full-length or 15-char truncated sample IDs")
    else:
        trunc_len = None
        print(f"  Detected format: {sample_length}-char patient IDs, will try flexible matching")
    
    # Build mappings for both 12-char and 15-char truncations (and full IDs)
    truncated_to_indices_12 = {}
    truncated_to_indices_15 = {}
    full_id_to_indices = {}
    
    for idx, full_sid in enumerate(dataset.sample_ids):
        # Map for 12-char truncation
        trunc_12 = full_sid[:12]
        if trunc_12 not in truncated_to_indices_12:
            truncated_to_indices_12[trunc_12] = []
        truncated_to_indices_12[trunc_12].append(idx)
        
        # Map for 15-char truncation
        trunc_15 = full_sid[:15]
        if trunc_15 not in truncated_to_indices_15:
            truncated_to_indices_15[trunc_15] = []
        truncated_to_indices_15[trunc_15].append(idx)
        
        # Map for full ID
        full_id_to_indices[full_sid] = [idx]
    
    def match_patient_id(patient_id_str):
        """Match a patient ID using multiple strategies."""
        patient_id_str = str(patient_id_str)
        
        # Strategy 1: Try exact match (for full IDs)
        if patient_id_str in full_id_to_indices:
            return full_id_to_indices[patient_id_str]
        
        # Strategy 2: Try 15-char truncation match
        if len(patient_id_str) >= 15:
            trunc_15 = patient_id_str[:15]
            if trunc_15 in truncated_to_indices_15:
                return truncated_to_indices_15[trunc_15]
        
        # Strategy 3: Try 12-char truncation match
        if len(patient_id_str) >= 12:
            trunc_12 = patient_id_str[:12]
            if trunc_12 in truncated_to_indices_12:
                return truncated_to_indices_12[trunc_12]
        
        # Strategy 4: If saved ID is shorter, try truncating dataset IDs to match
        if trunc_len == 12 and len(patient_id_str) == 12:
            if patient_id_str in truncated_to_indices_12:
                return truncated_to_indices_12[patient_id_str]
        
        return []
    
    # Deduplicate patient IDs, then match to get all full sample IDs
    unique_train_patients = np.unique([str(pid) for pid in train_patients])
    unique_valid_patients = np.unique([str(pid) for pid in valid_patients])
    unique_test_patients = np.unique([str(pid) for pid in test_patients])
    
    # Match train patients
    train_idx = []
    for patient_id in unique_train_patients:
        train_idx.extend(match_patient_id(patient_id))
    
    # Match valid patients
    valid_idx = []
    for patient_id in unique_valid_patients:
        valid_idx.extend(match_patient_id(patient_id))
    
    # Match test patients
    test_idx = []
    for patient_id in unique_test_patients:
        test_idx.extend(match_patient_id(patient_id))
    
    train_idx = np.array(train_idx, dtype=int)
    valid_idx = np.array(valid_idx, dtype=int)
    test_idx = np.array(test_idx, dtype=int)
    
    print(f"  Matched: {len(train_idx)} train, {len(valid_idx)} valid, {len(test_idx)} test")
    
    # Verify no overlap between splits
    train_set = set(train_idx)
    valid_set = set(valid_idx)
    test_set = set(test_idx)
    
    assert len(train_set & valid_set) == 0, "train/valid overlap detected"
    assert len(train_set & test_set) == 0, "train/test overlap detected"
    assert len(valid_set & test_set) == 0, "valid/test overlap detected"
    
    print(f"  Split integrity verified (no overlap between train/valid/test)")
    
    return train_idx, valid_idx, test_idx


def match_patient_kfold(dataset, splits_path: str) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    """
    Recover previously saved patient splits for cross-validation.
    
    Loads a pickle file containing splits and matches them to the current dataset.
    Supports two formats:
    1. Old format: 12-char truncated patient IDs (e.g., 'TCGA-06-0157')
    2. New format: Full-length sample IDs (e.g., 'TCGA-06-0157-01A') or 15-char truncated
    
    The pickle file should contain a list of 3 elements: [train_splits, valid_splits, test_splits],
    where each element is a list of n_folds arrays containing patient IDs.
    
    Args:
        dataset: TCGADataset instance
        splits_path: Path to pickle file with saved splits
        
    Returns:
        train_idx: List of n_splits arrays of training indices
        valid_idx: List of n_splits arrays of validation indices  
        test_idx: List of n_splits arrays of test indices
    """
    # If a tuple/list of patient splits is provided directly, skip loading from path
    if isinstance(splits_path, (tuple, list)) and len(splits_path) == 3:
        train_patients_list, valid_patients_list, test_patients_list = splits_path
        n_folds = len(train_patients_list)
        print(f"\nUsing provided patient splits ({n_folds} folds)")
    else:
        import pickle
        print(f"\nLoading patient splits from {splits_path}")
        with open(splits_path, 'rb') as f:
            saved_splits = pickle.load(f)
        # saved_splits structure: [train_splits, valid_splits, test_splits]
        # Each is a list of n_folds arrays containing patient IDs
        train_patients_list, valid_patients_list, test_patients_list = saved_splits
        n_folds = len(train_patients_list)
    print(f"  Loaded {n_folds} folds from saved splits")
    
    # Detect format by checking length of first patient ID
    # Sample a few patient IDs from different folds to determine format
    sample_patient_ids = []
    for k in range(min(3, n_folds)):
        if len(train_patients_list[k]) > 0:
            sample_patient_ids.append(str(train_patients_list[k][0]))
        if len(valid_patients_list[k]) > 0:
            sample_patient_ids.append(str(valid_patients_list[k][0]))
        if len(test_patients_list[k]) > 0:
            sample_patient_ids.append(str(test_patients_list[k][0]))
    
    if not sample_patient_ids:
        raise ValueError("No patient IDs found in saved splits")
    
    # Determine truncation length based on saved patient ID format
    sample_length = len(sample_patient_ids[0])
    if sample_length == 12:
        trunc_len = 12
        print(f"  Detected old format: 12-char truncated patient IDs")
    elif sample_length >= 15:
        trunc_len = 15
        print(f"  Detected new format: full-length or 15-char truncated sample IDs")
    else:
        # Try to match as-is first, then try common truncation lengths
        trunc_len = None
        print(f"  Detected format: {sample_length}-char patient IDs, will try flexible matching")
    
    # Build mappings for both 12-char and 15-char truncations (and full IDs)
    # This allows matching regardless of saved format
    truncated_to_indices_12 = {}
    truncated_to_indices_15 = {}
    full_id_to_indices = {}
    
    for idx, full_sid in enumerate(dataset.sample_ids):
        # Map for 12-char truncation
        trunc_12 = full_sid[:12]
        if trunc_12 not in truncated_to_indices_12:
            truncated_to_indices_12[trunc_12] = []
        truncated_to_indices_12[trunc_12].append(idx)
        
        # Map for 15-char truncation
        trunc_15 = full_sid[:15]
        if trunc_15 not in truncated_to_indices_15:
            truncated_to_indices_15[trunc_15] = []
        truncated_to_indices_15[trunc_15].append(idx)
        
        # Map for full ID
        full_id_to_indices[full_sid] = [idx]
    
    def match_patient_id(patient_id_str, truncated_12, truncated_15, full_ids):
        """
        Match a patient ID using multiple strategies.
        
        For truncated IDs, returns ALL matching sample IDs (to include pairs).
        For full IDs, returns the exact match.
        """
        patient_id_str = str(patient_id_str)
        
        # Strategy 1: Try exact match (for full IDs) - returns single index
        if patient_id_str in full_ids:
            return full_ids[patient_id_str]
        
        # Strategy 2: Try 15-char truncation match - return all matches
        if len(patient_id_str) >= 15:
            trunc_15 = patient_id_str[:15]
            if trunc_15 in truncated_15:
                # Return all matching indices (includes pairs)
                return truncated_15[trunc_15]
        
        # Strategy 3: Try 12-char truncation match - return all matches
        if len(patient_id_str) >= 12:
            trunc_12 = patient_id_str[:12]
            if trunc_12 in truncated_12:
                # Return all matching indices (includes pairs)
                return truncated_12[trunc_12]
        
        # Strategy 4: If saved ID is shorter, try truncating dataset IDs to match
        if trunc_len == 12 and len(patient_id_str) == 12:
            if patient_id_str in truncated_12:
                return truncated_12[patient_id_str]
        
        return []
    
    train_idx = []
    valid_idx = []
    test_idx = []
    
    for k in range(n_folds):
        # Deduplicate patient IDs first, then match to get all full sample IDs
        # Convert to string and get unique values
        unique_train_patients = np.unique([str(pid) for pid in train_patients_list[k]])
        unique_valid_patients = np.unique([str(pid) for pid in valid_patients_list[k]])
        unique_test_patients = np.unique([str(pid) for pid in test_patients_list[k]])
        
        # Match train patients (deduplicated)
        fold_train_idx = []
        for patient_id in unique_train_patients:
            matched = match_patient_id(
                patient_id, 
                truncated_to_indices_12, 
                truncated_to_indices_15, 
                full_id_to_indices
            )
            fold_train_idx.extend(matched)
        
        # Match valid patients (deduplicated)
        fold_valid_idx = []
        for patient_id in unique_valid_patients:
            matched = match_patient_id(
                patient_id,
                truncated_to_indices_12,
                truncated_to_indices_15,
                full_id_to_indices
            )
            fold_valid_idx.extend(matched)
        
        # Match test patients (deduplicated)
        fold_test_idx = []
        for patient_id in unique_test_patients:
            matched = match_patient_id(
                patient_id,
                truncated_to_indices_12,
                truncated_to_indices_15,
                full_id_to_indices
            )
            fold_test_idx.extend(matched)
        
        train_idx.append(np.array(fold_train_idx, dtype=int))
        valid_idx.append(np.array(fold_valid_idx, dtype=int))
        test_idx.append(np.array(fold_test_idx, dtype=int))
        
        print(f"    Fold {k}: {len(fold_train_idx)} train, {len(fold_valid_idx)} valid, {len(fold_test_idx)} test")
    
    # Verify no overlap between splits
    for k in range(n_folds):
        train_set = set(train_idx[k])
        valid_set = set(valid_idx[k])
        test_set = set(test_idx[k])
        
        assert len(train_set & valid_set) == 0, f"Fold {k}: train/valid overlap"
        assert len(train_set & test_set) == 0, f"Fold {k}: train/test overlap"
        assert len(valid_set & test_set) == 0, f"Fold {k}: valid/test overlap"
    
    print(f"  Split integrity verified (no overlap between train/valid/test)")
    
    return train_idx, valid_idx, test_idx


if __name__ == "__main__":
    # features_dir = "/gpfs/work4/0/prjs1086/pepsi/data/processed/tcga_resnet_feats"
    features_aggregated = "/home/dvanerp/pepsi/data/processed/tcga_supertiles.h5"
    # targets_csv = "/home/dvanerp/pepsi/data/raw/tcga_rna/tcga_all_transcriptomes_tpm_10782.csv"
    targets_csv = "/home/dvanerp/pepsi/data/raw/tcga_rna/tcga_all_transcriptomes_tpm_10782_first5last5.csv"
    keyfile_path = "/home/dvanerp/pepsi/data/raw/manifests/new_keyfiles/new_master_keyfile.csv"
    genes = "ENSG00000000003.15,ENSG00000000005.6,ENSG00000000419.13"
    project_filter = None
    # project_filter = ["TCGA-BRCA"]
    split_file = "/home/dvanerp/projects/HE2RNA_code/patient_splits.pkl"

    if os.path.exists(genes):
        genes = pd.read_csv(genes)
        genes = genes['gene'].tolist() if 'gene' in genes.columns else genes.iloc[:,0].tolist()
    else:
        genes = genes.split(',')
    for gene in genes:
        assert gene.startswith('ENSG'), "Unknown gene format"
    dataset = TCGADataset(
        features_aggregated=features_aggregated,
        targets_csv=targets_csv,
        keyfile_path=keyfile_path,
        genes=genes,
        project_filter=project_filter,
        max_tiles=100
    )

    train_idx, valid_idx, test_idx = match_patient_kfold(dataset, split_file)

    # Compatible with your existing patient_kfold
    print(f"First 16 sample IDs: \n{dataset.patients[:16]}")
    print(f"Unique Projects:\n{dataset.projects.unique()}")

    # loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=4, collate_fn=cscc_collate_with_ids)
    loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=4)
    train_subset = Subset(dataset, train_idx[0])
    train_loader = DataLoader(train_subset, batch_size=16, shuffle=False, num_workers=4)
    seen = set()

    for features, targets, sample_ids in train_loader:
        # features: (B, max_tiles, feature_dim)
        # targets: (B, n_genes)
        # sample_id: (B,)
        # print(f"Sample features shape: {features.shape}")
        # print(f"Sample targets shape: {targets.shape}")
        # print(f"Sample sample_id shape: {len(sample_ids)}")
        # print(f"Study numbers:\n {sample_ids}")
        # print(f"Patient 0: {sample_ids[0]};\n  features mean: {features[0].mean()};\n features std: {features[0].std()};\n  targets mean: {targets[0].mean()};\n targets std: {targets[0].std()}")
        print(f"Sample IDs first train loader batch: \n{sample_ids}")
        exit()
        for sample_id in sample_ids:
            if sample_id in seen:
                print(f"Warning: Duplicate study number: {sample_id}")
            seen.add(sample_id)
    print(f"Seen {len(seen)} study numbers")