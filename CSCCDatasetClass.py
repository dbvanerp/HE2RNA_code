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


class CSCCDataset(Dataset):
    """
    PyTorch Dataset for CSCC multi-instance learning.
    
    Features: H5 files per patient (filename = study_number.h5) OR aggregated H5 file
    Targets: Bulk RNA-seq from CSV
    Linking: keyfile maps skylinedx_id -> study_number
    
    Args:
        features_dir: Directory with H5 files (one per patient). Either this or features_aggregated must be provided.
        features_aggregated: Path to aggregated H5 file containing all slides. Either this or features_dir must be provided.
        targets_csv: CSV with RNA counts (ID column = skylinedx_id)
        keyfile_paths: Comma-separated list of CSV paths mapping skylinedx_id_rsm2 -> study_number, QC status, CP Score etc
        genes: Optional list of gene columns to use (None = all)
        max_tiles: Max tiles per patient for padding
        log_transform: Whether to log10(1+x) transform targets
    """
    
    def __init__(
        self,
        features_dir: Optional[str] = None,
        features_aggregated: Optional[str] = None,
        targets_csv: Optional[str] = None,
        keyfile_paths: Optional[str] = None,
        genes: Optional[List[str]] = None,
        max_tiles: int = 8000,
        log_transform: bool = True,
        project_column: str = 'metastasis',
        allow_missing_targets: bool = False
    ):
        print(f"\n{'-'*15} Initializing CSCCDataset {'-'*15}\n")
        
        self.features_dir = Path(features_dir) if features_dir is not None else None
        self.features_aggregated = features_aggregated
        self.max_tiles = max_tiles
        self.log_transform = log_transform
        self.project_column = project_column
        self.keyfile_paths = keyfile_paths.split(',') if keyfile_paths else []
        self.training = False  # Training mode flag for stochastic tile subsampling
        self.allow_missing_targets = bool(allow_missing_targets)
        self.has_targets = targets_csv is not None

        # Validate that at least one feature source is provided
        if self.features_dir is None and self.features_aggregated is None:
            raise ValueError("Either features_dir or features_aggregated must be provided")

        # Load keyfile for ID mapping (if provided)
        if self.keyfile_paths:
            self.id_to_study = self._load_keyfile(self.keyfile_paths)
        else:
            self.id_to_study = {}
        
        # Load features index based on mode
        if self.features_dir is not None:
            # Per-slide H5 files mode
            self.aggregated = False
            self.feature_files = self._index_features()
        elif self.features_aggregated is not None:
            # Aggregated H5 file mode
            self.aggregated = True
            self.aggregated_h5 = h5py.File(self.features_aggregated, 'r')
            # Build mapping: slide_name -> index in the dataset
            slide_names = self.aggregated_h5['slide_name'][:]
            # Convert bytes to strings if needed and create index mapping
            if isinstance(slide_names[0], bytes):
                self.feature_files = {k.decode(): idx for idx, k in enumerate(slide_names)}
            else:
                self.feature_files = {str(k): idx for idx, k in enumerate(slide_names)}
            print(f"Indexed {len(self.feature_files)} samples from aggregated H5 file")
        
        if self.has_targets:
            # Load targets and map to study_numbers
            self.targets_df, self.target_study_numbers = self._load_targets(targets_csv)
            # Find intersection: patients with both features AND targets
            self.study_numbers = self._align_patients()
        else:
            if not self.allow_missing_targets:
                raise ValueError(
                    "targets_csv is required unless allow_missing_targets=True."
                )
            self.targets_df = None
            self.target_study_numbers = []
            self.study_numbers = self._align_patients_without_targets()
        
        # For aggregated case, load features into memory to avoid I/O bottlenecks
        if self.aggregated:
            self._load_aggregated_features_to_memory()
        
        # Select genes
        if genes is None:
            if not self.has_targets:
                raise ValueError(
                    "genes must be provided when targets_csv is missing."
                )
            # Auto-detect gene columns (adjust pattern as needed)
            self.genes = [c for c in self.targets_df.columns if c not in ['ID', 'study_number']]
            print(f"No gene filter provided. Auto-detected {len(self.genes)} genes")
        else:
            self.genes = self._filter_genes_by_list(genes) if self.has_targets else list(genes)
        
        # Build aligned target matrix
        if self.has_targets:
            self.targets = self._build_target_matrix()
        else:
            self.targets = np.zeros((len(self.study_numbers), len(self.genes)), dtype=np.float32)
            print(
                f"No transcriptome provided. Using zero dummy targets with shape {self.targets.shape} "
                "for inference-only compatibility."
            )
        
        # Infer feature dimension
        self.feature_dim = self._get_feature_dim()
        
        if self.keyfile_paths:
            self.metadata = self._load_metadata(self.keyfile_paths[0])
        else:
            self.metadata = None

        print(f"\n{'-'*15} CSCCDataset initialized: {len(self)} patients, {len(self.genes)} genes, {self.feature_dim}D features {'-'*15}\n")
        print(f"Full summary of the dataset:")
        summarize_class(self)
    
    def _load_keyfile(self, keyfile_paths: List[str]) -> dict:
        """Load keyfile and filter by QC status."""
        print(f"\nFiltering patients by QC status")
        for keyfile_path in keyfile_paths:
            if 'rsm2' in keyfile_path:
                keyfile_rsm2 = pd.read_csv(keyfile_path)
            elif 'DSQUAME' in keyfile_path:
                keyfile_DSQUAME = pd.read_csv(keyfile_path)
            else:
                raise ValueError(f"Invalid keyfile path, expecting 'rsm2' or 'DSQUAME' in the paths: {keyfile_path}")

        original_set = set(keyfile_rsm2['sample_id'])
        # Only include rows where 'CP Score' is not NA/nan
        keyfile_DSQUAME = keyfile_DSQUAME[keyfile_DSQUAME['CP_score'].notna()]
        print(f"  Found {len(keyfile_DSQUAME)} rows with non-NA CP Score in DSQUAME keyfile after filtering out {len(keyfile_DSQUAME[keyfile_DSQUAME['CP_score'].isna()])} rows with NA CP Score")
        # Build one big list where each Sample_id is split on '+' if present, else just the id
        all_rsm1_ids = []
        for sid in keyfile_DSQUAME['Sample_id']:
            if '+' in str(sid):
                all_rsm1_ids.extend([s.strip() for s in str(sid).split('+')])
            else:
                all_rsm1_ids.append(str(sid).strip())
        print(f"  Found {len(all_rsm1_ids)} split Sample_ids in DSQUAME keyfile (flattened list)")

        # TODO I comment out the DSQUAME keyfile filtering for now, until CP Score is available for all patients in the DSQUAME keyfile.
        # Keep only rows of keyfile_rsm2 that have 'sample_id' entry in all_rsm1_ids
        # keyfile_rsm2 = keyfile_rsm2[keyfile_rsm2['sample_id'].astype(str).isin(all_rsm1_ids)]
        filtered_set = set(keyfile_rsm2['sample_id'])
        print(f"  Filtered out {len(original_set - filtered_set)} patients due to missing (split) Sample_id in DSQUAME keyfile. Remaining: {len(filtered_set)}")
        print(f"  All missing Sample_ids: {original_set - filtered_set}")
        print(f"  DSQUAME keyfile filtering commented out for now, until CP Score is available for all patients in the DSQUAME keyfile.")

        
        if 'rna_qc_status' in keyfile_rsm2.columns:
            initial_count = len(keyfile_rsm2)
            # Normalize to lowercase and strip whitespace to be safe
            # Handle potential NaN values by converting to string first
            keyfile_rsm2 = keyfile_rsm2[keyfile_rsm2['rna_qc_status'].astype(str).str.lower().str.strip() != 'fail']
            filtered_count = len(keyfile_rsm2)
            print(f"  Filtered out {initial_count - filtered_count} patients due to QC failure. Remaining: {filtered_count}")
        else:
            print("  Warning: 'rna_qc_status' column not found in keyfile. Skipping QC filtering.")
            
        return dict(zip(keyfile_rsm2['skylinedx_id_rsm2'], keyfile_rsm2['study_number']))

    def _index_features(self) -> dict:
        """Map study_number -> h5 file path."""
        print(f"\nIndexing features from {self.features_dir}")
        feature_files = {}
        for f in os.listdir(self.features_dir):
            if f.endswith('.h5'):
                study_number = os.path.splitext(f)[0]
                feature_files[study_number] = self.features_dir / f
        print(f"  Found {len(feature_files)} H5 feature files")
        return feature_files

    def _load_aggregated_features_to_memory(self):
        """Load all aggregated features into memory with pre-processing (padding/random selection) for fast access during training."""
        print(f"\nLoading aggregated features into memory...")
        print(f"  Loading {len(self.study_numbers)} samples from H5 file")
        
        # Check initial memory
        used_gb, available_gb, total_gb = _get_memory_usage()
        if used_gb is not None:
            print(f"  Initial memory: {used_gb:.2f} GB used, {available_gb:.2f} GB available, {total_gb:.2f} GB total")
        
        # First, get feature_dim from first sample
        first_study = self.study_numbers[0]
        first_h5_idx = self.feature_files[first_study]
        first_data = self.aggregated_h5['X'][first_h5_idx, :]
        first_data = first_data[:, 3:]  # Remove first 3 columns (metadata)
        feature_dim = first_data.shape[1]
        
        # Estimate memory per sample (max_tiles x feature_dim x 4 bytes for float32)
        estimated_mb_per_sample = (self.max_tiles * feature_dim * 4) / (1024 * 1024)
        estimated_total_gb = (estimated_mb_per_sample * len(self.study_numbers)) / 1024
        print(f"  Estimated memory per sample: {estimated_mb_per_sample:.2f} MB")
        print(f"  Estimated total cache size: {estimated_total_gb:.2f} GB")
        
        self.aggregated_features_cache = {}
        rng = np.random.default_rng(seed=42)
        
        # Memory check intervals: every 1000 samples or every 10GB estimated
        check_interval = max(100, min(1000, len(self.study_numbers) // 20))  # Check ~20 times during loading
        last_check_idx = 0
        last_check_memory = used_gb
        
        # Load and pre-process features for all aligned study_numbers
        for idx, study_number in enumerate(tqdm(self.study_numbers, desc="  Loading & processing features")):
            if study_number in self.feature_files:
                h5_idx = self.feature_files[study_number]
                # Load from H5
                data = self.aggregated_h5['X'][h5_idx, :]
                data = data[:, 3:]  # Remove first 3 columns (metadata)
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
                self.aggregated_features_cache[study_number] = torch.from_numpy(data.astype(np.float32))
            
            # Periodic memory check
            if (idx + 1) % check_interval == 0 or idx == len(self.study_numbers) - 1:
                used_gb, available_gb, total_gb = _get_memory_usage()
                if used_gb is not None:
                    samples_loaded = idx + 1
                    memory_delta = used_gb - last_check_memory if last_check_memory is not None else None
                    cache_memory_mb = sum(
                        tensor.element_size() * tensor.nelement() / (1024 * 1024) 
                        for tensor in self.aggregated_features_cache.values()
                    )
                    cache_memory_gb = cache_memory_mb / 1024
                    
                    print(f"  Progress: {samples_loaded}/{len(self.study_numbers)} samples loaded")
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

    def _load_targets(self, targets_csv: str):
        """
        Load and transpose targets CSV so that columns are genes and rows are patients (samples),
        then map IDs to study_numbers.
        """
        print(f"\nLoading targets from {os.path.basename(targets_csv)}\nMight take a while...")
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
            print(f"  Warning: {len(missing_ids)} IDs from targets CSV not found in keyfile (or filtered by QC).")
            print(f"    First 5 missing: {missing_ids[:5]}...")
        
        df = df.iloc[valid_rows].reset_index(drop=True)
        df['study_number'] = study_numbers

        print(f"  Loaded {len(df)} targets with valid study_numbers after transpose fix")
        return df, study_numbers

    def _load_features(self, study_number: str) -> torch.Tensor:
        """Load and pad features for a patient. If there are more than max_tiles, pick a random subset.

        During training: uses torch.randperm without local seed for stochastic subsampling
        (different subset each oversampled call, controlled by global RNG).
        During evaluation: uses deterministic seed for reproducibility.

        Returns tensor of shape (max_tiles, feature_dim).
        """
        if self.aggregated:
            # Aggregated mode: load from pre-processed cache
            # Cache stores features already padded to (max_tiles, feature_dim)
            return self.aggregated_features_cache[study_number]
        else:
            # Per-slide H5 mode: load from individual H5 file
            h5_path = self.feature_files[study_number]
            with h5py.File(h5_path, 'r') as f:
                data = f['features'][:]

            n_tiles = data.shape[0]

            if n_tiles > self.max_tiles:
                if self.training:
                    # Training: stochastic subsampling using global PyTorch RNG
                    # Each oversampled call gets a different subset, controlled by DataLoader's worker seed
                    indices = torch.randperm(n_tiles)[:self.max_tiles]
                    indices = torch.sort(indices)[0]  # Sort for consistency
                else:
                    # Evaluation: deterministic subsampling for reproducibility
                    # Use a fixed seed based on patient index for consistency across runs
                    patient_idx = self.study_numbers.index(study_number)
                    g = torch.Generator().manual_seed(42 + patient_idx)
                    indices = torch.randperm(n_tiles, generator=g)[:self.max_tiles]
                    indices = torch.sort(indices)[0]
                data = data[indices.numpy(), :]
            elif n_tiles < self.max_tiles:
                padded = np.zeros((self.max_tiles, self.feature_dim), dtype=np.float32)
                padded[:n_tiles] = data
                data = padded
            # If n_tiles == self.max_tiles do nothing

            return torch.from_numpy(data.astype(np.float32))
    
    def _align_patients(self) -> List[str]:
        """Find patients with both features and targets."""
        print(f"\nAligning features and targets")
        has_features = set(self.feature_files.keys())
        has_targets = set(self.target_study_numbers)

        valid = has_features & has_targets
        missing_features = has_targets - has_features
        missing_targets = has_features - has_targets
        
        if missing_features:
            print(f"  Warning: {len(missing_features)} patients have targets but no features")
            print(f"    Missing feature examples: {list(missing_features)[:5]}...")
                
        if missing_targets:
            print(f"  Warning: {len(missing_targets)} patients have features but no targets")
            print(f"    Missing target examples: {list(missing_targets)[:10]}...")
        # Return in consistent order (sorted for reproducibility)
        study_numbers = sorted(list(valid))
        if len(study_numbers) == 0:
            raise ValueError("No feature samples found with targets after filtering")
        else:
            print(f"  Aligned {len(study_numbers)} patients with both features and targets")
        return study_numbers

    def _align_patients_without_targets(self) -> List[str]:
        """Find feature samples without requiring metadata (for true holdout inference)."""
        print("\nAligning feature samples for inference-only run (no transcriptome targets)")
        has_features = set(self.feature_files.keys())
        study_numbers = sorted(list(has_features))
        if len(study_numbers) == 0:
            raise ValueError("No feature samples found for inference-only run.")
        print(f"  Aligned {len(study_numbers)} feature samples for inference-only run")
        if self.keyfile_paths:
            metadata_df = self._load_metadata(self.keyfile_paths[0])
            has_metadata = set(metadata_df.index.astype(str))
            missing_metadata = has_features - has_metadata
            if missing_metadata:
                print(
                    f"  Note: {len(missing_metadata)} feature samples not found in metadata. "
                    "This is expected for true holdout samples without keyfile entries."
                )
                print(f"    Examples: {list(missing_metadata)[:5]}...")
        return study_numbers
    
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
        # Get available gene columns (exclude ID and study_number)
        available_columns = [c for c in self.targets_df.columns if c not in ['ID', 'study_number']]
        
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
        """Build (n_patients, n_genes) target matrix aligned to study_numbers."""
        print(f"\nBuilding target matrix")
        # Index targets_df by study_number for fast lookup
        self.targets_df = self.targets_df.set_index('study_number')
        
        targets = self.targets_df.loc[self.study_numbers, self.genes].values.astype(np.float32)
        
        if self.log_transform:
            targets = np.log10(1 + targets)
        
        return targets
    
    def _get_feature_dim(self) -> int:
        """Infer feature dimension from first sample."""
        first_study = self.study_numbers[0]
        if self.aggregated:
            # Aggregated mode: read from H5 file using index
            # Check if cache exists (loaded after alignment)
            if hasattr(self, 'aggregated_features_cache') and first_study in self.aggregated_features_cache:
                # Cache is loaded, get feature_dim from cached tensor (already processed)
                cached_tensor = self.aggregated_features_cache[first_study]
                return cached_tensor.shape[1]
            elif self.aggregated_h5 is not None:
                # H5 file still open, read from it
                h5_idx = self.feature_files[first_study]
                data = self.aggregated_h5['X'][h5_idx, :]
                data = data[:, 3:]  # Strip first 3 metadata columns
                return data.shape[1]
            else:
                # Fallback to cache if H5 is closed
                cached_tensor = self.aggregated_features_cache[first_study]
                return cached_tensor.shape[1]
        else:
            # Per-slide H5 mode
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
        Group samples by matching_set_id to prevent pair leakage.
        Samples with NaN or unique IDs become singleton groups.
        
        Returns:
            groups: List of lists, each inner list contains study_numbers in that group
            group_labels: Normalized tissue_type for each group (Biopsy/Excision/other)
        """
        import time
        t0 = time.time()
        
        groups = []
        group_labels = []
        
        # Get pair IDs and tissue types as numpy arrays for fast indexing
        # Use positional indexing instead of label-based .loc in loops
        pair_ids_series = self.metadata.loc[self.study_numbers, 'matching_set_id']
        tissue_types_series = self.metadata.loc[self.study_numbers, 'tissue_type']
        
        # Create position mapping from study_number to position
        study_to_pos = {s: i for i, s in enumerate(self.study_numbers)}
        pair_ids_arr = pair_ids_series.values
        tissue_types_arr = tissue_types_series.values
        
        # Build a mapping from pair_id to study_numbers using numpy where possible
        pair_to_studies = {}
        for i, study_num in enumerate(self.study_numbers):
            pair_id = pair_ids_arr[i]
            # Check if pair_id is NaN or missing
            if pd.isna(pair_id):
                # Singleton group - no pair
                groups.append([study_num])
                raw_tissue = tissue_types_arr[i]
                group_labels.append(self._normalize_tissue_type(raw_tissue))
            else:
                # Has a pair_id - group together
                if pair_id not in pair_to_studies:
                    pair_to_studies[pair_id] = []
                pair_to_studies[pair_id].append(study_num)
        
        # Add paired groups (those with the same pair_id)
        for pair_id, study_list in pair_to_studies.items():
            groups.append(study_list)
            # Use first member's tissue_type for the group label (normalized)
            # Find position of first study number in our array
            first_study_pos = study_to_pos[study_list[0]]
            raw_tissue = tissue_types_arr[first_study_pos]
            group_labels.append(self._normalize_tissue_type(raw_tissue))
        
        # Print distribution of stratification labels
        from collections import Counter
        label_counts = Counter(group_labels)
        
        t1 = time.time()
        print(f"Built {len(groups)} sample groups from {len(self.study_numbers)} samples ({t1-t0:.3f}s)")
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
        
        # Map study_numbers to dataset indices using a dictionary for O(1) lookup
        # instead of np.isin which can be O(n*m) for large arrays
        study_to_idx = {s: i for i, s in enumerate(self.study_numbers)}
        result_indices = [study_to_idx[s] for s in selected_study_numbers]
        return np.array(result_indices, dtype=int)
    
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
        
        Pairs (same matching_set_id) stay together in the same split.
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
        # Convert to numpy array for faster indexing
        study_numbers_arr = np.array(self.study_numbers)
        # Get tissue types as numpy array indexed by position
        tissue_types_arr = self.metadata.loc[self.study_numbers, 'tissue_type'].values
        
        for name, idx in [('Train', train_idx), ('Valid', valid_idx), ('Test', test_idx)]:
            # Use numpy indexing which is O(1) per element vs pandas .loc list lookups
            split_tissues = tissue_types_arr[idx]
            unique, counts = np.unique(split_tissues, return_counts=True)
            pcts = np.round(counts / len(split_tissues) * 100, 1)
            dist_dict = {str(u): float(p) for u, p in zip(unique, pcts)}
            print(f"  {name} tissue_type distribution: {dist_dict}%")
    
    def stratified_kfold(
        self,
        n_splits: int = 5,
        valid_size: float = 0.1,
        random_state: int = 42
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
        """
        K-fold cross-validation with stratification and pair protection.
        
        Pairs (same matching_set_id) stay together in the same split.
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

        # Features from _load_features are (max_tiles, feature_dim);
        # transpose to (feature_dim, max_tiles) to match model expectations.
        features = self._load_features(study_number).transpose(0, 1)

        # Targets: (n_genes,)
        targets = torch.from_numpy(self.targets[idx])
        
        # For now return without study_number for compatibility 
        return features, targets
    
    # === Utility properties for model training ===
    
    @property
    def projects(self) -> pd.Series:
        """Return the projects in the dataset as a pandas Series, matched to study_number order (study_number is the index)."""
        if self.metadata is None:
            # For true holdout inference without metadata, return a Series of 'UNKNOWN' values
            return pd.Series(['UNKNOWN'] * len(self.study_numbers), index=self.study_numbers)
        return self.metadata.loc[self.study_numbers, self.project_column]
    
    @property
    def patients(self) -> np.ndarray:
        """For compatibility with patient_kfold and other split functions."""
        return np.array(self.study_numbers)
    
    @property
    def sample_ids(self) -> list:
        """For compatibility with match_patient_kfold (alias for study_numbers)."""
        return self.study_numbers
    
    @property
    def dim(self) -> int:
        """Feature dimension for model initialization."""
        return self.feature_dim

    def train(self, mode: bool = True):
        """Set training mode for stochastic tile subsampling.

        During training, slides with >max_tiles will use different random subsets
        each time they are loaded (data augmentation). During evaluation, the same
        deterministic subset is used for reproducibility.

        Args:
            mode: True for training mode (stochastic), False for eval mode (deterministic)
        """
        self.training = bool(mode)
        return self

    def eval(self):
        """Set evaluation mode (deterministic tile subsampling)."""
        return self.train(False)


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
    keyfile_paths = "/gpfs/work4/0/prjs1086/derm_shared/cscc/doc/keyfiles/20260113_project_keyfile_anonymized_rsm2.csv,/gpfs/work4/0/prjs1086/derm_shared/cscc/processed/cd/DSQUAME_development_NCC_dataset_clean_no_newlines.csv"
    genes = "/home/dvanerp/pepsi/data/processed/cscc_rna/bailey_dvp_signature_annotated.csv"


    if os.path.exists(genes):
        genes = pd.read_csv(genes)
        genes = genes['gene'].tolist() if 'gene' in genes.columns else genes.iloc[:,0].tolist()
    else:
        genes = genes.split(',')
    for gene in genes:
        # print(gene)
        assert gene.startswith('ENSG'), "Unknown gene format"
    dataset = CSCCDataset(
        features_dir=features_dir,
        targets_csv=targets_csv,
        keyfile_paths=keyfile_paths,
        max_tiles=8000,
        genes=genes
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