import math
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler


class MixedDataset(Dataset):
    """
    Concatenates CSCC and TCGA datasets and tracks sample domain.
    Domain labels: 0 = CSCC, 1 = TCGA.
    """

    def __init__(self, cscc_dataset, tcga_dataset):
        self.cscc_dataset = cscc_dataset
        self.tcga_dataset = tcga_dataset
        self.cscc_len = len(cscc_dataset)
        self.tcga_len = len(tcga_dataset)
        tcga_base = self._unwrap_base_dataset(tcga_dataset)
        cscc_base = self._unwrap_base_dataset(cscc_dataset)
        self._dim = int(tcga_base.dim)
        self._genes = list(tcga_base.genes)

        if int(cscc_base.dim) != self._dim:
            raise ValueError(
                f"Feature dim mismatch between CSCC ({cscc_base.dim}) and "
                f"TCGA ({tcga_base.dim})."
            )
        if len(cscc_base.genes) != len(self._genes):
            raise ValueError(
                f"Gene dim mismatch between CSCC ({len(cscc_base.genes)}) and "
                f"TCGA ({len(self._genes)})."
            )

        self.domain_labels = np.concatenate(
            [
                np.zeros(self.cscc_len, dtype=np.int64),
                np.ones(self.tcga_len, dtype=np.int64),
            ]
        )

    def _unwrap_base_dataset(self, dataset):
        base = dataset
        # torch.utils.data.Subset stores underlying dataset in `.dataset`.
        while hasattr(base, 'dataset'):
            base = base.dataset
        return base

    def __len__(self):
        return self.cscc_len + self.tcga_len

    def _normalize_feature_shape(self, x):
        """
        Normalize to (feature_dim, tiles) to match HE2RNA Conv1d input.
        """
        if not isinstance(x, torch.Tensor):
            x = torch.as_tensor(x)
        if x.ndim != 2:
            raise ValueError(f"Expected rank-2 feature tensor, got shape {tuple(x.shape)}")

        # If already (feature_dim, tiles), keep as-is.
        if int(x.shape[0]) == self._dim:
            return x
        # If (tiles, feature_dim), transpose.
        if int(x.shape[1]) == self._dim:
            return x.transpose(0, 1)
        raise ValueError(
            f"Cannot normalize feature shape {tuple(x.shape)} to dim={self._dim}."
        )

    def __getitem__(self, idx):
        if idx < self.cscc_len:
            x, y = self.cscc_dataset[idx]
            domain = 0
        else:
            x, y = self.tcga_dataset[idx - self.cscc_len]
            domain = 1

        x = self._normalize_feature_shape(x)
        return x, y, domain

    @property
    def dim(self):
        return self._dim

    @property
    def genes(self):
        return self._genes

    def train(self, mode=True):
        """Propagate train/eval mode to underlying datasets for tile subsampling."""
        cscc_base = self._unwrap_base_dataset(self.cscc_dataset)
        tcga_base = self._unwrap_base_dataset(self.tcga_dataset)
        if hasattr(cscc_base, 'train'):
            cscc_base.train(mode)
        if hasattr(tcga_base, 'train'):
            tcga_base.train(mode)
        return self

    def eval(self):
        return self.train(False)


class BalancedDomainBatchSampler(Sampler):
    """
    Produces balanced batches with a fixed CSCC:TCGA ratio (default 1:1).
    Sampling is with replacement when a domain is exhausted within an epoch.

    epoch_length controls how many batches constitute one epoch:
      - 'target' (default): driven by the target/smaller domain (CSCC).
          Each CSCC slide is seen ~1x per epoch; TCGA rotates across epochs.
      - 'anchor': driven by the anchor/larger domain (TCGA).
          Each TCGA slide is seen ~1x per epoch; CSCC is oversampled.
      - 'max': same as 'anchor' (backward compatible).
    """

    def __init__(self, domain_labels, batch_size, cscc_fraction=0.5,
                 seed=42, drop_last=False, epoch_length='target'):
        if batch_size < 2:
            raise ValueError("batch_size must be >= 2 for balanced sampling")
        self.domain_labels = np.asarray(domain_labels)
        self.batch_size = int(batch_size)
        self.cscc_fraction = float(cscc_fraction)
        self.seed = int(seed)
        self.drop_last = bool(drop_last)
        self.epoch = 0

        self.cscc_indices = np.where(self.domain_labels == 0)[0]
        self.tcga_indices = np.where(self.domain_labels == 1)[0]
        if len(self.cscc_indices) == 0 or len(self.tcga_indices) == 0:
            raise ValueError("BalancedDomainBatchSampler requires both CSCC and TCGA samples.")

        n_cscc = max(1, int(round(self.batch_size * self.cscc_fraction)))
        n_tcga = self.batch_size - n_cscc
        if n_tcga == 0:
            n_tcga = 1
            n_cscc = self.batch_size - 1
        self.n_cscc_per_batch = n_cscc
        self.n_tcga_per_batch = n_tcga

        cscc_batches = math.ceil(len(self.cscc_indices) / self.n_cscc_per_batch)
        tcga_batches = math.ceil(len(self.tcga_indices) / self.n_tcga_per_batch)

        epoch_length = str(epoch_length).lower()
        if epoch_length in ('target', 'min'):
            self.batches_per_epoch = cscc_batches
        elif epoch_length in ('anchor', 'max'):
            self.batches_per_epoch = max(cscc_batches, tcga_batches)
        else:
            raise ValueError(
                f"Unknown epoch_length '{epoch_length}'. "
                "Use 'target', 'anchor', 'min', or 'max'."
            )

        self._epoch_length_mode = epoch_length
        self.observed_counts = {"cscc": 0, "tcga": 0}
        print(
            f"BalancedDomainBatchSampler: {len(self.cscc_indices)} CSCC, "
            f"{len(self.tcga_indices)} TCGA | "
            f"{self.n_cscc_per_batch}+{self.n_tcga_per_batch}/batch | "
            f"epoch_length='{epoch_length}' -> {self.batches_per_epoch} batches/epoch"
        )

    def __len__(self):
        return self.batches_per_epoch

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        self.observed_counts = {"cscc": 0, "tcga": 0}

        # CSCC is always freshly shuffled each epoch (one full pass in 'target' mode).
        cscc_pool = rng.permutation(self.cscc_indices)
        cscc_ptr = 0

        # TCGA: in 'target' mode only a fraction is used per epoch, so we
        # use a persistent offset seeded by epoch to rotate through TCGA
        # across epochs rather than always starting from the same samples.
        tcga_rng = np.random.default_rng(self.seed + self.epoch * 7919)
        tcga_pool = tcga_rng.permutation(self.tcga_indices)
        tcga_ptr = 0

        for _ in range(self.batches_per_epoch):
            if cscc_ptr + self.n_cscc_per_batch > len(cscc_pool):
                cscc_pool = np.concatenate([cscc_pool[cscc_ptr:], rng.permutation(self.cscc_indices)])
                cscc_ptr = 0
            if tcga_ptr + self.n_tcga_per_batch > len(tcga_pool):
                tcga_pool = np.concatenate([tcga_pool[tcga_ptr:], tcga_rng.permutation(self.tcga_indices)])
                tcga_ptr = 0

            cscc_batch = cscc_pool[cscc_ptr: cscc_ptr + self.n_cscc_per_batch]
            tcga_batch = tcga_pool[tcga_ptr: tcga_ptr + self.n_tcga_per_batch]
            cscc_ptr += self.n_cscc_per_batch
            tcga_ptr += self.n_tcga_per_batch

            batch = np.concatenate([cscc_batch, tcga_batch])
            rng.shuffle(batch)
            self.observed_counts["cscc"] += len(cscc_batch)
            self.observed_counts["tcga"] += len(tcga_batch)
            yield batch.tolist()
