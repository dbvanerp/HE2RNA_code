"""
HE2RNA: Train a model to predict gene expression on TCGA slides, either on a single train/valid/test split or in cross-validation 
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
import configparser
import argparse
import pickle as pkl
import json
import random
import pandas as pd
import numpy as np
import copy as cp
from time import sleep
from datetime import datetime
from tqdm import tqdm
import h5py
import torch
from torch import nn
from torch.utils.data import Subset, DataLoader
from torch import optim
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, LinearLR, SequentialLR
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit, ShuffleSplit
from utils import compute_metrics, summarize_class, save_patient_splits, _plot_pred_vs_gt_scatter
# from transcriptome_data import TranscriptomeDataset
# from wsi_data import load_labels, AggregatedDataset, TCGAFolder, \
#     H5Dataset, patient_split, match_patient_split, \
#     patient_kfold, match_patient_kfold
from CSCCDatasetClass import CSCCDataset
from TCGADatasetClass import TCGADataset, match_patient_kfold, match_patient_single
from mixed_dataset import MixedDataset, BalancedDomainBatchSampler
from model import HE2RNA, fit, predict
# from constant import PATH_TO_TILES


class Experiment(object):
    """A class that uses a config file to setup and run a gene expression
    prediction experiment.

    Args:
        configfile (str): Path to the configuration file.
    """

    def __init__(self,
                 configfile='config.ini',
                 null_run=False,
                 inference=False):
        print(f"Initializing the Experiment class")
        # Read configuration file
        self.config = configparser.ConfigParser()
        self.config.read(configfile)
        
        # For debugging purposes, print the parsed config
        print("========== Loaded Config ==========")
        for section in self.config.sections():
            print(f"[{section}]")
            for key, value in self.config[section].items():
                print(f"{key}: {value}")
            print()
        print("===================================\n")
        
        assert 'main' in self.config.sections(), \
            "No 'main' section in config file"

        if 'path' in self.config['main'].keys():
            self.savedir = self.config['main']['path']
            if not os.path.exists(self.savedir):
                os.mkdir(self.savedir)
        else:
            self.savedir = '.'

        if 'use_saved_model' in self.config['main'].keys():
            self.use_saved_model = self.config['main']['use_saved_model']
            model_num = os.path.basename(self.use_saved_model.rstrip('/')).split('_')[-1] 
            self.fold = int(model_num) if model_num.isdigit() else None
            print(f"Using saved model from {self.use_saved_model} with fold {self.fold}")
        else:
            self.use_saved_model = False

        # Inference mode: require use_saved_model path and always use saved models
        self.inference = inference
        if 'use_saved_model' in self.config['main'].keys():
            self.use_saved_model = self.config['main']['use_saved_model']
        else:
            self.use_saved_model = None
        if self.inference:
            if not self.use_saved_model:
                raise ValueError("Inference mode requires [main] use_saved_model in the config.")
            if not os.path.exists(self.use_saved_model):
                raise FileNotFoundError(f"Inference use_saved_model path does not exist: {self.use_saved_model}")
            print(f"Inference mode enabled. Using saved models from {self.use_saved_model}")

        if 'single_split' in self.config['main'].keys():
            self.split = pkl.load(open(self.config['main']['single_split'], 'rb'))
        else:
            self.split = None
        if 'splits' in self.config['main'].keys():
            self.splits = pkl.load(open(self.config['main']['splits'], 'rb'))
        else:
            self.splits = None

        data_section = self.config['data'] if 'data' in self.config.sections() else {}
        self.has_transcriptome = (
            ('path_to_transcriptome' in data_section.keys())
            and bool(str(data_section['path_to_transcriptome']).strip())
        )
        self.true_holdout_no_transcriptome = (
            self.inference
            and (self.splits is None)
            and (self.split is None)
            and (not self.has_transcriptome)
        )
        if self.true_holdout_no_transcriptome:
            self._print_no_transcriptome_inference_warning()

        if 'subsample' in self.config['main'].keys():
            self.subsample = float(self.config['main']['subsample'])
        else:
            self.subsample = None

        # Data mode: 'tcga' (default), 'cscc', or mixed mode.
        if 'data_mode' in self.config['main'].keys():
            self.data_mode = self.config['main']['data_mode'].lower()
        elif 'cscc_mode' in self.config['main'].keys() and self.config['main']['cscc_mode'] == 'True':
            # Backwards compatibility with old cscc_mode config
            self.data_mode = 'cscc'
        else:
            self.data_mode = 'tcga'
        supported_modes = ['tcga', 'cscc', 'mixed_cscc_tcga_anchor']
        if self.data_mode not in supported_modes:
            raise ValueError(f"Unsupported data_mode '{self.data_mode}'. Expected one of {supported_modes}.")
        self.default_anchor_projects = [
            'TCGA-HNSC', 'TCGA-LUSC', 'TCGA-CESC', 'TCGA-ESCA'
        ]

        if 'p_value' in self.config['main'].keys():
            self.p_value = self.config['main']['p_value']
        else:
            self.p_value = 't_test'
        assert self.p_value in ['empirical', 't_test'], \
            "Unrecognized test, should be 'empirical' or 't_test'"

        self.null_run = null_run

    def _read_architecture(self):
        model_params = {}
        if 'architecture' in self.config.sections():
            dic = self.config['architecture']
            if 'layers' in dic.keys():
                layers = dic['layers'].split(',')
                model_params['layers'] = [int(dim) for dim in layers]
            if 'dropout' in dic.keys():
                model_params['dropout'] = float(dic['dropout'])
            if 'ks' in dic.keys():
                if 'proportional_ks' in dic.keys() and dic['proportional_ks'].strip().lower() == 'true':
                    model_params['proportional_ks'] = True
                else:
                    model_params['proportional_ks'] = False
                ks = dic['ks'].split(',')
                model_params['ks'] = [int(k) for k in ks]
            if 'nonlin' in dic.keys():
                if dic['nonlin'] == 'relu':
                    model_params['nonlin'] = nn.ReLU()
                elif dic['nonlin'] == 'tanh':
                    model_params['nonlin'] = nn.Tanh()
                elif dic['nonlin'] == 'sigmoid':
                    model_params['nonlin'] = nn.Sigmoid()
            if 'device' in dic.keys():
                model_params['device'] = dic['device']

        return model_params

    def _read_training_params(self):

        training_params = {}
        if 'training' in self.config.sections():
            dic = self.config['training']
            if 'max_epochs' in dic.keys():
                training_params['max_epochs'] = int(dic['max_epochs'])
            if 'patience' in dic.keys():
                training_params['patience'] = int(dic['patience'])
            if 'batch_size' in dic.keys():
                training_params['batch_size'] = int(dic['batch_size'])
            if 'num_workers' in dic.keys():
                training_params['num_workers'] = int(dic['num_workers'])
            if 'mixed_precision' in dic.keys():
                training_params['mixed_precision'] = dic['mixed_precision'].strip().lower() in ['1', 'true', 'yes', 'y']
            if 'mixed_precision_dtype' in dic.keys():
                training_params['mixed_precision_dtype'] = dic['mixed_precision_dtype'].strip().lower()

        return training_params

    def _read_loss_params(self):
        """Read loss function parameters from config."""
        loss_params = {}
        if 'loss' in self.config.sections():
            dic = self.config['loss']
            if 'loss_mode' in dic.keys():
                loss_params['loss_mode'] = dic['loss_mode']
            if 'mse_weight' in dic.keys():
                loss_params['mse_weight'] = float(dic['mse_weight'])
            if 'corr_weight' in dic.keys():
                loss_params['corr_weight'] = float(dic['corr_weight'])
        return loss_params

    def _read_bool(self, section, key, default=False):
        if section not in self.config.sections() or key not in self.config[section].keys():
            return default
        return self.config[section][key].strip().lower() in ['1', 'true', 'yes', 'y']

    def _read_float_list(self, section, key, default):
        if section not in self.config.sections() or key not in self.config[section].keys():
            return default
        values = [x.strip() for x in self.config[section][key].split(',') if x.strip()]
        return [float(x) for x in values]

    def _set_global_seeds(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _get_anchor_projects(self):
        dic = self.config['data']
        if 'tcga_project_filter' in dic.keys():
            raw = dic['tcga_project_filter']
            projects = [p.strip() for p in raw.split(',') if p.strip()]
            if len(projects) > 0:
                return projects
        return self.default_anchor_projects

    def _setup_scheduler(self, optimizer):
        """Create and return a learning rate scheduler based on config.

        Returns None if no scheduler is configured.
        """
        if 'scheduler' not in self.config.sections():
            return None

        cfg = self.config['scheduler']
        name = cfg.get('scheduler', 'none').lower()

        if name == 'none' or name == '':
            return None

        if name == 'cosine':
            T_max = int(cfg.get('T_max', '100'))
            eta_min = float(cfg.get('eta_min', '1e-6'))
            warmup_epochs = int(cfg.get('warmup_epochs', '0'))

            cosine = CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)

            if warmup_epochs > 0:
                warmup = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs)
                return SequentialLR(optimizer, [warmup, cosine], milestones=[warmup_epochs])
            return cosine

        if name == 'plateau':
            factor = float(cfg.get('factor', '0.5'))
            plateau_patience = int(cfg.get('plateau_patience', '10'))
            mode = cfg.get('mode', 'max')  # 'max' for correlation, 'min' for loss
            return ReduceLROnPlateau(optimizer, mode=mode, factor=factor, patience=plateau_patience)

        raise ValueError(f"Unknown scheduler: {name}. Supported: cosine, plateau, none")

    def _setup_optimization(self, model, override_lr=None):
        """Create optimizer and scheduler, returning them as a tuple."""
        if 'optimization' in self.config.sections():
            dic = self.config['optimization']
            base_lr = float(dic['lr']) if override_lr is None else float(override_lr)
            weight_decay = float(dic['weight_decay']) if 'weight_decay' in dic.keys() else 0.0
            optimizer_name = dic.get('optimizer', 'adam').lower()
            use_diff_lr = self._read_bool('optimization', 'differential_lr', default=False)

            optimizer = None

            if use_diff_lr and hasattr(model, 'layers') and len(model.layers) > 1:
                head_lr_mult = float(dic['head_lr_multiplier']) if 'head_lr_multiplier' in dic.keys() else 5.0
                base_params = []
                for layer in model.layers[:-1]:
                    base_params += list(layer.parameters())
                head_params = list(model.layers[-1].parameters())
                param_groups = [
                    {'params': base_params, 'lr': base_lr},
                    {'params': head_params, 'lr': base_lr * head_lr_mult},
                ]
                print(
                    f"Using differential LR parameter groups: base_lr={base_lr:.2e}, "
                    f"head_lr={base_lr * head_lr_mult:.2e}"
                )
                if optimizer_name == 'sgd':
                    momentum = float(dic['momentum']) if 'momentum' in dic.keys() else 0.9
                    optimizer = optim.SGD(param_groups, momentum=momentum, nesterov=True, weight_decay=weight_decay)
                elif optimizer_name == 'adam':
                    optimizer = optim.Adam(param_groups, weight_decay=weight_decay)
                elif optimizer_name == 'adamw':
                    optimizer = optim.AdamW(param_groups, weight_decay=weight_decay)
                else:
                    raise ValueError(f"Unsupported optimizer '{optimizer_name}'")
            else:
                optim_params = {'params': model.parameters(), 'lr': base_lr}
                if 'momentum' in dic.keys():
                    optim_params['momentum'] = float(dic['momentum'])
                    optim_params['nesterov'] = True
                if 'weight_decay' in dic.keys():
                    optim_params['weight_decay'] = weight_decay

                if optimizer_name == 'sgd':
                    optimizer = optim.SGD(**optim_params)
                elif optimizer_name == 'adam':
                    optimizer = optim.Adam(**optim_params)
                elif optimizer_name == 'adamw':
                    optimizer = optim.AdamW(**optim_params)
                else:
                    raise ValueError(f"Unsupported optimizer '{optimizer_name}'")

            # Create scheduler after optimizer is created
            scheduler = self._setup_scheduler(optimizer)
            if scheduler is not None:
                print(f"Using scheduler: {type(scheduler).__name__}")

            return optimizer, scheduler

        # Default: Adam optimizer, no scheduler
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        return optimizer, None

    def _load_saved_model_for_fold(self, fold_idx, model_params):
        if not self.use_saved_model:
            return None
        fold_path = os.path.join(self.use_saved_model, f'model_{fold_idx}', 'model.pt')
        base_path = os.path.join(self.use_saved_model, 'model.pt')
        model_path = fold_path if os.path.exists(fold_path) else base_path
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Saved model not found at fold path '{fold_path}' or base path '{base_path}'."
            )
        model = torch.load(model_path)
        if 'ks' in model_params.keys():
            model.ks = model_params['ks']
        if 'top_k' in model_params.keys():
            model.top_k = model_params['top_k']
        if 'bottom_ks' in model_params.keys():
            model.bottom_ks = model_params['bottom_ks']
        if 'dropout' in model_params.keys():
            model.do.p = model_params['dropout']
        if 'proportional_ks' in model_params.keys():
            model.proportional_ks = model_params['proportional_ks']
        else:
            model.proportional_ks = False
        return model

    def _initialize_model(self, model_params, train_set, fold_idx):
        if self.use_saved_model:
            return self._load_saved_model_for_fold(fold_idx, model_params)
        print("Initializing model without saved model")
        try:
            model_params['bias_init'] = torch.nn.Parameter(
                torch.Tensor(
                    np.mean([sample[1] for sample in train_set], axis=0)
                ).cuda()
            )
        except ValueError:
            model_params['bias_init'] = torch.nn.Parameter(
                torch.Tensor(
                    np.mean([sample[1].numpy() for sample in train_set], axis=0)
                ).cuda()
            )
        return HE2RNA(**model_params)

    def _safe_stratified_split_indices(self, n_items, labels, test_size, random_state):
        labels = np.asarray(labels)
        item_indices = np.arange(n_items)
        unique, counts = np.unique(labels, return_counts=True)
        min_count = counts.min()
        if min_count >= 2:
            try:
                sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
                train_idx, test_idx = next(sss.split(item_indices, labels))
                return train_idx, test_idx
            except ValueError as e:
                print(f"Warning: stratified split failed ({e}), falling back to random split.")
        else:
            print(
                f"Warning: class '{unique[counts.argmin()]}' has only {min_count} sample(s); "
                "falling back to random split."
            )
        ss = ShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        train_idx, test_idx = next(ss.split(item_indices))
        return train_idx, test_idx

    def _build_cscc_inner_folds(self, cscc_dataset, eligible_indices, n_splits, random_state):
        groups, group_labels = cscc_dataset._build_sample_groups()
        group_labels = np.asarray(group_labels)
        eligible_studies = set(cscc_dataset.patients[eligible_indices])
        study_to_idx = {sid: i for i, sid in enumerate(cscc_dataset.patients)}
        filtered_groups = []
        filtered_labels = []
        for g, lab in zip(groups, group_labels):
            if all(s in eligible_studies for s in g):
                filtered_groups.append(g)
                filtered_labels.append(lab)
        if len(filtered_groups) < n_splits:
            raise ValueError(
                f"Not enough CSCC groups ({len(filtered_groups)}) for {n_splits} inner folds."
            )
        filtered_labels = np.asarray(filtered_labels)
        fold_train_idx, fold_valid_idx = [], []
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        for train_groups_idx, valid_groups_idx in skf.split(np.arange(len(filtered_groups)), filtered_labels):
            tr = []
            va = []
            for gi in train_groups_idx:
                tr.extend([study_to_idx[s] for s in filtered_groups[gi]])
            for gi in valid_groups_idx:
                va.extend([study_to_idx[s] for s in filtered_groups[gi]])
            fold_train_idx.append(np.array(sorted(set(tr)), dtype=int))
            fold_valid_idx.append(np.array(sorted(set(va)), dtype=int))
        return fold_train_idx, fold_valid_idx

    def _build_cscc_train_valid_split(self, cscc_dataset, eligible_indices, valid_size, random_state):
        groups, group_labels = cscc_dataset._build_sample_groups()
        group_labels = np.asarray(group_labels)
        eligible_studies = set(cscc_dataset.patients[eligible_indices])
        study_to_idx = {sid: i for i, sid in enumerate(cscc_dataset.patients)}
        filtered_groups = []
        filtered_labels = []
        for g, lab in zip(groups, group_labels):
            if all(s in eligible_studies for s in g):
                filtered_groups.append(g)
                filtered_labels.append(lab)
        group_indices = np.arange(len(filtered_groups))
        tr_rel, va_rel = self._safe_stratified_split_indices(
            n_items=len(group_indices),
            labels=np.asarray(filtered_labels),
            test_size=valid_size,
            random_state=random_state
        )
        tr = []
        va = []
        for gi in group_indices[tr_rel]:
            tr.extend([study_to_idx[s] for s in filtered_groups[gi]])
        for gi in group_indices[va_rel]:
            va.extend([study_to_idx[s] for s in filtered_groups[gi]])
        return np.array(sorted(set(tr)), dtype=int), np.array(sorted(set(va)), dtype=int)

    def _build_mixed_train_loader(self, cscc_dataset, tcga_dataset, cscc_indices, batch_size, num_workers, seed):
        cscc_subset = Subset(cscc_dataset, cscc_indices)
        tcga_subset = Subset(tcga_dataset, np.arange(len(tcga_dataset)))
        mixed_train_set = MixedDataset(cscc_subset, tcga_subset)
        # Set training mode for stochastic tile subsampling (different tiles each epoch)
        mixed_train_set.train()
        epoch_length = self.config.get('training', 'epoch_length', fallback='target')
        sampler = BalancedDomainBatchSampler(
            domain_labels=mixed_train_set.domain_labels,
            batch_size=batch_size,
            cscc_fraction=0.5,
            seed=seed,
            drop_last=False,
            epoch_length=epoch_length
        )
        # Allow workers by default; only force 0 for high-risk configurations
        # unless user explicitly overrides this behavior.
        allow_risky_workers = self._read_bool('training', 'allow_mixed_multiworker', default=False)
        tcga_aggregated = bool(getattr(tcga_dataset, 'aggregated', False))
        if tcga_aggregated and not allow_risky_workers:
            safe_num_workers = 0
            if num_workers != 0:
                print(
                    f"Overriding num_workers={num_workers} to 0 for mixed mode "
                    "because aggregated TCGA mode can duplicate large in-memory caches "
                    "across workers. Set [training] allow_mixed_multiworker=true to override."
                )
        else:
            safe_num_workers = int(num_workers)

        # Worker initialization function for reproducible randomness in multi-worker DataLoader
        def seed_worker(worker_id):
            worker_seed = torch.initial_seed() % 2**32
            np.random.seed(worker_seed)
            random.seed(worker_seed)

        # Generator for reproducible shuffling
        g = torch.Generator()
        g.manual_seed(seed)

        return DataLoader(
            mixed_train_set,
            batch_sampler=sampler,
            num_workers=safe_num_workers,
            worker_init_fn=seed_worker if safe_num_workers > 0 else None,
            generator=g
        )

    def _write_run_metadata(self, log_object):
        output_path = os.path.join(self.savedir, 'run_metadata.json')
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(log_object, f, indent=2)
        print(f"Saved run metadata to {output_path}")

    def _predict_without_training(self, model, eval_set, training_params):
        batch_size = training_params.get('batch_size', 16)
        num_workers = training_params.get('num_workers', 0)
        loader = DataLoader(
            eval_set,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers)
        preds, labels = predict(model, loader)
        return preds, labels



    def _build_dataset(self):

        assert 'data' in self.config.sections(), \
            "'data' not found in config file"
        dic = self.config['data']
        transcriptome_path = dic['path_to_transcriptome'] if 'path_to_transcriptome' in dic.keys() else None

        if 'genes' in dic.keys():
            genes = dic['genes']
            if os.path.exists(genes):
                if genes.endswith('.pkl'):
                    genes = pkl.load(open(genes, 'rb'))
                elif genes.endswith('.csv'):
                    genes = pd.read_csv(genes)
                    genes = genes['gene'].tolist() if 'gene' in genes.columns else genes.iloc[:,0].tolist()
                else:
                    raise ValueError(f"Unknown gene file format: {genes}\n Only .pkl and .csv files are supported")
            else:
                genes = genes.split(',')
            for gene in genes:
                assert gene.startswith('ENSG'), "Unknown gene format"
        else:
            genes = None

        # CSCC mode: load full dataset using CSCCDataset class
        if self.data_mode == 'cscc':
            if transcriptome_path is None and not self.true_holdout_no_transcriptome:
                raise ValueError(
                    "CSCC mode requires [data] path_to_transcriptome unless running "
                    "in inference mode with no splits (true-holdout inference)."
                )
            cscc_path_to_data = dic['path_to_data']
            project_column = dic['project_column']

            # Support both per-slide H5 directories and aggregated H5 files
            if os.path.isdir(cscc_path_to_data):
                dataset = CSCCDataset(
                    features_dir=cscc_path_to_data,
                    targets_csv=transcriptome_path,
                    keyfile_paths=dic['keyfile_path'],
                    genes=genes,
                    project_column=project_column,
                    allow_missing_targets=self.true_holdout_no_transcriptome,
                )
            elif cscc_path_to_data.endswith('.h5'):
                # Aggregated CSCC H5: use same 100-tile convention as aggregated TCGA
                dataset = CSCCDataset(
                    features_aggregated=cscc_path_to_data,
                    targets_csv=transcriptome_path,
                    keyfile_paths=dic['keyfile_path'],
                    genes=genes,
                    max_tiles=100,
                    project_column=project_column,
                    allow_missing_targets=self.true_holdout_no_transcriptome,
                )
            else:
                raise ValueError(f"Unsupported path_to_data format for CSCC: {cscc_path_to_data}")

            print(
                "Data mode: CSCC. Loading from "
                f"{os.path.basename(dic['path_to_data'])}, "
                f"{os.path.basename(transcriptome_path) if transcriptome_path else 'NO_TRANSCRIPTOME'}, "
                f"{os.path.basename(dic['keyfile_path'])}"
            )
            summarize_class(dataset)
            return dataset 

        elif self.data_mode == 'tcga':
            if transcriptome_path is None and not self.true_holdout_no_transcriptome:
                raise ValueError(
                    "TCGA mode requires [data] path_to_transcriptome unless running "
                    "in inference mode with no splits (true-holdout inference)."
                )
            path_to_data = dic['path_to_data']

            # Convert project_filter from comma-separated string to list, if present
            project_filter = dic['project_filter'] if 'project_filter' in dic.keys() else None
            if project_filter is not None:
                project_filter = [p.strip() for p in project_filter.split(",") if p.strip()]
            
            if os.path.isdir(path_to_data):
                print(f"Loading TCGA dataset from features directory: {path_to_data}\nMax tiles = 8000")
                features_dir = path_to_data
                dataset = TCGADataset(
                    targets_csv=transcriptome_path,
                    keyfile_path=dic['keyfile_path'],
                    features_dir=features_dir,
                    genes=genes,
                    project_filter=project_filter,
                    max_tiles=8000,
                    allow_missing_targets=self.true_holdout_no_transcriptome,
                )
            elif path_to_data.endswith('.h5'):
                print(f"Loading TCGA dataset from aggregated H5 file: {path_to_data}\nMax tiles = 100")
                features_aggregated = path_to_data
                dataset = TCGADataset(
                    targets_csv=transcriptome_path,
                    keyfile_path=dic['keyfile_path'],
                    features_aggregated=features_aggregated,
                    genes=genes,
                    project_filter=project_filter,
                    max_tiles = 100,
                    allow_missing_targets=self.true_holdout_no_transcriptome,
                )
            else:
                raise ValueError(f"Unsupported path_to_data format for TCGA: {path_to_data}")
            print(
                f"Data mode: TCGA. Loaded from {os.path.basename(dic['path_to_data'])}, "
                f"{os.path.basename(transcriptome_path) if transcriptome_path else 'NO_TRANSCRIPTOME'}, "
                f"{os.path.basename(dic['keyfile_path'])}"
            )
            
            return dataset

        elif self.data_mode == 'mixed_cscc_tcga_anchor':
            # CSCC dataset (target domain)
            project_column = dic['project_column'] if 'project_column' in dic.keys() else 'metastasis'
            cscc_path_to_data = dic['path_to_data']

            if os.path.isdir(cscc_path_to_data):
                cscc_dataset = CSCCDataset(
                    features_dir=cscc_path_to_data,
                    targets_csv=dic['path_to_transcriptome'],
                    keyfile_paths=dic['keyfile_path'],
                    genes=genes,
                    project_column=project_column,
                )
            elif cscc_path_to_data.endswith('.h5'):
                cscc_dataset = CSCCDataset(
                    features_aggregated=cscc_path_to_data,
                    targets_csv=dic['path_to_transcriptome'],
                    keyfile_paths=dic['keyfile_path'],
                    genes=genes,
                    max_tiles=100,
                    project_column=project_column,
                )
            else:
                raise ValueError(f"Unsupported path_to_data format for CSCC in mixed mode: {cscc_path_to_data}")

            # TCGA anchor dataset (replay buffer domain)
            if 'tcga_path_to_data' not in dic.keys() or 'tcga_path_to_transcriptome' not in dic.keys() or 'tcga_keyfile_path' not in dic.keys():
                raise ValueError(
                    "Mixed mode requires tcga_path_to_data, tcga_path_to_transcriptome and tcga_keyfile_path in [data]."
                )
            tcga_projects = self._get_anchor_projects()
            tcga_path_to_data = dic['tcga_path_to_data']
            if os.path.isdir(tcga_path_to_data):
                tcga_dataset = TCGADataset(
                    targets_csv=dic['tcga_path_to_transcriptome'],
                    keyfile_path=dic['tcga_keyfile_path'],
                    features_dir=tcga_path_to_data,
                    genes=genes,
                    project_filter=tcga_projects,
                    max_tiles=8000
                )
            elif tcga_path_to_data.endswith('.h5'):
                tcga_dataset = TCGADataset(
                    targets_csv=dic['tcga_path_to_transcriptome'],
                    keyfile_path=dic['tcga_keyfile_path'],
                    features_aggregated=tcga_path_to_data,
                    genes=genes,
                    project_filter=tcga_projects,
                    max_tiles=100
                )
            else:
                raise ValueError(f"Unsupported tcga_path_to_data format: {tcga_path_to_data}")

            mixed_dataset = MixedDataset(cscc_dataset, tcga_dataset)
            print(
                "Data mode: mixed_cscc_tcga_anchor\n"
                f"CSCC samples: {len(cscc_dataset)} | "
                f"TCGA anchor samples: {len(tcga_dataset)} | "
                f"Anchor projects: {tcga_projects}"
            )
            return {
                'cscc_dataset': cscc_dataset,
                'tcga_dataset': tcga_dataset,
                'mixed_dataset': mixed_dataset,
                'anchor_projects': tcga_projects,
            }

    def _cross_validation_mixed_outer(self, n_folds=5, random_state=0, logdir='exp'):
        model_params = self._read_architecture()
        training_params = self._read_training_params()
        data_bundle = self._build_dataset()
        cscc_dataset = data_bundle['cscc_dataset']
        tcga_dataset = data_bundle['tcga_dataset']
        anchor_projects = data_bundle['anchor_projects']
        model_params['input_dim'] = data_bundle['mixed_dataset'].dim
        model_params['output_dim'] = len(data_bundle['mixed_dataset'].genes)

        valid_size = 0.1 if 'patience' in training_params.keys() else 0
        train_idx, valid_idx, test_idx = cscc_dataset.stratified_kfold(
            n_splits=n_folds, valid_size=valid_size, random_state=random_state
        )

        report = {'gene': list(cscc_dataset.genes)}
        full_preds = np.zeros((len(cscc_dataset), model_params['output_dim']))
        full_labels = np.zeros((len(cscc_dataset), model_params['output_dim']))
        visited_mask = np.zeros(len(cscc_dataset), dtype=bool)

        train_patients_list = []
        valid_patients_list = []
        test_patients_list = []
        metadata = {
            'mode': 'mixed_cscc_tcga_anchor',
            'phase': 'outer_cv',
            'date_time': datetime.now().isoformat(),
            'anchor_projects': anchor_projects,
            'folds': []
        }

        for k in range(n_folds):
            print(f"Running mixed outer fold {k}...")
            fold_logdir = os.path.join(logdir, f'fold_{k}')
            os.makedirs(fold_logdir, exist_ok=True)
            self._set_global_seeds(random_state + k)

            train_set_cscc = Subset(cscc_dataset, train_idx[k])
            valid_set = Subset(cscc_dataset, valid_idx[k]) if len(valid_idx[k]) > 0 else None
            test_set = Subset(cscc_dataset, test_idx[k])

            tr_set = set(train_idx[k])
            va_set = set(valid_idx[k])
            te_set = set(test_idx[k])
            assert len(tr_set & va_set) == 0, f"Fold {k}: train/valid overlap"
            assert len(tr_set & te_set) == 0, f"Fold {k}: train/test overlap"
            assert len(va_set & te_set) == 0, f"Fold {k}: valid/test overlap"

            train_loader = self._build_mixed_train_loader(
                cscc_dataset=cscc_dataset,
                tcga_dataset=tcga_dataset,
                cscc_indices=train_idx[k],
                batch_size=training_params.get('batch_size', 16),
                num_workers=training_params.get('num_workers', 0),
                seed=random_state + k,
            )

            if valid_set is not None:
                valid_projects = cscc_dataset.projects.iloc[valid_idx[k]]
                valid_projects = valid_projects.astype('category').cat.codes.values.astype('int64')
            else:
                valid_projects = None

            test_projects = np.array([str(x).replace('_', '-') for x in cscc_dataset.projects.iloc[test_idx[k]].values])

            if self.inference:
                model = self._load_saved_model_for_fold(k, cp.deepcopy(model_params))
                if model is None:
                    raise ValueError(
                        "Inference mode for mixed_cscc_tcga_anchor requires use_saved_model "
                        "to point to a directory with per-fold models."
                    )
                preds, labels = self._predict_without_training(model, test_set, training_params)
            else:
                model = self._initialize_model(cp.deepcopy(model_params), train_set_cscc, k)
                if self.null_run:
                    preds, labels = self._predict_without_training(model, test_set, training_params)
                    fold_path = os.path.join(self.savedir, 'model_' + str(k))
                    os.makedirs(fold_path, exist_ok=True)
                    torch.save(model, os.path.join(fold_path, 'model.pt'))
                else:
                    optimizer, scheduler = self._setup_optimization(model)
                    loss_params = self._read_loss_params()
                    preds, labels = fit(
                        model,
                        train_set_cscc,
                        valid_set,
                        valid_projects,
                        test_set=test_set,
                        params=training_params,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        logdir=fold_logdir,
                        path=os.path.join(self.savedir, 'model_' + str(k)),
                        train_loader=train_loader,
                        **loss_params
                    )

            full_preds[test_idx[k]] = preds
            full_labels[test_idx[k]] = labels
            visited_mask[test_idx[k]] = True

            for project in np.unique(test_projects):
                pred = preds[test_projects == project]
                label = labels[test_projects == project]
                report['correlation_' + project + '_fold_' + str(k)] = compute_metrics(label, pred)

            train_patients_fold = cscc_dataset.patients[train_idx[k]]
            valid_patients_fold = cscc_dataset.patients[valid_idx[k]] if valid_set is not None else np.array([], dtype=object)
            test_patients_fold = cscc_dataset.patients[test_idx[k]]
            train_patients_list.append(np.array(train_patients_fold))
            valid_patients_list.append(np.array(valid_patients_fold))
            test_patients_list.append(np.array(test_patients_fold))

            metadata['folds'].append({
                'fold': k,
                'n_train_cscc': int(len(train_idx[k])),
                'n_valid_cscc': int(len(valid_idx[k])),
                'n_test_cscc': int(len(test_idx[k])),
                'n_anchor_tcga': int(len(tcga_dataset)),
            })

        splits_output_path = os.path.join(self.savedir, 'uni_patient_splits_rs0.pkl')
        save_patient_splits(train_patients_list, valid_patients_list, test_patients_list, splits_output_path)

        pct_visited = visited_mask.mean() * 100
        print(f"Visited {pct_visited:.2f}% of CSCC patients during cross-validation.")
        all_projects = np.array([str(x).replace('_', '-') for x in cscc_dataset.projects.values])
        unique_projects = np.unique(all_projects)

        # Compute correlations for the whole dataset as a *concatenation* of cross-validation folds
        report_whole_dataset = {'gene': list(cscc_dataset.genes)}
        for project in unique_projects:
            pred = full_preds[all_projects == project]
            label = full_labels[all_projects == project]
            report_whole_dataset['correlation_' + project] = compute_metrics(label, pred)

        report = pd.DataFrame(report)
        report_whole_dataset = pd.DataFrame(report_whole_dataset)
        if self.inference:
            filename = 'results_per_fold_inference.csv'
            filename_whole_dataset = 'results_whole_dataset_inference.csv'
        else:
            filename = 'results_per_fold_null.csv' if self.null_run else 'results_per_fold.csv'
            filename_whole_dataset = 'results_whole_dataset_null.csv' if self.null_run else 'results_whole_dataset.csv'
        report.to_csv(os.path.join(self.savedir, filename), index=False)
        report_whole_dataset.to_csv(os.path.join(self.savedir, filename_whole_dataset), index=False)
        self._write_run_metadata(metadata)

        if self.inference:
            patients_array = cscc_dataset.patients
            test_mask = visited_mask
            raw_pred_df = pd.DataFrame(full_preds[test_mask].T, index=list(cscc_dataset.genes), columns=patients_array[test_mask])
            raw_gt_df = pd.DataFrame(full_labels[test_mask].T, index=list(cscc_dataset.genes), columns=patients_array[test_mask])
            raw_pred_df.to_csv(os.path.join(self.savedir, 'raw_predictions.csv'))
            raw_gt_df.to_csv(os.path.join(self.savedir, 'raw_ground_truth.csv'))

            scatter_dir = os.path.join(self.savedir, 'scatter_plots')
            all_projects = np.array([str(x).replace('_', '-') for x in cscc_dataset.projects.values])
            unique_projects = np.unique(all_projects)
            for project in unique_projects:
                proj_mask = (all_projects == project) & test_mask
                if not np.any(proj_mask):
                    continue
                proj_title = f'{project}: Predicted vs Ground Truth Gene Expression'
                out_path = os.path.join(scatter_dir, f'{project}_pred_vs_gt.png')
                _plot_pred_vs_gt_scatter(
                    pred_matrix=full_preds[proj_mask],
                    gt_matrix=full_labels[proj_mask],
                    gene_names=list(cscc_dataset.genes),
                    title=proj_title,
                    output_path=out_path,
                    average_per_gene=False,
                )
            overall_title = 'OVERALL: Predicted vs Ground Truth Gene Expression'
            overall_path = os.path.join(scatter_dir, 'OVERALL_pred_vs_gt.png')
            _plot_pred_vs_gt_scatter(
                pred_matrix=full_preds[test_mask],
                gt_matrix=full_labels[test_mask],
                gene_names=list(cscc_dataset.genes),
                title=overall_title,
                output_path=overall_path,
                average_per_gene=False,
            )

        return report, report_whole_dataset

    def _cross_validation_mixed_nested(self, n_folds=5, random_state=0, logdir='exp'):
        """
        Nested cross-validation for mixed mode: 
        Outer loop: split CSCC samples into n_folds.
        Inner loop: split CSCC samples into inner_folds.
        For each inner fold, train a model and evaluate on the inner valid set.
        Select the best learning rate from the inner loop and retrain on the entire inner train set.
        Evaluate on the outer test set.
        Repeat for all outer folds.
        Return the best model and the metrics.
        Not tested in its entirety.
        """
        model_params = self._read_architecture()
        training_params = self._read_training_params()
        data_bundle = self._build_dataset()
        cscc_dataset = data_bundle['cscc_dataset']
        tcga_dataset = data_bundle['tcga_dataset']
        anchor_projects = data_bundle['anchor_projects']
        model_params['input_dim'] = data_bundle['mixed_dataset'].dim
        model_params['output_dim'] = len(data_bundle['mixed_dataset'].genes)

        inner_folds = int(self.config['training']['inner_folds']) if 'training' in self.config.sections() and 'inner_folds' in self.config['training'].keys() else 3
        lr_candidates = self._read_float_list(
            section='optimization',
            key='nested_lr_candidates',
            default=[float(self.config['optimization']['lr'])] if 'optimization' in self.config.sections() and 'lr' in self.config['optimization'].keys() else [1e-3]
        )

        outer_train_idx, _, outer_test_idx = cscc_dataset.stratified_kfold(
            n_splits=n_folds, valid_size=0.0, random_state=random_state
        )

        report = {'gene': list(cscc_dataset.genes)}
        full_preds = np.zeros((len(cscc_dataset), model_params['output_dim']))
        full_labels = np.zeros((len(cscc_dataset), model_params['output_dim']))
        visited_mask = np.zeros(len(cscc_dataset), dtype=bool)
        metadata = {
            'mode': 'mixed_cscc_tcga_anchor',
            'phase': 'nested_cv',
            'date_time': datetime.now().isoformat(),
            'anchor_projects': anchor_projects,
            'inner_folds': inner_folds,
            'lr_candidates': lr_candidates,
            'folds': []
        }

        for k in range(n_folds):
            print(f"Running nested outer fold {k}...")
            self._set_global_seeds(random_state + k)
            pool_idx = outer_train_idx[k]
            test_idx = outer_test_idx[k]
            test_set = Subset(cscc_dataset, test_idx)

            inner_train_idx, inner_valid_idx = self._build_cscc_inner_folds(
                cscc_dataset=cscc_dataset,
                eligible_indices=pool_idx,
                n_splits=inner_folds,
                random_state=random_state + k
            )

            candidate_scores = []
            for cand_lr in lr_candidates:
                fold_scores = []
                for j in range(inner_folds):
                    fold_logdir = os.path.join(logdir, f'outer_{k}_inner_{j}_lr_{cand_lr}')
                    os.makedirs(fold_logdir, exist_ok=True)
                    train_loader = self._build_mixed_train_loader(
                        cscc_dataset=cscc_dataset,
                        tcga_dataset=tcga_dataset,
                        cscc_indices=inner_train_idx[j],
                        batch_size=training_params.get('batch_size', 16),
                        num_workers=training_params.get('num_workers', 0),
                        seed=random_state + k * 100 + j,
                    )
                    train_set_cscc = Subset(cscc_dataset, inner_train_idx[j])
                    valid_set_inner = Subset(cscc_dataset, inner_valid_idx[j])
                    valid_projects_inner = cscc_dataset.projects.iloc[inner_valid_idx[j]]
                    valid_projects_inner = valid_projects_inner.astype('category').cat.codes.values.astype('int64')

                    model = self._initialize_model(cp.deepcopy(model_params), train_set_cscc, k)
                    optimizer, scheduler = self._setup_optimization(model, override_lr=cand_lr)
                    loss_params = self._read_loss_params()
                    preds_inner, labels_inner = fit(
                        model,
                        train_set_cscc,
                        valid_set_inner,
                        valid_projects_inner,
                        test_set=valid_set_inner,
                        params=training_params,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        logdir=fold_logdir,
                        path=None,
                        train_loader=train_loader,
                        **loss_params
                    )
                    corr_values = np.asarray(compute_metrics(labels_inner, preds_inner), dtype=float)
                    fold_scores.append(float(np.nanmean(corr_values)))
                candidate_scores.append(float(np.nanmean(fold_scores)))
                print(f"Outer fold {k} candidate lr={cand_lr:.2e} mean inner score={candidate_scores[-1]:.4f}")

            best_idx = int(np.nanargmax(candidate_scores))
            best_lr = float(lr_candidates[best_idx])
            valid_size = 0.1 if 'patience' in training_params.keys() else 0
            if valid_size > 0:
                retrain_train_idx, retrain_valid_idx = self._build_cscc_train_valid_split(
                    cscc_dataset=cscc_dataset,
                    eligible_indices=pool_idx,
                    valid_size=valid_size,
                    random_state=random_state + k
                )
                valid_set = Subset(cscc_dataset, retrain_valid_idx)
                valid_projects = cscc_dataset.projects.iloc[retrain_valid_idx]
                valid_projects = valid_projects.astype('category').cat.codes.values.astype('int64')
            else:
                retrain_train_idx = np.array(pool_idx, dtype=int)
                retrain_valid_idx = np.array([], dtype=int)
                valid_set = None
                valid_projects = None

            train_loader = self._build_mixed_train_loader(
                cscc_dataset=cscc_dataset,
                tcga_dataset=tcga_dataset,
                cscc_indices=retrain_train_idx,
                batch_size=training_params.get('batch_size', 16),
                num_workers=training_params.get('num_workers', 0),
                seed=random_state + k
            )
            train_set_cscc = Subset(cscc_dataset, retrain_train_idx)
            model = self._initialize_model(cp.deepcopy(model_params), train_set_cscc, k)
            optimizer, scheduler = self._setup_optimization(model, override_lr=best_lr)
            fold_logdir = os.path.join(logdir, f'outer_fold_{k}')
            os.makedirs(fold_logdir, exist_ok=True)
            loss_params = self._read_loss_params()
            preds, labels = fit(
                model,
                train_set_cscc,
                valid_set,
                valid_projects,
                test_set=test_set,
                params=training_params,
                optimizer=optimizer,
                scheduler=scheduler,
                logdir=fold_logdir,
                path=os.path.join(self.savedir, 'model_' + str(k)),
                train_loader=train_loader,
                **loss_params
            )

            full_preds[test_idx] = preds
            full_labels[test_idx] = labels
            visited_mask[test_idx] = True
            test_projects = np.array([str(x).replace('_', '-') for x in cscc_dataset.projects.iloc[test_idx].values])
            for project in np.unique(test_projects):
                pred = preds[test_projects == project]
                label = labels[test_projects == project]
                report['correlation_' + project + '_fold_' + str(k)] = compute_metrics(label, pred)

            metadata['folds'].append({
                'fold': k,
                'best_lr': best_lr,
                'candidate_scores': candidate_scores,
                'n_train_cscc': int(len(retrain_train_idx)),
                'n_valid_cscc': int(len(retrain_valid_idx)),
                'n_test_cscc': int(len(test_idx)),
                'n_anchor_tcga': int(len(tcga_dataset)),
            })

        pct_visited = visited_mask.mean() * 100
        print(f"Visited {pct_visited:.2f}% of CSCC patients during nested CV.")
        all_projects = np.array([str(x).replace('_', '-') for x in cscc_dataset.projects.values])
        unique_projects = np.unique(all_projects)
        report_whole_dataset = {'gene': list(cscc_dataset.genes)}
        for project in unique_projects:
            pred = full_preds[all_projects == project]
            label = full_labels[all_projects == project]
            report_whole_dataset['correlation_' + project] = compute_metrics(label, pred)

        report = pd.DataFrame(report)
        report_whole_dataset = pd.DataFrame(report_whole_dataset)
        report.to_csv(os.path.join(self.savedir, 'results_per_fold_nested.csv'), index=False)
        report_whole_dataset.to_csv(os.path.join(self.savedir, 'results_whole_dataset_nested.csv'), index=False)
        self._write_run_metadata(metadata)
        return report, report_whole_dataset

    def single_run(self, random_state=0, logdir='./exp'):
        """Experiment with a single train/test split.

        Args:
            random_state (int): Random seed used for splitting the data.
            logdir (str): Path for TensoboardX.

        Returns:
            pandas DataFrame: The metrics per gene.
        """

        model_params = self._read_architecture()
        training_params = self._read_training_params()
        dataset = self._build_dataset()
        if self.data_mode == 'mixed_cscc_tcga_anchor':
            raise NotImplementedError(
                "single_run is not implemented for mixed mode. Use --run cross_validation."
            )
        evalset = dataset
        print(f"Single run dataset after _build_dataset():")
        summarize_class(dataset)
        model_params['input_dim'] = dataset.dim
        model_params['output_dim'] = len(dataset.genes)

        if self.splits is None:
            # generating splits from scratch
            if self.data_mode == 'cscc':
                # Use stratified splitting with pair protection for CSCC
                train_idx, valid_idx, test_idx = dataset.stratified_split(
                    test_size=0.1, valid_size=0.1, random_state=random_state)
            elif self.data_mode == 'tcga':
                train_idx, valid_idx, test_idx = dataset.stratified_split(
                    test_size=0.1, valid_size=0.1, random_state=random_state)
            else:
                train_idx, valid_idx, test_idx = patient_split(dataset, random_state)
        elif self.splits is not None and self.fold is not None:
            # crossval to single run configuration
            self.split = [self.splits[0][self.fold], self.splits[1][self.fold], self.splits[2][self.fold]]
            train_idx, valid_idx, test_idx = match_patient_single(dataset, self.split)
        elif self.split is not None:
            # single run to single run configuration
            train_idx, valid_idx, test_idx = match_patient_split(dataset, self.split)
        else:
            raise ValueError("Unrecognized split configuration")

        train_set = Subset(dataset, train_idx)
        valid_set = Subset(evalset, valid_idx)
        test_set = Subset(evalset, test_idx)
        
        if self.data_mode == 'tcga':
            dic = {}
            for project in dataset.projects.unique():
                if project in ['TCGA-LUAD', 'TCGA-LUSC', 'TCGA_LUAD', 'TCGA_LUSC']:
                    dic[project] = 'TCGA-LUNG'
                elif project in ['TCGA-KICH', 'TCGA-KIRC', 'TCGA-KIRP', 'TCGA_KICH', 'TCGA_KIRC', 'TCGA_KIRP']:
                    dic[project] = 'TCGA-KIDN'
                elif project in ['TCGA-UCS', 'TCGA-UCEC']:
                    dic[project] = 'TCGA-UTER'
                else:
                    dic[project] = project
            dataset.projects = dataset.projects.map(dic)

        valid_projects = dataset.projects[valid_idx]
        valid_projects = valid_projects.astype(
            'category').cat.codes.values.astype('int64')
        test_projects = dataset.projects[test_idx].apply(
            lambda x: x.replace('_', '-')).values

        if self.inference:
            model_path = os.path.join(self.use_saved_model, 'model.pt')
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Inference model not found at {model_path}")
            model = torch.load(model_path)
            if 'ks' in model_params.keys():
                model.ks = model_params['ks']
            if 'top_k' in model_params.keys():
                model.top_k = model_params['top_k']
            if 'bottom_ks' in model_params.keys():
                model.bottom_ks = model_params['bottom_ks']
            if 'dropout' in model_params.keys():
                model.do.p = model_params['dropout']
            if 'proportional_ks' in model_params.keys():
                model.proportional_ks = model_params['proportional_ks']
            else:
                model.proportional_ks = False
            preds, labels = self._predict_without_training(model, test_set, training_params)
        else:
            if self.use_saved_model:
                
                model = torch.load(os.path.join(
                                   self.use_saved_model,
                                   'model.pt'))
                if 'ks' in model_params.keys():
                    model.ks = model_params['ks']
                if 'top_k' in model_params.keys():
                    model.top_k = model_params['top_k']
                if 'bottom_ks' in model_params.keys():
                    model.bottom_ks = model_params['bottom_ks']
                if 'dropout' in model_params.keys():
                    model.do.p = model_params['dropout']
                if 'proportional_ks' in model_params.keys():
                    model.proportional_ks = model_params['proportional_ks']
                else:
                    model.proportional_ks = False

            else:
                print("Initializing model without saved model")
                # Initialize bias of the last layer with the average target value on the train set
                try:
                    model_params['bias_init'] = torch.nn.Parameter(
                        torch.Tensor(
                            np.mean(
                                [sample[1] for sample in train_set], axis=0)
                            ).cuda())
                except ValueError:
                    model_params['bias_init'] = torch.nn.Parameter(
                        torch.Tensor(
                            np.mean(
                                [sample[1].numpy() for sample in train_set], axis=0)
                            ).cuda())
                model = HE2RNA(**model_params)
            if self.null_run:
                preds, labels = self._predict_without_training(
                    model, test_set, training_params)
                torch.save(model, os.path.join(self.savedir, 'model_null.pt'))
            else:
                optimizer, scheduler = self._setup_optimization(model)
                loss_params = self._read_loss_params()
                preds, labels = fit(model,
                                    train_set,
                                    valid_set,
                                    valid_projects,
                                    test_set=test_set,
                                    params=training_params,
                                    optimizer=optimizer,
                                    scheduler=scheduler,
                                    logdir=logdir,
                                    path=self.savedir,
                                    **loss_params)

        report = {'gene': list(dataset.genes)}

        for project in np.unique(test_projects):
            pred = preds[test_projects == project]
            label = labels[test_projects == project]
            report['correlation_' + project] = compute_metrics(
                label, pred)

        report = pd.DataFrame(report)
        if self.inference:
            filename = 'results_single_split_inference.csv'
        else:
            filename = 'results_single_split_null.csv' if self.null_run else 'results_single_split.csv'
        report.to_csv(os.path.join(self.savedir, filename), index=False)

        # Save raw predictions and ground truth for single-run inference
        if self.inference:
            patients_array = dataset.patients.values if hasattr(dataset.patients, 'values') else np.array(dataset.patients)
            test_patients = patients_array[test_idx]
            raw_pred_df = pd.DataFrame(preds.T, index=list(dataset.genes), columns=test_patients)
            raw_gt_df = pd.DataFrame(labels.T, index=list(dataset.genes), columns=test_patients)
            raw_pred_df.to_csv(os.path.join(self.savedir, 'raw_predictions_single_run.csv'))
            raw_gt_df.to_csv(os.path.join(self.savedir, 'raw_ground_truth_single_run.csv'))

            # Per-project and overall scatter plots
            scatter_dir = os.path.join(self.savedir, 'scatter_plots')
            for project in np.unique(test_projects):
                mask = test_projects == project
                if not np.any(mask):
                    continue
                proj_title = f'{project}: Predicted vs Ground Truth Gene Expression'
                out_path = os.path.join(scatter_dir, f'{project}_pred_vs_gt.png')
                _plot_pred_vs_gt_scatter(
                    pred_matrix=preds[mask],
                    gt_matrix=labels[mask],
                    gene_names=list(dataset.genes),
                    title=proj_title,
                    output_path=out_path,
                    average_per_gene=False,
                )
            overall_title = 'OVERALL: Predicted vs Ground Truth Gene Expression'
            overall_path = os.path.join(scatter_dir, 'OVERALL_pred_vs_gt.png')
            _plot_pred_vs_gt_scatter(
                pred_matrix=preds,
                gt_matrix=labels,
                gene_names=list(dataset.genes),
                title=overall_title,
                output_path=overall_path,
                average_per_gene=False,
            )

        return report

    def cross_validation(self, n_folds=5, random_state=0, logdir='exp'):
        """
        N-fold cross-validation.
        Function is always called when --cross_validation is specified,
        but true holdout inference and mixed mode are immediately handed off to specific functions.
        """
        if self.true_holdout_no_transcriptome:
            return self._cross_validation_true_holdout_no_transcriptome(
                n_folds=n_folds, random_state=random_state, logdir=logdir
            )
        if self.data_mode == 'mixed_cscc_tcga_anchor':
            use_nested = self._read_bool('training', 'nested_cv', default=False)
            if use_nested and not self.inference:
                return self._cross_validation_mixed_nested(n_folds=n_folds, random_state=random_state, logdir=logdir)
            # For inference, always use outer CV with provided saved models
            return self._cross_validation_mixed_outer(n_folds=n_folds, random_state=random_state, logdir=logdir)

        model_params = self._read_architecture()
        training_params = self._read_training_params()
        dataset = self._build_dataset()
        evalset = dataset
        model_params['input_dim'] = dataset.dim
        model_params['output_dim'] = len(dataset.genes)

        if self.subsample is not None:
            np.random.seed(random_state)
            ind = np.random.permutation(len(dataset))[:int(self.subsample * len(dataset))]
            genes = dataset.genes
            patients = dataset.patients[ind]
            projects = dataset.projects[ind].reset_index(drop=True)
            dataset = Subset(dataset, ind)
            dataset.genes = genes
            dataset.patients = patients
            dataset.projects = projects

        if self.splits is None:
            print(f"No splits file provided")
            valid_size = 0.1 if 'patience' in training_params.keys() else 0
            if valid_size == 0:
                print(f"No patience provided, so no early stopping will be performed. Valid_size = {valid_size} (no validation set)")
            
            # stratified_kfold function is different depending on the data mode thus the TCGA/CSCC Dataset class. 
            # this is because the pairing and stratification is vastly different for the two datasets
            if self.data_mode == 'cscc':
                # Use stratified k-fold with pair protection for CSCC
                print(f"CSCC mode: generating splits with stratified_kfold(); n_folds = {n_folds}")
                train_idx, valid_idx, test_idx = dataset.stratified_kfold(
                    n_splits=n_folds, valid_size=valid_size, random_state=random_state)
            elif self.data_mode == 'tcga':
                print(f"TCGA mode: generating splits with stratified_kfold(); n_folds = {n_folds}")
                train_idx, valid_idx, test_idx = dataset.stratified_kfold(
                    n_splits=n_folds, valid_size=valid_size, random_state=random_state)
            else:
                raise ValueError(f"Invalid data mode: {self.data_mode}")
        else:
            print(f"Using provided splits file")
            train_patients, valid_patients, test_patients = self.splits
            splits = (train_patients, valid_patients, test_patients)
            train_idx, valid_idx, test_idx = match_patient_kfold(dataset, splits)

        if self.data_mode == 'tcga':
            dic = {}
            for project in dataset.projects.unique():
                if project in ['TCGA-LUAD', 'TCGA-LUSC', 'TCGA_LUAD', 'TCGA_LUSC']:
                    dic[project] = 'TCGA-LUNG'
                elif project in ['TCGA-KICH', 'TCGA-KIRC', 'TCGA-KIRP', 'TCGA_KICH', 'TCGA_KIRC', 'TCGA_KIRP']:
                    dic[project] = 'TCGA-KIDN'
                elif project in ['TCGA-UCS', 'TCGA-UCEC']:
                    dic[project] = 'TCGA-UTER'
                else:
                    dic[project] = project
            dataset.projects = dataset.projects.map(dic)


        report = {'gene': list(dataset.genes)}
        if self.p_value == 'empirical':
            random = {'gene': list(dataset.genes)}
        else:
            n_samples = {project: [] for project in dataset.projects}

        # We create empty arrays for the whole dataset
        # Shape: (Total_Patients, Total_Genes)
        full_preds = np.zeros((len(dataset), model_params['output_dim']))
        full_labels = np.zeros((len(dataset), model_params['output_dim']))
        
        # We also track which indices we have visited to be 100% sure
        visited_mask = np.zeros(len(dataset), dtype=bool)

        # Collect patient IDs for each fold to save splits
        train_patients_list = []
        valid_patients_list = []
        test_patients_list = []
        
        summarize_class(dataset)

        for k in range(n_folds):
            print(f"Running Fold {k}...")
            if self.data_mode == 'cscc':
                dataset._print_split_stats(train_idx[k], valid_idx[k], test_idx[k])
            fold_logdir = os.path.join(logdir, f'fold_{k}')
            os.makedirs(fold_logdir, exist_ok=True)

            train_set = Subset(dataset, train_idx[k])
            print("Shape of train_idx:", train_idx[k].shape, "first entries:", train_idx[k][:16])
            summarize_class(train_set)
            
            test_set = Subset(evalset, test_idx[k])


            if len(valid_idx) > 0:
                valid_set = Subset(evalset, valid_idx[k])
                valid_projects = dataset.projects[valid_idx[k]]
                valid_projects = valid_projects.astype(
                    'category').cat.codes.values.astype('int64')
            else:
                valid_set = None
                valid_projects = None

            test_projects = dataset.projects[test_idx[k]].apply(
                lambda x: x.replace('_', '-')).values

            # Collect patient IDs for this fold
            train_patients_fold = dataset.patients[train_idx[k]].values if hasattr(dataset.patients, 'values') else np.array(dataset.patients[train_idx[k]])
            test_patients_fold = dataset.patients[test_idx[k]].values if hasattr(dataset.patients, 'values') else np.array(dataset.patients[test_idx[k]])
            
            train_patients_list.append(train_patients_fold)
            test_patients_list.append(test_patients_fold)
            
            valid_patients_fold = np.array([], dtype=object)
            if valid_set is not None:
                valid_patients_fold = dataset.patients[valid_idx[k]].values if hasattr(dataset.patients, 'values') else np.array(dataset.patients[valid_idx[k]])
                valid_patients_list.append(valid_patients_fold)
            else:
                valid_patients_list.append(np.array([], dtype=object))
            print(f"Fold {k} patient lengths: {len(train_patients_fold)}, {len(test_patients_fold)}, {len(valid_patients_fold)}")


            # Initialize the model and define optimizer / inference behaviour
            if self.inference:
                model_path = os.path.join(self.use_saved_model, f'model_{k}', 'model.pt')
                if not os.path.exists(model_path):
                    raise FileNotFoundError(f"Inference model for fold {k} not found at {model_path}")
                model = torch.load(model_path)
                if 'ks' in model_params.keys():
                    model.ks = model_params['ks']
                if 'top_k' in model_params.keys():
                    model.top_k = model_params['top_k']
                if 'bottom_ks' in model_params.keys():
                    model.bottom_ks = model_params['bottom_ks']
                if 'dropout' in model_params.keys():
                    model.do.p = model_params['dropout']
                if 'proportional_ks' in model_params.keys():
                    model.proportional_ks = model_params['proportional_ks']
                else:
                    model.proportional_ks = False
                preds, labels = self._predict_without_training(model, test_set, training_params)
            else:
                if self.use_saved_model:
                    model = torch.load(os.path.join(
                                       self.use_saved_model,
                                       'model_' + str(k),
                                       'model.pt'))
                    if 'ks' in model_params.keys():
                        model.ks = model_params['ks']
                    if 'top_k' in model_params.keys():
                        model.top_k = model_params['top_k']
                    if 'bottom_ks' in model_params.keys():
                        model.bottom_ks = model_params['bottom_ks']
                    if 'dropout' in model_params.keys():
                        model.do.p = model_params['dropout']
                    if 'proportional_ks' in model_params.keys():
                        model.proportional_ks = model_params['proportional_ks']
                    else:
                        model.proportional_ks = False

                else:
                    # Initialize bias of the last layer with the average target value on the train set
                    print("Initializing model without saved model")
                    try:
                        model_params['bias_init'] = torch.nn.Parameter(
                            torch.Tensor(
                                np.mean(
                                    [sample[1] for sample in train_set], axis=0)
                            ).cuda())
                    except ValueError:
                        model_params['bias_init'] = torch.nn.Parameter(
                            torch.Tensor(
                                np.mean(
                                    [sample[1].numpy() for sample in train_set], axis=0)
                            ).cuda())
                    model = HE2RNA(**model_params)
                if self.null_run:
                    preds, labels = self._predict_without_training(
                        model, test_set, training_params)
                    fold_path = os.path.join(self.savedir, 'model_' + str(k))
                    os.makedirs(fold_path, exist_ok=True)
                    torch.save(model, os.path.join(fold_path, 'model.pt'))
                else:
                    optimizer, scheduler = self._setup_optimization(model)
                    loss_params = self._read_loss_params()

                    # Train model
                    preds, labels = fit(model,
                                        train_set,
                                        valid_set,
                                        valid_projects,
                                        test_set=test_set,
                                        params=training_params,
                                        optimizer=optimizer,
                                        scheduler=scheduler,
                                        logdir=fold_logdir,
                                        path=os.path.join(
                                            self.savedir,
                                            'model_' + str(k)),
                                        **loss_params)

            current_indices = test_idx[k]
            full_preds[current_indices] = preds
            full_labels[current_indices] = labels
            visited_mask[current_indices] = True

            # Compute metrics for each fold
            for project in np.unique(test_projects):
                pred = preds[test_projects == project]
                label = labels[test_projects == project]
                report['correlation_' + project + '_fold_' + str(k)] = compute_metrics(
                    label, pred)
        
        # Save patient splits to pickle file
        splits_output_path = os.path.join(self.savedir, 'uni_patient_splits_rs0.pkl')
        save_patient_splits(train_patients_list, valid_patients_list, test_patients_list, splits_output_path)

        pct_visited = visited_mask.mean() * 100
        print(f"Visited {pct_visited:.2f}% of patients during cross-validation.")
        
        # Compute metrics for the whole dataset
        all_projects = dataset.projects.apply(lambda x: x.replace('_', '-')).values
        unique_projects = np.unique(all_projects)
        report_whole_dataset = {'gene': list(dataset.genes)}
        for project in unique_projects:
            pred = full_preds[all_projects == project]
            label = full_labels[all_projects == project]

            # Calculate Standard Deviation of the True Labels
            label_std = np.std(label, axis=0)
            
            # Identify "Degenerate" Genes (Low variance / sparse)
            # Threshold: 0.01 is usually sufficient to kill the 'binary' artifacts
            bad_gene_mask = label_std < 0.01
            
            # Compute Metric (Vectorized Pearson)
            # This returns an array of shape (N_genes,)
            corr = compute_metrics(label, pred)
            
            # Apply Filter
            # Instead of deleting the rows (which breaks the DataFrame structure),
            # we set them to NaN. The plotting script will ignore NaNs.
            # corr[bad_gene_mask] = np.nan
            
            # Print how many genes with low variance were found
            n_dropped = np.sum(bad_gene_mask)
            if n_dropped > 0:
                print(f"Project {project}: Found {n_dropped} genes with low variance.")

            report_whole_dataset['correlation_' + project] = corr

        
        report = pd.DataFrame(report)
        report_whole_dataset = pd.DataFrame(report_whole_dataset)
        if self.inference:
            filename = 'results_per_fold_inference.csv'
            filename_whole_dataset = 'results_whole_dataset_inference.csv'
        else:
            filename = 'results_per_fold_null.csv' if self.null_run else 'results_per_fold.csv'
            filename_whole_dataset = 'results_whole_dataset_null.csv' if self.null_run else 'results_whole_dataset.csv'
        report.to_csv(os.path.join(self.savedir, filename), index=False)
        report_whole_dataset.to_csv(os.path.join(self.savedir, filename_whole_dataset), index=False)

        # Save raw predictions and ground truth concatenated over all test folds
        if self.inference:
            patients_array = dataset.patients.values if hasattr(dataset.patients, 'values') else np.array(dataset.patients)
            test_mask = visited_mask
            test_patients = patients_array[test_mask]
            raw_pred_df = pd.DataFrame(full_preds[test_mask].T, index=list(dataset.genes), columns=test_patients)
            raw_gt_df = pd.DataFrame(full_labels[test_mask].T, index=list(dataset.genes), columns=test_patients)
            raw_pred_df.to_csv(os.path.join(self.savedir, 'raw_predictions.csv'))
            raw_gt_df.to_csv(os.path.join(self.savedir, 'raw_ground_truth.csv'))

            # Per-project and overall scatter plots
            scatter_dir = os.path.join(self.savedir, 'scatter_plots')
            for project in unique_projects:
                proj_mask = (all_projects == project) & test_mask
                if not np.any(proj_mask):
                    continue
                proj_title = f'{project}: Predicted vs Ground Truth Gene Expression'
                out_path = os.path.join(scatter_dir, f'{project}_pred_vs_gt.png')
                _plot_pred_vs_gt_scatter(
                    pred_matrix=full_preds[proj_mask],
                    gt_matrix=full_labels[proj_mask],
                    gene_names=list(dataset.genes),
                    title=proj_title,
                    output_path=out_path,
                    average_per_gene=False,
                )
            overall_title = 'OVERALL: Predicted vs Ground Truth Gene Expression'
            overall_path = os.path.join(scatter_dir, 'OVERALL_pred_vs_gt.png')
            _plot_pred_vs_gt_scatter(
                pred_matrix=full_preds[test_mask],
                gt_matrix=full_labels[test_mask],
                gene_names=list(dataset.genes),
                title=overall_title,
                output_path=overall_path,
                average_per_gene=False,
            )

        return report, report_whole_dataset

    def _print_no_transcriptome_inference_warning(self):
        print("\n" + "=" * 96)
        print("WARNING: INFERENCE IS RUNNING WITHOUT path_to_transcriptome (TRUE HOLDOUT MODE).")
        print("No ground truth is available, so correlation metrics will NOT be computed.")
        print("No results_per_fold.csv/results_whole_dataset.csv/raw_ground_truth.csv will be written.")
        print("The run will predict every sample for every fold model and average raw predictions.")
        print("=" * 96 + "\n")

    def _cross_validation_true_holdout_no_transcriptome(self, n_folds=5, random_state=0, logdir='exp'):
        if self.data_mode == 'mixed_cscc_tcga_anchor':
            raise ValueError(
                "True holdout no-transcriptome inference is not supported for mixed mode."
            )
        model_params = self._read_architecture()
        training_params = self._read_training_params()
        dataset = self._build_dataset()
        model_params['input_dim'] = dataset.dim
        model_params['output_dim'] = len(dataset.genes)
        n_samples = len(dataset)
        n_genes = model_params['output_dim']
        print(
            f"Running true-holdout inference with {n_folds} folds over {n_samples} samples "
            f"and {n_genes} genes."
        )
        fold_pred_list = []
        for k in range(n_folds):
            print(f"Running holdout inference for fold {k} on all samples...")
            model = self._load_saved_model_for_fold(k, cp.deepcopy(model_params))
            if model is None:
                raise ValueError(
                    f"Could not load saved model for fold {k} from {self.use_saved_model}"
                )
            preds, _ = self._predict_without_training(model, dataset, training_params)
            if preds.shape[0] != n_samples:
                raise ValueError(
                    f"Fold {k} prediction sample count mismatch: expected {n_samples}, got {preds.shape[0]}"
                )
            print(f"Fold {k} prediction shape: {preds.shape}")
            fold_pred_list.append(preds.astype(np.float32))
        fold_preds = np.stack(fold_pred_list, axis=0)
        avg_preds = np.mean(fold_preds, axis=0)
        print(f"Fold prediction tensor shape: {fold_preds.shape}")
        print(f"Averaged prediction matrix shape: {avg_preds.shape}")
        patients_array = dataset.patients.values if hasattr(dataset.patients, 'values') else np.array(dataset.patients)
        if len(patients_array) != n_samples:
            raise ValueError(
                f"Patient/sample count mismatch: expected {n_samples}, got {len(patients_array)}"
            )
        raw_pred_df = pd.DataFrame(
            avg_preds.T,
            index=list(dataset.genes),
            columns=patients_array
        )
        output_path = os.path.join(self.savedir, 'raw_predictions.csv')
        raw_pred_df.to_csv(output_path)
        print(f"Saved fold-averaged raw predictions to {output_path}")
        report = pd.DataFrame({
            'mode': ['true_holdout_no_transcriptome_inference'],
            'n_samples': [n_samples],
            'n_genes': [n_genes],
            'n_folds': [n_folds],
        })
        report_whole_dataset = report.copy()
        return report, report_whole_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="path to the configuration file")
    parser.add_argument(
        "--run", help="type of experiment, 'single_run' or 'cross_validation'",
        default='single_run')
    parser.add_argument(
        "--n_folds", help="number of folds for 'cross_validation'",
        default=5)
    parser.add_argument(
        "--logdir", help="path to the directory used by TensorboardX",
        default='./exp')
    parser.add_argument(
        "--rs", help="random state",
        default=0)
    parser.add_argument(
        "--output_dir", help="override the output path in the config file",
        default=None)
    parser.add_argument(
        "--null_run", help="skip training and run an untrained/null model",
        action='store_true')
    parser.add_argument(
        "--inference", help="run inference only using a saved model specified in the config [main] use_saved_model",
        action='store_true')
    args = parser.parse_args()
    if args.null_run and args.inference:
        raise ValueError("Cannot use --null_run and --inference at the same time.")
    print("Using configuration defined in {}".format(args.config))

    # Main loop. Supports multiple configs in a single call.
    for config in args.config.split(','):
        exp = Experiment(config, null_run=args.null_run, inference=args.inference)
        if args.output_dir:
            exp.savedir = args.output_dir
            if not os.path.exists(exp.savedir):
                os.makedirs(exp.savedir, exist_ok=True)
        print(f"Path where models will be saved: \n{exp.savedir}")
        print(f"TensorboardX logdir: \n{args.logdir}")
        assert args.run in ['single_run', 'cross_validation'], \
            "Unrecognized experiment, must be either 'single_run' or 'cross_validation"
        if args.run == 'single_run':
            report = exp.single_run(logdir=args.logdir)
        elif args.run == 'cross_validation':
            report, report_whole_dataset = exp.cross_validation(
                n_folds=int(args.n_folds),
                random_state=int(args.rs), logdir=args.logdir)
        print(report)
        print(report_whole_dataset)


if __name__ == '__main__':

    main()
