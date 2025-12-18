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
import pandas as pd
import numpy as np
import copy as cp
from time import sleep
from tqdm import tqdm
import h5py
import torch
from torch import nn
from torch.utils.data import Subset, DataLoader
from torch import optim
from sklearn.metrics import roc_auc_score
from transcriptome_data import TranscriptomeDataset
from wsi_data import load_labels, AggregatedDataset, TCGAFolder, \
    H5Dataset, patient_split, match_patient_split, \
    patient_kfold, match_patient_kfold
from model import HE2RNA, fit, predict
from utils import compute_metrics, summarize_class, save_patient_splits
from constant import PATH_TO_TILES


class Experiment(object):
    """An class that uses a config file to setup and run a gene expression
    prediction experiment.

    Args:
        configfile (str): Path to the configuration file.
    """

    def __init__(self,
                 configfile='config.ini',
                 null_run=False):

        # Read configuration file
        self.config = configparser.ConfigParser()
        self.config.read(configfile)

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
        else:
            self.use_saved_model = False

        if 'single_split' in self.config['main'].keys():
            self.split = pkl.load(open(self.config['main']['single_split'], 'rb'))
        else:
            self.split = None
        if 'splits' in self.config['main'].keys():
            self.splits = pkl.load(open(self.config['main']['splits'], 'rb'))
        else:
            self.splits = None

        if 'subsample' in self.config['main'].keys():
            self.subsample = float(self.config['main']['subsample'])
        else:
            self.subsample = None

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

        return training_params

    def _setup_optimization(self, model):

        if 'optimization' in self.config.sections():
            dic = self.config['optimization']
            optim_params = {'params': model.parameters(),
                            'lr': float(dic['lr'])}
            if 'momentum' in self.config['optimization'].keys():
                optim_params['momentum'] = float(dic['momentum'])
                optim_params['nesterov'] = True
            if 'weight_decay' in self.config['optimization'].keys():
                optim_params['weight_decay'] = float(dic['weight_decay'])

            if dic['optimizer'] == 'sgd':
                return optim.SGD(**optim_params)
            elif dic['optimizer'] == 'adam':
                return optim.Adam(**optim_params)

        else:
            return optim.Adam(lr=1e-3)

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

        if 'genes' in dic.keys():
            genes = dic['genes']
            if os.path.exists(genes):
                genes = pkl.load(
                    open(genes, 'rb'))
            else:
                genes = genes.split(',')
                for gene in genes:
                    assert gene.startswith('ENSG'), "Unknown gene format"
        else:
            genes = None

        if 'path_to_transcriptome' in dic.keys() and 'projectname' in dic.keys():
            projectname = dic['projectname'].split(',')
            transcriptome_data = TranscriptomeDataset.from_saved_file(
                dic['path_to_transcriptome'], projectname=projectname, genes=genes)
        elif 'path_to_transcriptome' in dic.keys():
            print(f"Loading transcriptome data via TranscriptomeDataset.from_saved_file() from {dic['path_to_transcriptome']} \nNo projects given in config, using all\n Length of gene filter: {None if genes is None else len(genes)}")
            transcriptome_data = TranscriptomeDataset.from_saved_file(
                dic['path_to_transcriptome'], genes=genes)
        elif 'projectname' in dic.keys():
            projectname = dic['projectname'].split(',')
            print(f"Loading transcriptome data via TranscriptomeDataset(projectname, genes) from {dic['path_to_transcriptome']} \nProjects given in config: {projectname}\n Length of gene filter: {None if genes is None else len(genes)}")
            transcriptome_data = TranscriptomeDataset(projectname, genes)
            transcriptome_data.load_transcriptomes()
        else:
            print(f"No projects or path to transcriptome given in config, building from scratch with TranscriptomeDataset(None, genes)\n Length of gene filter: {None if genes is None else len(genes)}")
            transcriptome_data = TranscriptomeDataset(None, genes)
            transcriptome_data.load_transcriptomes()

        if 'path_to_data' in dic.keys():

            if dic['path_to_data'].endswith('.pkl'):
                X = pkl.load(open(dic['path_to_data'], 'rb'))
                y, genes, patients, projects = load_labels(transcriptome_data)
                dataset = AggregatedDataset(
                    genes, patients, projects,
                    torch.Tensor(X), torch.Tensor(y))
            elif dic['path_to_data'].endswith('.h5'):
                print("Loading labels of the following transcriptome_data:")
                summarize_class(transcriptome_data)
                
                y, genes, patients, projects = load_labels(transcriptome_data)
                # for i, patient in tqdm(list(enumerate(patients)), desc='Iterating over patients'):
                #     gene_expr_raw = transcriptome_data.transcriptomes['ENSG00000000003.15'][i]
                #     gene_expr_log = np.log10(1 + gene_expr_raw)
                #     # Only print if log10(1 + gene_expr_raw) from above does not match y[i][0] (rounded to 4 decimals)
                #     # or if the patient names do not match
                #     log_eq = round(gene_expr_log, 4) == round(y[i][0], 4)
                #     patient_eq = str(patient) == str(transcriptome_data.transcriptomes['Case.ID'][i])
                #     mismatch_count = 0
                #     if not log_eq or not patient_eq:
                #         mismatch_count += 1
                #         print("Shape of y: ", y.shape, "Shape of y[i]: ", y[i].shape)
                #         print("From after load_labels / transcriptome_data.dataset:")
                #         print(f"Patient {i}: {patient} / {transcriptome_data.transcriptomes['Case.ID'][i]}")
                #         print(f"Patient {i} ENSG...3.15 expression: {genes[0]} / y (log): {y[i][0]} / transcriptome raw: {gene_expr_raw} / transcriptome log10(1+x): {gene_expr_log}")
                    
                
                # if mismatch_count == 0:
                #     print("No mismatches found, indexes of transcriptome data match")
                # else:
                #     print(f"Total mismatch count: {mismatch_count}")
                #     raise ValueError("Mismatch found in transcriptome data. Please check input files and index matching.")


                dataset = H5Dataset(
                    genes, patients, projects, dic['path_to_data'], y, in_memory=True)
                
                print(f"Shape of dataset.data: {dataset.data.shape}")
                # Print headers of dataset.data: print shape and sample information in a loop (num_samples, tile_dim, num_tiles)
                # for idx in range(25):  # print first 3 samples as a header preview
                #     print(f"Sample {idx}: patient {dataset.patients[idx]}")
                #     print(f"Sample {idx}: project {dataset.projects[idx]}")
                #     print(f"Sample {idx}: gene expression from self.targets: {dataset.targets[idx]} from y: {y[idx]}")
                    # print(idx, dataset.data[idx][:10]) # print first 10 entries of each sample

            # checking loaded transcriptomes and tile features against original file
            df = pd.read_csv('/home/dvanerp/pepsi/data/processed/tcga_rna/tcga_full_transcriptome_uni_filtered_ensembl_subset_log.csv')
            
            with h5py.File(dic['path_to_data'], 'r') as f:
                og_h5 = f['X'][:, 0, 3:]
            print("Shape of og_h5: ", og_h5.shape)
            print(f"Shape of dataset.data: {dataset.data.shape}")
            print(f"Length of idx: {len(dataset.targets)}")
            # Check dataset.targets against the dataframe for first (ENSG00000000003.15) and last (ENSG00000288675.1) gene columns
            for idx in range(len(dataset.targets)):
                target_first = dataset.targets[idx][0]
                target_last = dataset.targets[idx][-1]
                df_first = df.iloc[idx]["ENSG00000000003.15"]
                df_last = df.iloc[idx]["ENSG00000288675.1"]

                #2048 numpy features of dataset.data at idx tile 0
                # Ensure data_features is a 2048 np array (matches og_features: (2048,))
                # dataset.data shape: (num_samples, 2048, 100) -- want all 2048 dims, tile 0
                
                data_features = dataset.data[idx][:, 0].cpu().numpy() if hasattr(dataset.data[idx][:, 0], 'cpu') else dataset.data[idx][:, 0].numpy()
                og_features = og_h5[idx][:]
                # print("shape of data_features: ", data_features.shape, "shape of og_features: ", og_features.shape)
                # Compare data_features and og_features for exact match
                if not np.array_equal(data_features, og_features):
                    print(f"features mismatch at idx {idx}, patient {dataset.patients[idx]}")
                    print("mean of data_features: ", np.mean(data_features), "mean of og_features: ", np.mean(og_features))

                
                # Compare using float tolerance (since these are floats)
                if not (abs(target_first - df_first) < 1e-4 and abs(target_last - df_last) < 1e-4):
                    print(f"transcriptome mismatch at idx {idx}, patient {dataset.patients[idx]}: target_first={target_first}, df_first={df_first}, target_last={target_last}, df_last={df_last}")
            # print("First 5 values of each of the first 16 entries of og_h5:")
            # print(og_h5[:16, :5])
            # print("The first and last values of each of the first 16 entries of dataset.targets:")
            # print(dataset.targets[:16, [0, -1]])
            print("Completed checking for transcriptome and features mismatches")

        else:
            dataset = TCGAFolder.match_transcriptome_data(
                transcriptome_data)

        return dataset

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
        evalset = dataset
        if dataset.dim == 2051:  # Remove tile levels and coordinates
            model_params['input_dim'] = dataset.dim - 3
        else:
            model_params['input_dim'] = dataset.dim
        model_params['output_dim'] = len(dataset.genes)

        if self.split is None:
            train_idx, valid_idx, test_idx = patient_split(dataset, random_state)
        else:
            train_idx, valid_idx, test_idx = match_patient_split(dataset, self.split)

        train_set = Subset(dataset, train_idx)
        valid_set = Subset(evalset, valid_idx)
        test_set = Subset(evalset, test_idx)

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
        else:
            optimizer = self._setup_optimization(model)
            preds, labels = fit(model,
                                train_set,
                                valid_set,
                                valid_projects,
                                test_set=test_set,
                                params=training_params,
                                optimizer=optimizer,
                                logdir=logdir,
                                path=self.savedir)

        report = {'gene': list(dataset.genes)}

        for project in np.unique(test_projects):
            pred = preds[test_projects == project]
            label = labels[test_projects == project]
            report['correlation_' + project] = compute_metrics(
                label, pred)

        report = pd.DataFrame(report)
        filename = 'results_single_split_null.csv' if self.null_run else 'results_single_split.csv'
        report.to_csv(os.path.join(self.savedir, filename), index=False)
        return report

    def cross_validation(self, n_folds=5, random_state=0, logdir='exp'):
        """N-fold cross-validation.

        Args:
            n (int): Number of folds
            random_state (int): Random seed used for splitting the data.
            logdir (str): Path for TensoboardX.

        Returns:
            pandas DataFrame: The metrics per gene and per fold.
        """

        model_params = self._read_architecture()
        training_params = self._read_training_params()
        dataset = self._build_dataset()
        evalset = dataset
        if dataset.dim == 2051:  # Remove tile levels and coordinates
            model_params['input_dim'] = dataset.dim - 3
        else:
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
            if 'patience' in training_params.keys():
                print("No splits file provided, generating splits with patient_kfold(); n_folds = ", n_folds, "; valid_size = 0.1")
                train_idx, valid_idx, test_idx = patient_kfold(
                    dataset, n_splits=n_folds, valid_size=0.1,
                    random_state=random_state)
            else:
                print("No patience provided, generating splits with patient_kfold(); n_folds = ", n_folds, "; valid_size = 0 (no validation set!)")
                train_idx, valid_idx, test_idx = patient_kfold(
                    dataset, n_splits=n_folds, valid_size=0,
                    random_state=random_state)
        else:
            train_patients, valid_patients, test_patients = self.splits
            splits = zip(train_patients, valid_patients, test_patients)
            train_idx, valid_idx, test_idx = match_patient_kfold(dataset, splits)

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

        for k in range(n_folds):
            print(f"Running Fold {k}...")
            fold_logdir = os.path.join(logdir, f'fold_{k}')
            os.makedirs(fold_logdir, exist_ok=True)

            train_set = Subset(dataset, train_idx[k])
            print("Shape of train_idx:", train_idx[k].shape, "first entries:", train_idx[k][:16])
            summarize_class(train_set)
            summarize_class(dataset)
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
            
            if valid_set is not None:
                valid_patients_fold = dataset.patients[valid_idx[k]].values if hasattr(dataset.patients, 'values') else np.array(dataset.patients[valid_idx[k]])
                valid_patients_list.append(valid_patients_fold)
            else:
                valid_patients_list.append(np.array([], dtype=object))
            print(f"Fold {k} patient lengths: {len(train_patients_fold)}, {len(test_patients_fold)}, {len(valid_patients_fold)}")


            # Initialize the model and define optimizer
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
            else:
                optimizer = self._setup_optimization(model)

                # Train model
                preds, labels = fit(model,
                                    train_set,
                                    valid_set,
                                    valid_projects,
                                    test_set=test_set,
                                    params=training_params,
                                    optimizer=optimizer,
                                    logdir=fold_logdir,
                                    path=os.path.join(
                                        self.savedir,
                                        'model_' + str(k)))
            
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
        filename = 'results_per_fold_null.csv' if self.null_run else 'results_per_fold.csv'
        filename_whole_dataset = 'results_whole_dataset_null.csv' if self.null_run else 'results_whole_dataset.csv'
        report.to_csv(os.path.join(self.savedir, filename), index=False)
        report_whole_dataset.to_csv(os.path.join(self.savedir, filename_whole_dataset), index=False)

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
    args = parser.parse_args()
    print("Using configuration defined in {}".format(args.config))
    for config in args.config.split(','):
        exp = Experiment(config, null_run=args.null_run)
        if args.output_dir:
            exp.savedir = args.output_dir
            if not os.path.exists(exp.savedir):
                os.makedirs(exp.savedir, exist_ok=True)

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
