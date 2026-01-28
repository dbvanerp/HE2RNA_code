"""
HE2RNA: Arrange data and labels into pytorch datasets
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
import numpy as np
import pandas as pd
import h5py
import torch
from torch.utils.data import Dataset, TensorDataset, Subset
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import roc_auc_score
from torchvision.transforms import Compose
from tqdm import tqdm
from joblib import Parallel, delayed
from constant import PATH_TO_TILES, PATH_TO_TRANSCRIPTOME
from utils import summarize_class

def load_labels(transcriptome_dataset):
    """Clean up RNAseq data and return labels, genes and patients.
    """
    assert hasattr(transcriptome_dataset, 'transcriptomes'), \
        "Transcriptomes have not been loaded for this dataset"

    to_drop = ['Case.ID', 'Sample.ID', 'File.ID', 'Project.ID']
    df = transcriptome_dataset.transcriptomes.copy()
    patients = df['Case.ID'].values
    projects = df['Project.ID']
    df.drop(to_drop, axis=1, inplace=True)
    genes = df.columns
    df = np.log10(1 + df)
    y = df.values

    return y, genes, patients, projects


def load_and_aggregate_file(file, reduce=True):
    x = np.load(file)
    x = x[:, 3:]
    if reduce:
        x = np.mean(x, axis=0)
    else:
        x = np.concatenate((x, np.zeros((8000 - x.shape[0], x.shape[1])))).transpose(1, 0) # x.shape[1] 1536 <= 1 + 2 + 1536, UNI2 shape
    return x

def load_npy_data(file_list, reduce=True):
    """Load and aggregate data saved as npy files.

    Args
        reduce (bool): If True, perform mean pooling on slide.
            Else, pad every slide with zeros.
    """
    X = np.array(Parallel(n_jobs=32)(delayed(load_and_aggregate_file)(file) for file in tqdm(file_list)))
    return X


def make_dataset(dir, file_list, labels):
    """Associate file names and labels"""
    images = []
    dir = os.path.expanduser(dir)

    for fname, label in zip(file_list, labels):
        path = os.path.join(dir, fname)
        if os.path.exists(path):
            item = (path, label)
            images.append(item)

    return images


class AggregatedDataset(TensorDataset):
    """A subclass of TensorDataset to use for whole-slide analysis
    (with aggregated data).

    Args
        genes (list): List of Ensembl IDs of genes to be used as targets.
        patients (list): list of patient IDs to perform patient split.
    """
    def __init__(self, genes, patients, projects, *tensors):
        super(AggregatedDataset, self).__init__(*tensors)
        self.genes = genes
        self.patients = patients
        self.projects = projects
        self.dim = tensors[0].shape[1]

    @classmethod
    def match_transcriptome_data(cls, transcriptome_dataset):
        """Use a TranscriptomeDataset object to read corresponding .npy files
        and aggregate tiles.

        Args
            transcriptome_dataset (TranscriptomeDataset)
            binarize (bool): If True, target gene expressions are binarized with
                respect to their median value.
        """
        y, cols, patients, projects = load_labels(transcriptome_dataset)

        file_list = []
        for project, filename in transcriptome_dataset.metadata[['Project.ID', 'Slide.ID']].values:
            project_dir = os.path.join(PATH_TO_TILES, project)
            mpp_path = os.path.join(project_dir, '0.50_mpp')
            if os.path.isdir(mpp_path):
                file_path = os.path.join(mpp_path, filename)
            else:
                file_path = os.path.join(project_dir, filename)
            file_list.append(file_path)
        
        X = load_npy_data(file_list)
        return cls(cols, patients, projects, torch.Tensor(X), torch.Tensor(y))


class ToTensor(object):
    """A simple transformation on numpy array to obtain torch-friendly tensors.
    """
    def __init__(self, n_tiles=8000):
        self.n_tiles = n_tiles

    def __call__(self, sample):
        x = torch.from_numpy(sample).float()
        if x.shape[0] > self.n_tiles:
            x = x[:self.n_tiles]
        elif x.shape[0] < self.n_tiles:
            x = torch.cat((x, torch.zeros((self.n_tiles - x.shape[0], x.shape[1]))))
        return x.t()


class RemoveCoordinates(object):
    """Remove tile levels and coordinates."""
    def __call__(self, sample):
        return sample[3:]


class TCGAFolder(Dataset):
    """A class similar to torchvision.FolderDataset for dealing with npy files
    of one or several TCGA project(s).

    Args
        genes (list): List of Ensembl IDs of genes to be used as targets.
        patients (list): list of patient IDs to perform patient split.
        projectname (str or None): Project.ID
        file_list (list): list of paths to .npy files containing tiled slides.
        labels (list or np.array): the associated gene expression values.
        transform (callable): Preprocessing of the data.
        target_transform (callable): Preprocessing of the targets.
        slide_filter (list or None): Optional list of slide names to keep (matching basename without extension).
    """
    def __init__(self, genes, patients, projects, projectname, file_list, labels,
                 transform=Compose([ToTensor(), RemoveCoordinates()]),
                 target_transform=None, masks=None, slide_filter=None):
        root = PATH_TO_TILES
        
        # Apply slide filter if provided
        # if slide_filter is not None:
        if False:
            print(f"Filtering TCGAFolder dataset with {len(slide_filter)} slides...")
            # Extract slide names from file_list (basename without extension)
            file_slide_names = [os.path.splitext(os.path.basename(f))[0] for f in file_list]
            
            # Create mapping from slide name to index
            slide_to_idx = {name: i for i, name in enumerate(file_slide_names)}
            
            # Find indices that match filter, preserving filter order
            indices = []
            keep_mask = []
            for slide in slide_filter:
                if slide in slide_to_idx:
                    indices.append(slide_to_idx[slide])
                    keep_mask.append(True)
                else:
                    print(f"Warning: Slide {slide} not found in file list")
                    keep_mask.append(False)
            
            if not indices:
                raise ValueError(f"None of the {len(slide_filter)} filtered slides were found.")
            
            # Filter file_list and labels
            file_list = [file_list[i] for i in indices]
            if isinstance(labels, np.ndarray):
                labels = labels[indices]
            else:
                labels = [labels[i] for i in indices]
            
            # Filter patients
            if isinstance(patients, (pd.Series, pd.Index)):
                patients = patients.iloc[indices]
            elif isinstance(patients, np.ndarray):
                patients = patients[indices]
            else:
                patients = np.array([patients[i] for i in indices])
            
            # Filter projects
            if isinstance(projects, (pd.Series, pd.Index)):
                projects = projects.iloc[indices]
            elif isinstance(projects, np.ndarray):
                projects = projects[indices]
            else:
                projects = np.array([projects[i] for i in indices])
            
            if not all(keep_mask):
                print(f"Found {len(indices)} out of {len(slide_filter)} requested slides.")
            else:
                print(f"All {len(indices)} requested slides found.")
        
        samples = make_dataset(root, file_list, labels)
        if len(samples) == 0:
            raise(RuntimeError("Found 0 files in subfolders of: " + root + "\n"))

        self.root = root

        self.patients = patients
        self.projects = projects
        self.samples = samples
        print(f"Number of samples: {len(samples)}")
        # print(samples)

        self.transform = transform
        self.target_transform = target_transform

        self.genes = genes
        self.masks = masks
        # Infer dim from the first sample
        sample_0, _ = self[0]
        self.dim = sample_0.shape[0] - 3
        

    @classmethod
    def match_transcriptome_data(cls, transcriptome_dataset, binarize=False, slide_filter=None):
        """Use a TranscriptomeDataset object to read corresponding .npy files.

        Args
            transcriptome_dataset (TranscriptomeDataset)
            binarize (bool): If True, target gene expressions are binarized with
                respect to their median value.
            slide_filter (list or None): Optional list of slide names to keep (matching slide_name without extension).
        """
        projectname = transcriptome_dataset.projectname
        labels, cols, patients, projects = load_labels(transcriptome_dataset)
        file_list = []
        for project, filename in transcriptome_dataset.metadata[['Project.ID', 'Slide.ID']].values:
            project_dir = os.path.join(PATH_TO_TILES, project)
            subdir = os.path.join(project_dir, '0.50_mpp')
            if os.path.isdir(subdir):
                file_path = os.path.join(subdir, filename)
            else:
                file_path = os.path.join(project_dir, filename)
            file_list.append(file_path)
        
        # If no slide_filter provided, use slide names from metadata (similar to H5Dataset)
        if slide_filter is None:
            slide_filter = [os.path.splitext(s)[0] for s in transcriptome_dataset.metadata['Slide.ID'].values]
        
        return cls(cols, patients, projects, projectname, file_list, labels, slide_filter=slide_filter)

    def __getitem__(self, index):
        path, target = self.samples[index]
        if self.masks is not None:
            mask = self.masks[path.split('/')[-1]]
            idx = np.argsort(mask[:, 0])[::-1]
            sample = np.load(path)[idx] * mask[idx]
        else:
            sample = np.load(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target

    def __len__(self):
        return len(self.samples)


class H5Dataset(Dataset):
    """A class for using data saved in an hdf5 file.

    Args
        genes (list): List of Ensembl IDs of genes to be used as targets.
        patients (list): list of patient IDs to perform patient split.
        filename (str): path to the hdf5 file containing the data.
        labels (list or np.array): the associated gene expression values.
        max_items (int): Maximum number of tiles to use for training.
        slide_filter (list or None): Optional list of slide names to keep (matching slide_name in H5).
    """
    def __init__(self, genes, patients, projects, filename, labels, max_items=8000, in_memory=True, slide_filter=None):
        self.in_memory = in_memory
        self.max_items = max_items
        self.targets = labels
        self.genes = genes
        self.patients = patients
        self.projects = projects
        self.filename = filename 
        self.indices = None
        
        # if slide_filter is not None:
        #     print(f"Filtering H5 dataset with {len(slide_filter)} slides...")
        #     with h5py.File(self.filename, 'r') as f:
        #         if 'slide_name' not in f:
        #             raise KeyError(f"Dataset 'slide_name' not found in {self.filename}. Cannot filter.")
                
        #         h5_slides = f['slide_name'][:]
        #         # Decode if they are bytes
        #         if h5_slides.dtype.kind in ['S', 'V']:
        #             h5_slides = np.array([s.decode('utf-8') for s in h5_slides])
                
        #         # Create a mapping from slide name to H5 index
        #         slide_to_idx = {name: i for i, name in enumerate(h5_slides)}
                
        #         # Find indices in H5 that match our filtered slides, in the CORRECT ORDER
        #         indices = []
        #         keep_mask = []
        #         for slide in slide_filter:
        #             if slide in slide_to_idx:
        #                 indices.append(slide_to_idx[slide])
        #                 keep_mask.append(True)
        #             else:
        #                 print(f"Warning: Slide {slide} not found in H5 file")
        #                 keep_mask.append(False)
                
        #         if not indices:
        #             raise ValueError(f"None of the {len(slide_filter)} filtered slides were found in the H5 file.")
                
        #         self.indices = np.array(indices)
                
        #         # We must also filter labels, patients, projects to match the slides actually found in H5
        #         if not all(keep_mask):
        #             print(f"Found {len(indices)} out of {len(slide_filter)} requested slides in H5.")
        #             self.targets = self.targets[keep_mask]
        #             # Handle both numpy arrays and pandas Series/Index
        #             if isinstance(self.patients, (pd.Series, pd.Index)):
        #                 self.patients = self.patients[keep_mask]
        #             else:
        #                 self.patients = self.patients[keep_mask]
                    
        #             if isinstance(self.projects, (pd.Series, pd.Index)):
        #                 self.projects = self.projects[keep_mask]
        #             else:
        #                 self.projects = self.projects[keep_mask]

        if self.in_memory:
            print(f"Loading {filename} into RAM...")
            with h5py.File(self.filename, 'r') as f:
                if self.indices is not None:
                    # h5py requires indices to be in increasing order
                    sort_idx = np.argsort(self.indices)
                    rev_idx = np.argsort(sort_idx)
                    
                    data = f['X'][self.indices[sort_idx], :self.max_items, 3:]
                    # Restore original metadata order
                    data = data[rev_idx]
                else:
                    data = f['X'][:, :self.max_items, 3:]
            print(f"Shape of h5 file data: {data.shape}")


            data = np.asarray(data, dtype=np.float32)
            
            self.data = torch.from_numpy(data).permute(0, 2, 1).contiguous()
            
            self.length = self.data.shape[0]
            self.dim = self.data.shape[1]
            print(f"Loaded successfully. Shape: {self.data.shape}")
            
        else:
            # Just get metadata here, DO NOT open the file yet
            with h5py.File(self.filename, 'r') as f:
                if self.indices is not None:
                    self.length = len(self.indices)
                else:
                    self.length = f['X'].shape[0]
                self.dim = f['X'].shape[2] - 3
            self.file_handle = None # Placeholder

    @classmethod
    def match_transcriptome_data(cls, transcriptome_dataset, filename, max_items=8000, in_memory=True):
        """Use a TranscriptomeDataset object to read corresponding .h5 file.

        Args
            transcriptome_dataset (TranscriptomeDataset)
            filename (str): Path to the H5 file.
        """
        labels, genes, patients, projects = load_labels(transcriptome_dataset)
        # slide_names in metadata usually have .npy, but H5 slide_name doesn't
        slide_names = [os.path.splitext(s)[0] for s in transcriptome_dataset.metadata['Slide.ID'].values]
        return cls(genes, patients, projects, filename, labels, 
                   max_items=max_items, in_memory=in_memory, slide_filter=slide_names)

    def __getitem__(self, index):
        if self.in_memory:
            sample = self.data[index]
        else:
            # LAZY LOADING: Open file only when needed, per worker
            if not hasattr(self, 'file_handle') or self.file_handle is None:
                 self.file_handle = h5py.File(self.filename, 'r')
            
            # Read specific chunk
            real_index = self.indices[index] if self.indices is not None else index
            data_numpy = self.file_handle['X'][real_index, :self.max_items, 3:]
            sample = torch.from_numpy(data_numpy.astype(np.float32)).t()

        target = self.targets[index]
        return sample, target

    def __len__(self):
        return self.length

    def __del__(self):
        if not self.in_memory and hasattr(self, 'data'):
            try:
                self.data.close()
            except Exception:
                pass


def patient_split(dataset, random_state=0):
    """Perform patient split of any of the previously defined datasets.
    """
    patients_unique = np.unique(dataset.patients)
    patients_train, patients_valid = train_test_split(
        patients_unique, test_size=0.2, random_state=random_state)
    patients_valid, patients_test = train_test_split(
        patients_valid, test_size=0.5, random_state=random_state)
    summarize_class(dataset)
    indices = np.arange(len(dataset))
    train_idx = indices[np.any(dataset.patients[:, np.newaxis] ==
                               patients_train[np.newaxis], axis=1)]
    valid_idx = indices[np.any(dataset.patients[:, np.newaxis] ==
                               patients_valid[np.newaxis], axis=1)]
    test_idx = indices[np.any(dataset.patients[:, np.newaxis] ==
                              patients_test[np.newaxis], axis=1)]

    return train_idx, valid_idx, test_idx


def match_patient_split(dataset, split):
    """Recover previously saved patient split
    """
    train_patients, valid_patients, test_patients = split
    indices = np.arange(len(dataset))
    train_idx = indices[np.any(dataset.patients[:, np.newaxis] ==
                               train_patients[np.newaxis], axis=1)]
    valid_idx = indices[np.any(dataset.patients[:, np.newaxis] ==
                               valid_patients[np.newaxis], axis=1)]
    test_idx = indices[np.any(dataset.patients[:, np.newaxis] ==
                              test_patients[np.newaxis], axis=1)]

    return train_idx, valid_idx, test_idx


def patient_kfold(dataset, n_splits=5, random_state=0, valid_size=0.1):
    """Perform cross-validation with patient split.
    """
    print("Starting patient_kfold() function with following dataset...")
    summarize_class(dataset)
    indices = np.arange(len(dataset))
    print(f"Number of indices: {len(indices)}")
    patients_unique = np.unique(dataset.patients)
    print(f"Number of unique patients: {len(patients_unique)}")
    print(f"Number of patients in dataset: {len(dataset.patients)}")
    skf = KFold(n_splits, shuffle=True, random_state=random_state)
    ind = skf.split(patients_unique)

    train_idx = []
    valid_idx = []
    test_idx = []

    for k, (ind_train, ind_test) in enumerate(ind):

        patients_train = patients_unique[ind_train]
        patients_test = patients_unique[ind_test]

        test_idx.append(indices[np.any(dataset.patients[:, np.newaxis] ==
                                       patients_test[np.newaxis], axis=1)])

        if valid_size > 0:
            patients_train, patients_valid = train_test_split(
                patients_train, test_size=valid_size, random_state=0)
            valid_idx.append(indices[np.any(dataset.patients[:, np.newaxis] ==
                                            patients_valid[np.newaxis], axis=1)])

        train_idx.append(indices[np.any(dataset.patients[:, np.newaxis] ==
                                        patients_train[np.newaxis], axis=1)])
        print(f"Fold {k} has {len(train_idx[k])} training samples, {len(valid_idx[k])} validation samples, and {len(test_idx[k])} test samples")
    return train_idx, valid_idx, test_idx


def match_patient_kfold(dataset, splits):
    """Recover previously saved patient splits for cross-validation.
    """

    indices = np.arange(len(dataset))
    train_idx = []
    valid_idx = []
    test_idx = []

    for train_patients, valid_patients, test_patients in splits:

        train_idx.append(indices[np.any(dataset.patients[:, np.newaxis] ==
                                        train_patients[np.newaxis], axis=1)])
        valid_idx.append(indices[np.any(dataset.patients[:, np.newaxis] ==
                                        valid_patients[np.newaxis], axis=1)])
        test_idx.append(indices[np.any(dataset.patients[:, np.newaxis] ==
                                       test_patients[np.newaxis], axis=1)])

    return train_idx, valid_idx, test_idx
