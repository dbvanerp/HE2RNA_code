"""
HE2RNA: Computation of correlations
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
import numpy as np
import pickle as pkl
from joblib import Parallel, delayed

def corr(pred, label, i):
    return np.corrcoef(
        label[:, i],
        pred[:, i])[0, 1]

def compute_metrics(label, pred):
    res = Parallel(n_jobs=16)(
        delayed(corr)(pred, label, i) for i in range(label.shape[1])
    )
    return res

def summarize_class(obj):
    print(f"--- Summary of {type(obj).__name__} ---")
    # Loop through all instance attributes
    for key, value in vars(obj).items():
        # If it has a shape (Tensor or Numpy), print the shape
        if hasattr(value, 'shape'):
            print(f"{key:<20} | Shape: {value.shape}  (Type: {type(value).__name__})")
        # If it is a list/tuple, print the length
        elif hasattr(value, '__len__') and not isinstance(value, str):
            print(f"{key:<20} | Length: {len(value)}  (Type: {type(value).__name__})")
        # Otherwise just print the type/value
        else:
            print(f"{key:<20} | Value: {value}  (Type: {type(value).__name__})")
    print("---------------------------------------")

def save_patient_splits(train_patients_list, valid_patients_list, test_patients_list, output_path):
    """
    Save patient splits to a pickle file.
    
    Args:
        train_patients_list: List of arrays/lists, one per fold containing train patient IDs
        valid_patients_list: List of arrays/lists, one per fold containing valid patient IDs
        test_patients_list: List of arrays/lists, one per fold containing test patient IDs
        output_path: Path to save the pickle file
    
    The saved structure is [train_patients_list, valid_patients_list, test_patients_list]
    where splits[0] = train, splits[1] = valid, splits[2] = test, and splits[i][fold] gives patients for that fold.
    """
    splits = [train_patients_list, valid_patients_list, test_patients_list]
    with open(output_path, 'wb') as f:
        pkl.dump(splits, f)
    print(f"Saved patient splits to {output_path}")

