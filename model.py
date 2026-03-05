"""
HE2RNA: definition of the algorithm to generate a model for gene expression prediction
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
import torch
import time
import os
from contextlib import nullcontext
from datetime import datetime
from torch import nn
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from tqdm import tqdm


def pearson_correlation_loss(pred, target, eps=1e-8):
    """
    Compute Pearson correlation loss: (1 - PearsonCorr(pred, target)).
    This is differentiable and can be used as a loss term.
    
    Args:
        pred: predictions tensor of shape (batch_size, n_genes)
        target: target tensor of shape (batch_size, n_genes)
        eps: small epsilon for numerical stability
        
    Returns:
        loss: scalar tensor, mean of (1 - correlation) across genes
    """
    # Compute mean and std for each gene (across batch dimension)
    pred_mean = pred.mean(dim=0, keepdim=True)
    target_mean = target.mean(dim=0, keepdim=True)
    
    pred_centered = pred - pred_mean
    target_centered = target - target_mean
    
    pred_std = torch.sqrt((pred_centered ** 2).mean(dim=0) + eps)
    target_std = torch.sqrt((target_centered ** 2).mean(dim=0) + eps)
    
    # Compute covariance
    covariance = (pred_centered * target_centered).mean(dim=0)
    
    # Compute correlation (per gene)
    correlation = covariance / (pred_std * target_std + eps)
    
    # Return mean of (1 - correlation) across all genes
    # We want to maximize correlation, so we minimize (1 - correlation)
    loss = (1.0 - correlation).mean()
    return loss


def combined_loss(pred, target, mse_weight=1.0, corr_weight=1.0, use_relu=False):
    """
    Combined MSE + Pearson Correlation loss.
    
    Args:
        pred: predictions tensor
        target: target tensor
        mse_weight: weight for MSE component
        corr_weight: weight for correlation loss component
        use_relu: whether to apply ReLU to predictions before computing loss
        
    Returns:
        total_loss, mse_component, corr_component
    """
    # ReLU is already applied in the forward pass
    # if use_relu:
    #     pred = nn.ReLU()(pred)
    
    mse = nn.MSELoss()(pred, target)
    corr_loss = pearson_correlation_loss(pred, target)
    
    total_loss = mse_weight * mse + corr_weight * corr_loss
    return total_loss, mse, corr_loss


class HE2RNA(nn.Module):
    """Model that generates one score per tile and per predicted gene.

    Args
        output_dim (int): Output dimension, must match the number of genes to
            predict.
        layers (list): List of the layers' dimensions
        nonlin (torch.nn.modules.activation)
        ks (list): list of numbers of highest-scored tiles to keep in each
            channel.
        dropout (float)
        device (str): 'cpu' or 'cuda'
        mode (str): 'binary' or 'regression'
    """
    def __init__(self, input_dim, output_dim,
                 layers=[1], nonlin=nn.ReLU(), ks=[10],
                 dropout=0.5, device='cpu', proportional_ks=False,
                 bias_init=None, **kwargs):
        super(HE2RNA, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        layers = [input_dim] + layers + [output_dim]
        self.layers = []
        for i in range(len(layers) - 1):
            layer = nn.Conv1d(in_channels=layers[i],
                              out_channels=layers[i+1],
                              kernel_size=1,
                              stride=1,
                              bias=True)
            setattr(self, 'conv' + str(i), layer)
            self.layers.append(layer)
        if bias_init is not None:
            self.layers[-1].bias = bias_init
        self.ks = np.array(ks)
        self.proportional_ks = proportional_ks
        self.nonlin = nonlin
        self.do = nn.Dropout(dropout)
        self.device = device
        self.to(self.device)

    def forward(self, x):
        """
        x: (B, C_in, T_max) where each sample can have a different
        effective number of tiles (others are padded and masked out).
        """
        if self.training:
            # Pick a base k from the configured list; per‑sample proportional
            # scaling happens inside forward_fixed_k.
            k = int(np.random.choice(self.ks))
            return self.forward_fixed_k(x, k)
        else:  # EVALUATION MODE - Vectorized
            # 1. Compute scores exactly ONCE
            B, _, T_max = x.shape
            mask, _ = torch.max(x, dim=1, keepdim=True)
            # Keep mask in activation dtype to avoid implicit upcast to float32 under AMP.
            mask = (mask > 0).to(dtype=x.dtype)
            scores = self.conv(x) * mask  # (B, C_out, T_max)
            valid_counts_clamped = mask.sum(dim=2).squeeze(1).clamp(min=1)

            # 2. Sort the scores ONCE (descending)
            # sorted_scores: (B, C_out, T_max)
            sorted_scores, _ = torch.sort(scores, dim=2, descending=True)

            pred_sum = 0
            ref_max_k = float(np.max(self.ks))

            # 3. Slice for each k
            for k in self.ks:
                base_k = max(1, min(int(k), T_max))

                if self.proportional_ks:
                    # Calculate proportional k per sample
                    scale = valid_counts_clamped / ref_max_k  # (B,)
                    # Only scale when the slide has fewer tiles than ref_max_k
                    scale = torch.where(
                        valid_counts_clamped < ref_max_k,
                        scale,
                        torch.ones_like(scale),
                    )
                    k_per_sample = (base_k * scale).round().clamp(min=1).long()  # (B,)
                else:
                    # Same k for all samples (but cannot exceed their valid counts)
                    k_per_sample = torch.full_like(valid_counts_clamped, base_k, dtype=torch.long)

                # Do not request more tiles than each sample actually has
                k_per_sample = torch.minimum(k_per_sample, valid_counts_clamped.long())  # (B,)

                # For each sample, gather the top k_per_sample scores from sorted_scores
                # Since sorted_scores is already sorted descending, we can use a mask approach
                # or gather with indices. Using gather with indices is more efficient for
                # variable-length slices.

                # Create indices for gathering: for each sample, indices 0 to k_per_sample[i]-1
                max_k_for_gather = int(k_per_sample.max().item())
                max_k_for_gather = max(1, min(max_k_for_gather, T_max))

                # Take the top max_k_for_gather from sorted_scores (already sorted)
                top_k_scores = sorted_scores[:, :, :max_k_for_gather]  # (B, C_out, max_k_for_gather)

                # Build a mask to zero out positions >= k_per_sample for each sample
                range_k = torch.arange(max_k_for_gather, device=x.device).view(1, 1, -1)  # (1, 1, max_k_for_gather)
                k_per_sample_expanded = k_per_sample.view(B, 1, 1)  # (B, 1, 1)
                topk_mask = (range_k < k_per_sample_expanded).to(dtype=scores.dtype)  # (B, 1, max_k_for_gather)

                # Apply mask and compute mean
                masked_scores = top_k_scores * topk_mask  # (B, C_out, max_k_for_gather)
                numer = masked_scores.sum(dim=2)  # (B, C_out)
                denom = k_per_sample.unsqueeze(1).float().clamp(min=1.0)  # (B, 1)
                pred_sum += numer / denom

            final_pred = pred_sum / len(self.ks)
            return nn.ReLU()(final_pred)

    def forward_fixed_k(self, x, k):
        """
        Aggregate per sample using (potentially) different k for each image
        in the batch, depending on how many valid tiles it has.
        """
        # x: (B, C_in, T_max)
        B, _, T_max = x.shape

        # Build a binary mask of valid tiles based on the raw input.
        # mask: (B, 1, T_max) with 1 where tile is valid.
        mask, _ = torch.max(x, dim=1, keepdim=True)
        # Keep mask in activation dtype to avoid implicit upcast to float32 under AMP.
        mask = (mask > 0).to(dtype=x.dtype)

        # Run the conv stack on all tiles, then mask invalid ones.
        scores = self.conv(x) * mask  # (B, C_out, T_max)

        # Count valid tiles per sample (B,)
        valid_counts = mask.sum(dim=2).squeeze(1)  # (B,)
        valid_counts_clamped = valid_counts.clamp(min=1)  # avoid zeros

        # Clamp the base k to something sensible for this batch
        base_k = max(1, min(int(k), T_max))

        if self.proportional_ks:
            # Interpret ks[-1] as the reference "max k" used in config.
            # For samples with fewer valid tiles than this reference, we
            # scale k down proportionally. For samples with at least
            # ref_max_k valid tiles, we leave k unscaled.
            ref_max_k = float(np.max(self.ks))
            scale = valid_counts_clamped / ref_max_k  # (B,)
            # Only scale when the slide has fewer tiles than ref_max_k
            scale = torch.where(
                valid_counts_clamped < ref_max_k,
                scale,
                torch.ones_like(scale),
            )
            k_per_sample = (base_k * scale).round().clamp(min=1).long()  # (B,)
        else:
            # Same k for all samples (but cannot exceed their valid counts)
            k_per_sample = torch.full_like(valid_counts_clamped, base_k, dtype=torch.long)

        # Do not request more tiles than each sample actually has
        k_per_sample = torch.minimum(k_per_sample, valid_counts_clamped.long())  # (B,)

        # Global K we need for topk across the batch
        max_k = int(k_per_sample.max().item())
        max_k = max(1, min(max_k, T_max))

        # Take top max_k tiles per sample/channel
        # t: (B, C_out, max_k), idx: (B, C_out, max_k)
        t, idx = torch.topk(scores, max_k, dim=2, largest=True, sorted=True)

        # Gather the validity mask at top-k indices.
        # gather() requires the channel dimension to match idx, so expand mask
        # from (B,1,T_max) to (B,C_out,T_max) first.
        mask_expanded = mask.expand(-1, scores.shape[1], -1)  # (B, C_out, T_max)
        mask_topk = mask_expanded.gather(2, idx)  # (B, C_out, max_k)

        # For proportional_ks, only the first k_per_sample[i] entries in
        # the sorted top‑k list for sample i should contribute.
        # Build per‑sample mask over the k dimension.
        range_k = torch.arange(max_k, device=x.device).view(1, 1, -1)  # (1,1,max_k)
        k_expanded = k_per_sample.view(B, 1, 1)  # (B,1,1)
        per_sample_mask = (range_k < k_expanded).to(dtype=scores.dtype)  # (B,1,max_k)

        # Combined mask over selected tiles: valid AND within per‑sample k
        combined_mask = mask_topk * per_sample_mask  # (B,C_out,max_k)

        # Numerator: sum of scores over selected tiles
        numer = (t * combined_mask).sum(dim=2)  # (B, C_out)

        # Denominator: number of selected valid tiles; clamp to avoid div‑by‑zero
        denom = combined_mask.sum(dim=2).clamp_min(1.0)  # (B, C_out)

        out = numer / denom  # (B, C_out)
        return out

    def conv(self, x):
        x = x[:, x.shape[1] - self.input_dim:]
        for i in range(len(self.layers) - 1):
            x = self.do(self.nonlin(self.layers[i](x)))
        x = self.layers[-1](x)
        return x


def _unpack_batch(batch):
    if isinstance(batch, (list, tuple)):
        if len(batch) < 2:
            raise ValueError("Batch must contain at least (features, targets).")
        return batch[0], batch[1]
    raise ValueError("Unsupported batch type. Expected tuple/list.")


def _autocast_context(use_mixed_precision=False, amp_dtype=torch.float16):
    """Return an autocast context manager when AMP is enabled on CUDA."""
    if use_mixed_precision and torch.cuda.is_available():
        return torch.autocast(device_type='cuda', dtype=amp_dtype)
    return nullcontext()


def training_epoch(model, dataloader, optimizer, 
                   loss_mode='mse', mse_weight=1.0, corr_weight=1.0,
                   use_mixed_precision=False, amp_dtype=torch.float16, scaler=None):
    """Train model for one epoch.
    
    Args:
        model: the HE2RNA model
        dataloader: training data loader
        optimizer: optimizer
        loss_mode: 'mse' for pure MSE, 'combined' for MSE + Pearson Correlation
        mse_weight: weight for MSE component (used in combined mode)
        corr_weight: weight for correlation loss component (used in combined mode)
    """
    model.train()
    train_loss = []
    train_mse = []
    train_corr_loss = []
    
    for batch in tqdm(dataloader):
        x, y = _unpack_batch(batch)
        x = x.float().to(model.device)
        y = y.float().to(model.device)

        optimizer.zero_grad(set_to_none=True)
        with _autocast_context(use_mixed_precision=use_mixed_precision, amp_dtype=amp_dtype):
            # new: ReLU activation is already applied in the forward pass
            # ReLU activation matching inference (evaluate) function
            # pred = nn.ReLU()(model(x))
            pred = model(x)

            if loss_mode == 'combined':
                loss, mse, corr_loss = combined_loss(
                    pred, y, 
                    mse_weight=mse_weight, 
                    corr_weight=corr_weight
                )
                train_mse.append(mse.detach().cpu().numpy())
                train_corr_loss.append(corr_loss.detach().cpu().numpy())
            else:
                # Pure MSE mode (original behavior) 
                loss = nn.MSELoss()(pred, y)

        train_loss.append(loss.detach().cpu().numpy())
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
    
    results = {'total': float(np.mean(train_loss))}
    if loss_mode == 'combined':
        results['mse'] = float(np.mean(train_mse))
        results['corr_loss'] = float(np.mean(train_corr_loss))
    
    return results

def compute_correlations_old(labels, preds, projects):
    metrics = []
    for project in np.unique(projects):
        for i in range(labels.shape[1]):
            y_true = labels[projects == project, i]
            if len(np.unique(y_true)) > 1:
                y_prob = preds[projects == project, i]
                # Check for constant prediction which causes NaN correlation
                if np.std(y_prob) == 0:
                    print(f"WARNING: Constant prediction detected for project {project}, gene index {i}. Setting correlation to 0.")
                    metrics.append(0.0)
                else:
                    corr = np.corrcoef(y_true, y_prob)[0, 1]
                    metrics.append(0.0 if np.isnan(corr) else corr)
    metrics = np.asarray(metrics)
    return np.nanmean(metrics) if len(metrics) > 0 else 0.0

def compute_correlations(labels, preds, projects):
    metrics = []
    unique_projects = np.unique(projects)
    
    for project in unique_projects:
        # Select data for this project
        mask = (projects == project)
        p_labels = labels[mask]
        p_preds = preds[mask]
        
        # 1. Calculate Standard Deviations (per gene)
        labels_std = np.std(p_labels, axis=0)
        preds_std = np.std(p_preds, axis=0)

        # Genes with (nearly) constant labels are skipped,
        # to mirror: "if len(np.unique(y_true)) > 1" in compute_correlations_old
        varying_labels_mask = labels_std > 1e-12
        if not np.any(varying_labels_mask):
            continue

        # Restrict to genes with varying labels
        p_labels_var = p_labels[:, varying_labels_mask]
        p_preds_var = p_preds[:, varying_labels_mask]
        preds_std_var = preds_std[varying_labels_mask]

        # 2. Center the data (subtract mean)
        p_labels_c = p_labels_var - np.mean(p_labels_var, axis=0)
        p_preds_c = p_preds_var - np.mean(p_preds_var, axis=0)

        # 3. Vectorized Correlation Calculation
        numerator = np.sum(p_labels_c * p_preds_c, axis=0)
        denominator = (
            np.sqrt(np.sum(p_labels_c ** 2, axis=0)) *
            np.sqrt(np.sum(p_preds_c ** 2, axis=0))
        )

        with np.errstate(divide='ignore', invalid='ignore'):
            corrs = numerator / denominator

        # Split behavior based on prediction variability, matching compute_correlations_old:
        # - If preds are constant (std ~ 0): append 0.0
        # - Else: append correlation (NaNs mapped to 0.0)
        const_pred_mask = preds_std_var < 1e-12
        var_pred_mask = ~const_pred_mask

        # Constant predictions for varying labels → 0.0
        if np.any(const_pred_mask):
            metrics.extend([0.0] * int(np.sum(const_pred_mask)))

        # Non-constant predictions → use correlation, with NaNs mapped to 0.0
        if np.any(var_pred_mask):
            valid_corrs = corrs[var_pred_mask]
            valid_corrs = np.where(np.isnan(valid_corrs), 0.0, valid_corrs)
            metrics.extend(valid_corrs.tolist())

    return np.nanmean(metrics) if len(metrics) > 0 else 0.0

def evaluate(model, dataloader, projects, use_mixed_precision=False, amp_dtype=torch.float16):
    """Evaluate the model on the validation set and return loss and metrics.
    """
    model.eval()
    loss_fn = nn.MSELoss()
    valid_loss = []
    preds = []
    labels = []
    with torch.no_grad():
        for batch in dataloader:
            x, y = _unpack_batch(batch)
            x = x.float().to(model.device)
            y_device = y.float().to(model.device)
            with _autocast_context(use_mixed_precision=use_mixed_precision, amp_dtype=amp_dtype):
                pred = model(x)
                loss = loss_fn(pred, y_device)
            labels += [y]
            valid_loss += [loss.detach().cpu().numpy()]
            # new: ReLU activation is already applied in the forward pass
            # pred = nn.ReLU()(pred)
            preds += [pred.detach().cpu().numpy()]
    valid_loss = np.mean(valid_loss)
    preds = np.concatenate(preds)
    labels = np.concatenate(labels)
    metrics = compute_correlations(labels, preds, projects)
    return valid_loss, metrics


def predict(model, dataloader, use_mixed_precision=False, amp_dtype=torch.float16):
    """Perform prediction on the test set.
    """
    model.eval()
    labels = []
    preds = []
    with torch.no_grad():
        for batch in dataloader:
            x, y = _unpack_batch(batch)
            x = x.float().to(model.device)
            with _autocast_context(use_mixed_precision=use_mixed_precision, amp_dtype=amp_dtype):
                pred = model(x)
            labels += [y]
            pred = nn.ReLU()(pred)
            preds += [pred.detach().cpu().numpy()]
    preds = np.concatenate(preds)
    labels = np.concatenate(labels)
    return preds, labels


def fit(model,
        train_set,
        valid_set,
        valid_projects,
        params={},
        optimizer=None,
        test_set=None,
        path=None,
        logdir='./exp',
        train_loader=None,
        loss_mode='mse',
        mse_weight=1.0,
        corr_weight=1.0):
    """Fit the model and make prediction on evaluation set.

    Args:
        model (nn.Module)
        train_set (torch.utils.data.Dataset)
        valid_set (torch.utils.data.Dataset)
        valid_projects (np.array): list of integers encoding the projects
            validation samples belong to.
        params (dict): Dictionary for specifying training parameters.
            keys are 'max_epochs' (int, default=200), 'patience' (int,
            default=20) and 'batch_size' (int, default=16).
        optimizer (torch.optim.Optimizer): Optimizer for training the model
        test_set (None or torch.utils.data.Dataset): If None, return
            predictions on the validation set.
        path (str): Path to the folder where th model will be saved.
        logdir (str): Path for TensoboardX.
        loss_mode (str): 'mse' for pure MSE, 'combined' for MSE + Pearson Correlation
        mse_weight (float): Weight for MSE component in combined loss
        corr_weight (float): Weight for correlation loss component
    """

    if path is not None and not os.path.exists(path):
        os.mkdir(path)

    default_params = {
        'max_epochs': 200,
        'patience': 20,
        'batch_size': 16,
        'num_workers': 0,
        'mixed_precision': False,
        'mixed_precision_dtype': 'float16'}
    default_params.update(params)
    batch_size = default_params['batch_size']
    patience = default_params['patience']
    max_epochs = default_params['max_epochs']
    num_workers = default_params['num_workers']
    use_mixed_precision = bool(default_params.get('mixed_precision', False))
    mixed_precision_dtype = str(default_params.get('mixed_precision_dtype', 'float16')).lower()

    if use_mixed_precision and (not torch.cuda.is_available() or 'cuda' not in str(model.device).lower()):
        print("Mixed precision requested but CUDA device is unavailable. Falling back to full precision.")
        use_mixed_precision = False

    if mixed_precision_dtype == 'bfloat16':
        amp_dtype = torch.bfloat16
    elif mixed_precision_dtype in ['float16', 'fp16']:
        amp_dtype = torch.float16
    else:
        raise ValueError(
            f"Unsupported mixed_precision_dtype '{mixed_precision_dtype}'. "
            "Use 'float16' or 'bfloat16'."
        )

    # GradScaler is only required for float16 AMP.
    scaler = None
    if use_mixed_precision and amp_dtype == torch.float16:
        scaler = torch.cuda.amp.GradScaler()
    print(
        f"Mixed precision active: {use_mixed_precision} "
        f"(dtype={str(amp_dtype).replace('torch.', '')})"
    )

    writer = SummaryWriter(log_dir=logdir, flush_secs=30)

    # SET num_workers TO 0 WHEN WORKING WITH hdf5 FILES
    if train_loader is None:
        train_loader = DataLoader(
            train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    if valid_set is not None:
        valid_loader = DataLoader(
            valid_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    if test_set is not None:
        test_loader = DataLoader(
            test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    if optimizer is None:
        optimizer = torch.optim.Adam(list(model.parameters()), lr=1e-3,
                                     weight_decay=0.)

    metrics = 'correlations'
    epoch_since_best = 0
    start_time = time.time()
    history = []

    if valid_set is not None:
        valid_loss, best = evaluate(
            model, valid_loader, valid_projects,
            use_mixed_precision=use_mixed_precision, amp_dtype=amp_dtype)
        print('{}: {:.3f}'.format(metrics, best))
        if np.isnan(best):
            best = 0
        if test_set is not None:
            preds, labels = predict(
                model, test_loader,
                use_mixed_precision=use_mixed_precision, amp_dtype=amp_dtype)
        else:
            preds, labels = predict(
                model, valid_loader,
                use_mixed_precision=use_mixed_precision, amp_dtype=amp_dtype)

    try:

        for e in range(max_epochs):

            epoch_since_best += 1
            if hasattr(train_loader, 'batch_sampler') and hasattr(train_loader.batch_sampler, 'set_epoch'):
                train_loader.batch_sampler.set_epoch(e)

            train_results = training_epoch(
                model, train_loader, optimizer,
                loss_mode=loss_mode,
                mse_weight=mse_weight,
                corr_weight=corr_weight,
                use_mixed_precision=use_mixed_precision,
                amp_dtype=amp_dtype,
                scaler=scaler
            )
            
            # Handle both dict and scalar returns for backward compatibility
            if isinstance(train_results, dict):
                train_loss = train_results['total']
                dic_loss = {'train_loss': train_loss}
                if 'mse' in train_results:
                    dic_loss['train_mse'] = train_results['mse']
                if 'corr_loss' in train_results:
                    dic_loss['train_corr_loss'] = train_results['corr_loss']
            else:
                train_loss = train_results
                dic_loss = {'train_loss': train_loss}

            print('Epoch {}/{} - {:.2f}s'.format(
                e + 1,
                max_epochs,
                time.time() - start_time))
            start_time = time.time()
            if hasattr(train_loader, 'batch_sampler') and hasattr(train_loader.batch_sampler, 'observed_counts'):
                counts = train_loader.batch_sampler.observed_counts
                total_seen = counts.get('cscc', 0) + counts.get('tcga', 0)
                if total_seen > 0:
                    cscc_ratio = counts['cscc'] / total_seen
                    print(
                        f"Train domain mix this epoch: CSCC={counts['cscc']} "
                        f"TCGA={counts['tcga']} ratio_cscc={cscc_ratio:.3f}"
                    )

            if valid_set is not None:
                valid_loss, scores = evaluate(
                    model, valid_loader, valid_projects,
                    use_mixed_precision=use_mixed_precision, amp_dtype=amp_dtype)
                dic_loss['valid_loss'] = valid_loss
                score = np.mean(scores)
                history.append(score)
                writer.add_scalars('data/losses',
                                   dic_loss,
                                   e)
                writer.add_scalar('data/metrics', score, e)
                if loss_mode == 'combined' and 'train_mse' in dic_loss:
                    print('loss: {:.4f} (mse={:.4f}, corr_loss={:.4f}), val loss: {:.4f}'.format(
                        train_loss,
                        dic_loss['train_mse'],
                        dic_loss['train_corr_loss'],
                        valid_loss))
                else:
                    print('loss: {:.4f}, val loss: {:.4f}'.format(
                        train_loss,
                        valid_loss))
                print('{}: {:.3f}'.format(metrics, score))
            else:
                writer.add_scalars('data/losses',
                                   dic_loss,
                                   e)
                if loss_mode == 'combined' and 'train_mse' in dic_loss:
                    print('loss: {:.4f} (mse={:.4f}, corr_loss={:.4f})'.format(
                        train_loss,
                        dic_loss['train_mse'],
                        dic_loss['train_corr_loss']))
                else:
                    print('loss: {:.4f}'.format(train_loss))

            if valid_set is not None:
                criterion = (score > best)

                if criterion:
                    print(f"Epoch {e + 1} - New best score: {score:.3f}\nSaving model...\nDate & time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                    epoch_since_best = 0
                    best = score
                    if path is not None:
                        torch.save(model, os.path.join(path, 'model.pt'))
                    elif test_set is not None:
                        preds, labels = predict(
                            model, test_loader,
                            use_mixed_precision=use_mixed_precision, amp_dtype=amp_dtype)
                    else:
                        preds, labels = predict(
                            model, valid_loader,
                            use_mixed_precision=use_mixed_precision, amp_dtype=amp_dtype)

                if epoch_since_best == patience:
                    # Velocity check to override early stopping
                    window = min(len(history), patience)
                    recent_history = history[-window:]
                    if len(recent_history) > 1:
                        x_idxs = np.arange(len(recent_history))
                        slope, _ = np.polyfit(x_idxs, recent_history, 1)
                        if slope > 1e-4:
                            print(f"Patience reached but model is improving (slope: {slope:.5f}). Giving 10 more epochs chance to reach new best... ")
                            epoch_since_best = patience - 10
                        else:
                            print('Early stopping at epoch {}'.format(e + 1))
                            break
                    else:
                        print('Early stopping at epoch {}'.format(e + 1))
                        break

    except KeyboardInterrupt:
        pass

    if path is not None and os.path.exists(os.path.join(path, 'model.pt')):
        model = torch.load(os.path.join(path, 'model.pt'))

    elif path is not None:
        # Edge case: model has never improved so no model.pt file exists yet
        torch.save(model, os.path.join(path, 'model.pt'))

    if test_set is not None:
        preds, labels = predict(
            model, test_loader,
            use_mixed_precision=use_mixed_precision, amp_dtype=amp_dtype)
    elif valid_set is not None:
        preds, labels = predict(
            model, valid_loader,
            use_mixed_precision=use_mixed_precision, amp_dtype=amp_dtype)
    else:
        preds = None
        labels = None

    writer.close()

    return preds, labels
