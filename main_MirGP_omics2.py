import os
import random
import argparse
import pickle
import yaml
from pathlib import Path
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

import torch
import torch.utils.data as torchData
import torch.nn as nn
import torch.optim as optim

# Custom model module
from model.MirGP2 import MirGPModel

# ===================== Global Configuration & Hyperparameters =====================
PV_THRESH = 0.05
ORDER_MODE = "p_then_absr"
EPOCHS = 1000
ROOT_PATH = '/MirGP'

# Set random seed for reproducibility
SEED = 123
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Device configuration (GPU if available, otherwise CPU)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ===================== Custom Omics Dataset Class =====================
class OmicsDataSet(torchData.Dataset):
    """
    Custom PyTorch Dataset for multi-omics input (SNP + sequencing features) and phenotype label
    """
    def __init__(self, snp, seq, trait):
        # Convert numpy array to torch float tensor
        self.snp = torch.from_numpy(snp).float()
        self.seq = torch.from_numpy(seq).float()
        self.trait = torch.from_numpy(trait).float()

    def __len__(self):
        # Return total number of samples
        return len(self.trait)

    def __getitem__(self, idx):
        # Return single sample by index: snp feature, sequencing feature, phenotype label
        return self.snp[idx], self.seq[idx], self.trait[idx]


# ===================== Data Loading & Feature Selection Utility Functions =====================
def load_raw_data(data_name: str, dir_suffix: str = '') -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load raw omics feature data and phenotype trait table from specified file path
    :param data_name: name of omics dataset
    :param dir_suffix: suffix for data folder path
    :return: trait dataframe, omics feature dataframe
    """
    if data_name == 'SE':
        traits = pd.read_csv(f'{ROOT_PATH}/datasets/250/traits.csv', index_col=0, low_memory=False)
        data = pd.read_csv(f'{ROOT_PATH}/datasets/250/{data_name}.csv', index_col=0, low_memory=False)
        data = data.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    else:
        traits = pd.read_csv(f'{ROOT_PATH}/datasets{dir_suffix}/traits.csv', index_col=0, low_memory=False)
        data = pd.read_csv(f'{ROOT_PATH}/datasets{dir_suffix}/{data_name}.csv', index_col=0, low_memory=False)
        data = data.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return traits, data


def pearsonr_by_columns_df(snp_df: pd.DataFrame, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate Pearson correlation coefficient and p-value for each feature column against phenotype
    :param snp_df: omics feature dataframe
    :param y: 1D phenotype label array
    :return: correlation array, p-value array, valid finite value mask
    """
    y = pd.to_numeric(pd.Series(y), errors="coerce").values
    X = snp_df.to_numpy(dtype=np.float64, copy=False)
    y = y.astype(np.float64, copy=False)

    n = y.shape[0]
    # Centralize label
    yc = y - y.mean()
    yc_ss = np.sqrt(np.sum(yc ** 2))

    # Centralize each feature column
    Xc = X - X.mean(axis=0, keepdims=True)
    Xc_ss = np.sqrt(np.sum(Xc ** 2, axis=0))

    # Compute Pearson correlation
    denom = Xc_ss * yc_ss
    num = Xc.T @ yc
    r = np.divide(num, denom, out=np.zeros_like(num), where=denom > 0)
    r = np.clip(r, -1.0, 1.0)

    # Two-tailed t-test for p-value calculation
    df = max(n - 2, 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = r * np.sqrt(df / (1 - r ** 2))
    pval = 2 * stats.t.sf(np.abs(t), df)

    # Mask invalid p-values and replace with 1.0
    valid_mask = np.isfinite(pval)
    pval[~valid_mask] = 1.0
    return r.astype(np.float32), pval.astype(np.float64), valid_mask


def select_features(ori_data: pd.DataFrame, y: np.ndarray) -> pd.DataFrame:
    """
    Feature filtering based on Pearson p-value threshold, then sort features
    Sort rule: first by p-value ascending, then by absolute correlation descending
    :param ori_data: original feature dataframe
    :param y: phenotype label array
    :return: filtered and sorted feature dataframe
    """
    r, pval, valid_mask = pearsonr_by_columns_df(ori_data, y)
    cols = ori_data.columns.to_numpy()

    # Filter valid features with finite p-values
    cols_valid = cols[valid_mask]
    r_valid = r[valid_mask]
    p_valid = pval[valid_mask]
    absr_valid = np.abs(r_valid)

    # Keep features with p-value less than threshold
    keep = (p_valid < PV_THRESH)
    if keep.any():
        cols_keep = cols_valid[keep]
        p_keep = p_valid[keep]
        absr_keep = absr_valid[keep]
    else:
        cols_keep = cols_valid
        p_keep = p_valid
        absr_keep = absr_valid

    # Sort selected features
    if cols_keep.size:
        if ORDER_MODE == "p_then_absr":
            # Sort by p-value ascending, then absolute correlation descending
            order = np.lexsort((-absr_keep, p_keep))
        else:
            # Sort only by absolute correlation descending
            order = np.argsort(-absr_keep)
        cols_sel = cols_keep[order]
    else:
        cols_sel = np.array([], dtype=object)
    
   
    return ori_data.loc[:, cols_sel]


def load_all_data(args) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load phenotype table, SNP feature matrix and sequencing feature matrix
    :param args: command line arguments
    :return: traits dataframe, SNP dataframe, sequencing dataframe
    """
    dir_suffix = '/250' if args.seq == 'SE' else ''
    traits, snp = load_raw_data('G', dir_suffix)
    _, seq = load_raw_data(args.seq, dir_suffix)
    return traits, snp, seq


# ===================== Model Training & Inference Functions =====================
def model_Predict(model, te_loader, criterion) -> tuple[float, float, list, list]:
    """
    Model inference on test set, calculate test loss and Pearson correlation
    :param model: trained MirGP model
    :param te_loader: test set dataloader
    :param criterion: loss function
    :return: average test loss, Pearson correlation, ground truth list, prediction list
    """
    total_loss = 0.0
    y_true = []
    y_pred = []

    model.eval()
    # Disable gradient computation for inference acceleration
    with torch.no_grad():
        for snp_X, seq_X, test_y in te_loader:
            # Move tensor to target device
            snp_X = snp_X.to(DEVICE)
            seq_X = seq_X.to(DEVICE)
            test_y = test_y.to(DEVICE)
            outputs = model(snp_X, seq_X)
            loss = criterion(outputs.flatten(), test_y).item()
            total_loss += loss

            # Collect predictions and labels, transfer back to CPU
            y_pred.extend(outputs.flatten().cpu().detach().numpy())
            y_true.extend(test_y.cpu().detach().numpy())

    test_loss = total_loss / max(1, len(te_loader))
    pcc, _ = stats.pearsonr(y_true, y_pred)
    return test_loss, pcc, y_true, y_pred


def train_omics_trait(args, config, trait_name: str,
                       snp_df: pd.DataFrame, seq_df: pd.DataFrame,
                       y_label: np.ndarray) -> float:
    """
    Train MirGP model for single phenotype trait, save best checkpoint
    :param args: command line arguments
    :param config: yaml configuration dict
    :param trait_name: target phenotype name
    :param snp_df: filtered SNP feature dataframe
    :param seq_df: filtered sequencing feature dataframe
    :param y_label: 1D phenotype label arrayd
    :return: best Pearson correlation on test set
    """
    # Step 1: Feature selection by correlation & p-valued
    snp_sel= select_features(snp_df, y_label)
    seq_sel = select_features(seq_df, y_label)

    print(f"Processed [{trait_name}] SNP dimension: {snp_sel.shape} | Seq dimension: {seq_sel.shape}")
    
   

    # Step 2: Split dataset into train set (80%) and test set (20%)
    snp_tr, snp_te, seq_tr, seq_te, y_tr, y_te = train_test_split(
        snp_sel.values, seq_sel.values, y_label,
        test_size=0.2, random_state=42
    )

    # Step 3: Standard normalization for omics features
    snp_scaler = StandardScaler()
    snp_tr = snp_scaler.fit_transform(snp_tr)
    snp_te = snp_scaler.transform(snp_te)

    seq_scaler = StandardScaler()
    seq_tr = seq_scaler.fit_transform(seq_tr)
    seq_te = seq_scaler.transform(seq_te)

    # Step 4: Build dataset and dataloader
    train_dataset = OmicsDataSet(snp_tr, seq_tr, y_tr)
    test_dataset = OmicsDataSet(snp_te, seq_te, y_te)
   
    # Step 5: Initialize model, loss function and optimizer
    model_cfg = config['trait_model'][trait_name]
    tr_loader = torchData.DataLoader(train_dataset, batch_size=model_cfg['bs'], shuffle=True)
    te_loader = torchData.DataLoader(test_dataset, batch_size=model_cfg['bs'], shuffle=False)

    model = MirGPModel(
        snp_tr.shape[1],
        seq_tr.shape[1],
        hidden_dim=model_cfg['hidden'],
        conv_c2=model_cfg['conv_c2']
    ).to(DEVICE)

    criterion = nn.SmoothL1Loss(reduction='mean')
    optimizer = optim.AdamW(model.parameters(), lr=model_cfg['lr'], weight_decay=1e-4)
    # Step 6: Load pre-trained checkpoint if exists
    ckpt_name = f"G+{args.seq_data}_{trait_name}_MirGP.pt"
    ckpt_path = Path(ROOT_PATH) / 'checkpoints/mirgp' /'omics2' / ckpt_name
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
        model.load_state_dict(ckpt["state_dict"])
        _, pcc, y_true, y_pred = model_Predict(model, te_loader, criterion)
        r2 = r2_score(y_true, y_pred)
        print(f"[{trait_name}] Load existing checkpoint | PCC = {pcc:.4f} | R2 = {r2:.4f}\n")
        return pcc

    # Step 7: Train model from scratch
    best_pcc = -1.0
    best_true, best_pred = None, None
    save_dir = os.path.join(ROOT_PATH, "checkpoints/mirgp/omics2")
    os.makedirs(save_dir, exist_ok=True)

    for epoch in range(1, EPOCHS + 1):
        # Training phase
        model.train()
        train_loss = 0.0
        for snp_X, seq_X, trait_y in tr_loader:
            snp_X = snp_X.to(DEVICE)
            seq_X = seq_X.to(DEVICE)
            trait_y = trait_y.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(snp_X, seq_X)
            loss = criterion(outputs, trait_y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
        train_loss /= len(tr_loader)
        

        # Evaluation phase on test set
        test_loss, curr_pcc, y_true, y_pred = model_Predict(model, te_loader, criterion)

        # Save model with best PCC performance
        if curr_pcc > best_pcc:
            best_pcc = float(curr_pcc)
            best_true = np.array(y_true, dtype=np.float32)
            best_pred = np.array(y_pred, dtype=np.float32)

            torch.save({
                "trait": trait_name,
                "state_dict": model.state_dict()
            }, os.path.join(save_dir, ckpt_name))

        # Print training log every 50 epochs
        if epoch % 50 == 0:
            print(f"[{trait_name}] Epoch {epoch:04d} | TrainLoss: {train_loss:.8f} | TestLoss: {test_loss:.8f}")

    # Summary training result
    best_r2 = r2_score(best_true, best_pred)
    print(f"[{trait_name}] Training finished | Best PCC = {best_pcc:.4f} | Best R2 = {best_r2:.4f}\n")
    return best_pcc


# ===================== Main Program Entry =====================
def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c",
                        default=f"{ROOT_PATH}/config/G+SE.yaml",
                        type=str)
    parser.add_argument("--trait", "-t", type=str)
    parser.add_argument("--snp", type=str, default="G", choices=['G'])
    # TE: transcriptome data; SE: small RNA sequencing data
    parser.add_argument("--seq", type=str, default='SE', choices=['TE', 'SE'])
    parser.add_argument("--seq_data", type=str, default="SE", choices=['TE','GE','CE', 'IE', 'SE', 'TRE'])
    args = parser.parse_args()

    # Load yaml configuration file
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Load all multi-omics and phenotype data
    traits_df, snp_df, seq_df = load_all_data(args)
    print(f"Original dimension -> SNP: {snp_df.shape} | Seq: {seq_df.shape}\n")

    # Iterate each phenotype for independent model training
    trait_list = traits_df.columns.tolist() if args.trait == None else [args.trait]
    all_pcc = []
    for trait_name in trait_list:
        y_label = traits_df[trait_name].values.astype(np.float32)
        pcc_val = train_omics_trait(args, config, trait_name, snp_df, seq_df, y_label)
        if pcc_val != -999.0:
            all_pcc.append(pcc_val)

    # Output all test set PCC results
    print("PCC list for all traits: ", all_pcc)


if __name__ == "__main__":
    main()