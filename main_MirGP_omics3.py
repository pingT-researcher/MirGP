import os
import random
import argparse
import pickle
import yaml
from pathlib import Path

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

# Custom multi-omics deep learning model
from model.MirGP3 import MirGPModel

# ===================== Global Configuration & Hyperparameters =====================
PV_THRESH = 0.05
K_FOLD = 5
ORDER_MODE = "p_then_absr"
EPOCHS = 1000
ROOT_PATH = '/sRNA-MirGP'

# Set random seed to ensure experimental reproducibility
SEED = 123
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Configure computing device (GPU preferred)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ===================== Custom Multi-Omics Dataset Class =====================
class OmicsDataSet(torchData.Dataset):
    """
    Pytorch Dataset for three omics inputs: SNP, RNA-seq, small RNA-seq and phenotypic label
    """
    def __init__(self, snp, te, se, trait):
        # Convert numpy array to 32-bit float tensor
        self.snp = torch.from_numpy(snp).float()
        self.te = torch.from_numpy(te).float()
        self.se = torch.from_numpy(se).float()
        self.trait = torch.from_numpy(trait).float()

    def __len__(self):
        # Return total number of samples
        return len(self.trait)

    def __getitem__(self, idx):
        # Return single sample: SNP, RNA-seq, small RNA-seq, phenotype
        return self.snp[idx], self.te[idx], self.se[idx], self.trait[idx]


# ===================== Data Loading & Feature Selection Utility Functions =====================
def load_raw_data(data_name: str, dir_suffix: str = '') -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load omics feature matrix and phenotypic trait table from local files
    :param data_name: name of omics dataset
    :param dir_suffix: suffix for specified data folder path
    :return: trait dataframe, omics feature dataframe
    """
    if data_name == 'ori_G':
        z = np.load("/home/ljx/code/bio_1/data/snp_numeric_pipeline_outputs.npz", mmap_mode="r")
        samples = z["samples"].astype(str)
        X_ld = z["X_filled"].astype(np.float32)
        snps_ld = z["snps_sel"].astype(str)
        data = pd.DataFrame(X_ld, index=pd.Index(samples, name="sample_id"), columns=snps_ld)
        traits = pd.read_csv(f'{ROOT_PATH}/datasets/traits.csv', index_col=0, low_memory=False)
    elif data_name == 'SE':
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
    Calculate Pearson correlation coefficient and two-tailed p-value for each feature against phenotype
    :param snp_df: omics feature dataframe
    :param y: 1D phenotypic label array
    :return: correlation array, p-value array, valid finite value mask
    """
    y = pd.to_numeric(pd.Series(y), errors="coerce").values
    X = snp_df.to_numpy(dtype=np.float64, copy=False)
    y = y.astype(np.float64, copy=False)

    n = y.shape[0]
    # Centralize phenotypic values
    yc = y - y.mean()
    yc_ss = np.sqrt(np.sum(yc ** 2))

    # Centralize each feature column
    Xc = X - X.mean(axis=0, keepdims=True)
    Xc_ss = np.sqrt(np.sum(Xc ** 2, axis=0))

    # Compute Pearson correlation coefficient
    denom = Xc_ss * yc_ss
    num = Xc.T @ yc
    r = np.divide(num, denom, out=np.zeros_like(num), where=denom > 0)
    r = np.clip(r, -1.0, 1.0)

    # Calculate p-value via t-distribution
    df = max(n - 2, 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = r * np.sqrt(df / (1 - r ** 2))
    pval = 2 * stats.t.sf(np.abs(t), df)

    # Replace invalid p-values with 1.0
    valid_mask = np.isfinite(pval)
    pval[~valid_mask] = 1.0
    return r.astype(np.float32), pval.astype(np.float64), valid_mask


def select_features(ori_data: pd.DataFrame, y: np.ndarray) -> pd.DataFrame:
    """
    Filter features by p-value threshold, then sort features by statistical significance and correlation strength
    :param ori_data: raw omics feature dataframe
    :param y: phenotypic label array
    :return: filtered and sorted feature dataframe
    """
    r, pval, valid_mask = pearsonr_by_columns_df(ori_data, y)
    cols = ori_data.columns.to_numpy()

    # Filter features with valid finite p-values
    cols_valid = cols[valid_mask]
    r_valid = r[valid_mask]
    p_valid = pval[valid_mask]
    absr_valid = np.abs(r_valid)

    # Retain features with p-value < predefined threshold
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
            # Sort by ascending p-value first, then descending absolute correlation
            order = np.lexsort((-absr_keep, p_keep))
        else:
            # Sort only by descending absolute correlation
            order = np.argsort(-absr_keep)
        cols_sel = cols_keep[order]
    else:
        cols_sel = np.array([], dtype=object)

    return ori_data.loc[:, cols_sel]


def load_all_data(args) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load phenotype, SNP, RNA-seq and small RNA-seq datasets with consistent sample alignment
    :param args: command line arguments
    :return: traits, SNP matrix, RNA-seq matrix, small RNA-seq matrix
    """
    dir_suffix = '/250' if args.srna_seq != '' else ''
    traits, snp = load_raw_data('G', dir_suffix)
    _, rna_seq = load_raw_data(args.rna_seq, dir_suffix)

    if args.srna_seq:
        _, srna_seq = load_raw_data(args.srna_seq, dir_suffix)
    else:
        srna_seq = pd.DataFrame()

    return traits, snp, rna_seq, srna_seq


# ===================== Model Training & Inference Functions =====================
def model_Predict(model, te_loader, criterion) -> tuple[float, float, list, list]:
    """
    Conduct model inference on test set, compute loss and prediction performance
    :param model: trained MirGP3 multi-omics model
    :param te_loader: test set dataloader
    :param criterion: loss function
    :return: average test loss, Pearson correlation, ground truth list, predicted value list
    """
    total_loss = 0.0
    y_true = []
    y_pred = []
    model.eval()
    # Disable gradient computation to save memory and accelerate inference
    with torch.no_grad():
        for snp_X, te_X, se_X, test_y in te_loader:
            # Transfer all tensors to target device
            snp_X = snp_X.to(DEVICE)
            se_X = se_X.to(DEVICE)
            te_X = te_X.to(DEVICE)
            test_y = test_y.to(DEVICE)
            outputs = model(snp_X, te_X, se_X)
            loss = criterion(outputs.flatten(), test_y).item()
            total_loss += loss

            # Transfer prediction and label back to CPU for metric calculation
            y_pred.extend(outputs.flatten().cpu().detach().numpy())
            y_true.extend(test_y.cpu().detach().numpy())

    test_loss = total_loss / max(1, len(te_loader))
    pcc, _ = stats.pearsonr(y_true, y_pred)
    return test_loss, pcc, y_true, y_pred


def train_omics_trait(args, config, trait_name: str,
                       snp_df: pd.DataFrame, rna_df: pd.DataFrame, srna_df: pd.DataFrame,
                       y_label: np.ndarray) -> float:
    """
    Train and validate MirGP3 model for a single phenotypic trait
    :param args: command line arguments
    :param config: yaml configuration dictionary
    :param trait_name: target phenotypic trait name
    :param snp_df: SNP feature dataframe
    :param rna_df: RNA-seq feature dataframe
    :param srna_df: small RNA-seq feature dataframe
    :param y_label: phenotypic value array
    :return: optimal Pearson correlation achieved on test set
    """
    # Step 1: Feature screening for each omics layer
    snp_sel = select_features(snp_df, y_label)
    rna_sel = select_features(rna_df, y_label)
    srna_sel = select_features(srna_df, y_label)


    print(f"Processed [{trait_name}] SNP dimension: {snp_sel.shape} | RNA-Seq dimension: {rna_sel.shape} | sRNA-Seq dimension: {srna_sel.shape}")

    # Step 2: Split dataset into training set (80%) and test set (20%)
    snp_tr, snp_te, rna_tr, rna_te, srna_tr, srna_te, y_tr, y_te = train_test_split(
        snp_sel.values, rna_sel.values, srna_sel.values, y_label,
        test_size=0.2, random_state=42
    )

    # Step 3: Standardize features to eliminate dimensional differences
    snp_scaler = StandardScaler()
    snp_tr = snp_scaler.fit_transform(snp_tr)
    snp_te = snp_scaler.transform(snp_te)

    rna_scaler = StandardScaler()
    rna_tr = rna_scaler.fit_transform(rna_tr)
    rna_te = rna_scaler.transform(rna_te)

    srna_scaler = StandardScaler()
    srna_tr = srna_scaler.fit_transform(srna_tr)
    srna_te = srna_scaler.transform(srna_te)

    # Step 4: Build dataset and dataloader for batch training
    train_dataset = OmicsDataSet(snp_tr, rna_tr, srna_tr, y_tr)
    test_dataset = OmicsDataSet(snp_te, rna_te, srna_te, y_te)
    tr_loader = torchData.DataLoader(train_dataset, batch_size=32, shuffle=True)
    te_loader = torchData.DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Step 5: Initialize model, loss function and optimizer
    model_cfg = config['trait_model'][trait_name]
    # Fix bug: use assignment operator = instead of comparison ==
    model = MirGPModel(
        snp_tr.shape[1],
        rna_tr.shape[1],
        srna_tr.shape[1],
        hidden_dim=model_cfg['hidden'],
        conv_c2=model_cfg['conv_c2']
    ).to(DEVICE)

    criterion = nn.SmoothL1Loss(reduction='mean')
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001, eps=1e-8)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.5)

    # Step 6: Load pre-trained checkpoint if existing
    ckpt_name = f"{args.snp}+{args.rna_seq}+{args.srna_seq}_{trait_name}_MirGP.pt"
    ckpt_path = Path(ROOT_PATH) / 'checkpoints/mirgp' /'omics3' / ckpt_name

    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
        model.load_state_dict(ckpt["state_dict"])
        _, pcc, y_true, y_pred = model_Predict(model, te_loader, criterion)
        r2 = r2_score(y_true, y_pred)
        print(f"[{trait_name}] Load pre-trained checkpoint | PCC = {pcc:.4f} | R2 = {r2:.4f}\n")
        return pcc

    # Step 7: Train model from scratch
    best_pcc = -1.0
    best_true, best_pred = None, None
    save_dir = os.path.join(ROOT_PATH, "checkpoints/mirgp/omics3")
    os.makedirs(save_dir, exist_ok=True)

    for epoch in range(1, EPOCHS + 1):
        # Training phase
        model.train()
        train_loss = 0.0
        for snp_X, te_X, se_X, trait_y in tr_loader:
            snp_X = snp_X.to(DEVICE)
            te_X = te_X.to(DEVICE)
            se_X = se_X.to(DEVICE)
            trait_y = trait_y.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(snp_X, te_X, se_X)
            loss = criterion(outputs, trait_y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
        train_loss /= len(tr_loader)
        scheduler.step()

        # Validation phase on test set
        test_loss, curr_pcc, y_true, y_pred = model_Predict(model, te_loader, criterion)

        # Save model with optimal PCC performance
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

    # Output final optimal performance
    best_r2 = r2_score(best_true, best_pred)
    print(f"[{trait_name}] Training completed | Best PCC = {best_pcc:.4f} | Best R2 = {best_r2:.4f}\n")
    return best_pcc


# ===================== Main Program Entry =====================
def main():
    # Parse command line input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c",
                        default=f"{ROOT_PATH}/config/G+TE+SE.yaml",
                        type=str)
    parser.add_argument("--trait", "-t", type=str)
    parser.add_argument("--snp", type=str, default="G", choices=['G'])
    parser.add_argument("--rna_seq", type=str, default='TE', choices=['TE', 'GE'])
    parser.add_argument("--srna_seq", type=str, default="SE", choices=['CE', 'IE', 'SE', 'TRE'])
    args = parser.parse_args()

    # Load configuration parameters from yaml file
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)



    # Load all three omics datasets and phenotypic data
    traits_df, snp_df, rna_df, srna_df = load_all_data(args)
    print(f"Original data dimension -> SNP: {snp_df.shape} | RNA-Seq: {rna_df.shape} | sRNA-Seq: {srna_df.shape}\n")

    # Train independent prediction model for each phenotypic trait
    trait_list = traits_df.columns.tolist() if args.trait == None else [args.trait]
    all_pcc = []
    for trait_name in trait_list:
        y_label = traits_df[trait_name].values.astype(np.float32)
        pcc_val = train_omics_trait(args, config, trait_name, snp_df, rna_df, srna_df, y_label)
        if pcc_val != -999.0:
            all_pcc.append(pcc_val)

    # Print prediction performance of all traits
    print("PCC list for all phenotypic traits: ", all_pcc)


if __name__ == "__main__":
    main()