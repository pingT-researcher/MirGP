import math
import random
import argparse
import os
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.linear_model import Ridge
import joblib

import torch
import torch.utils.data as torchData
from pathlib import Path

# Custom deep learning models for omics prediction
from model.DNNGP import DNNGPModel
from model.PNNGS import PNNGSModel
from model.SoyDNGP import SoyDNGPModel

# ===================== Global Configuration & Random Seed Setup =====================
SEED = 123
# Fix random seed for experimental reproducibility
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
# Enable deterministic CUDNN algorithm to guarantee identical results
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Hyperparameters for model training and feature selection
EPOCHS = 1000
PV_THRESH = 0.05
ORDER_MODE = "p_then_absr"
ROOT_PATH = '.'

# Suppress FutureWarning thrown by torch.load
warnings.filterwarnings("ignore", category=FutureWarning)
# Configure computing device (GPU preferred)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ===================== Custom Omics Dataset Class =====================
class OmicsDataSet(torchData.Dataset):
    """
    PyTorch Dataset class for single-modal omics features and phenotypic labels
    """
    def __init__(self, data: np.ndarray, trait: np.ndarray):
        # Convert numpy array to 32-bit floating point tensor
        self.data = torch.from_numpy(data).float()
        self.trait = torch.from_numpy(trait).float()

    def __len__(self):
        # Return total number of samples
        return len(self.trait)

    def __getitem__(self, idx):
        # Return feature and corresponding phenotypic label for one sample
        return self.data[idx], self.trait[idx]


# ===================== General Utility Functions =====================
def compute_k(k1: int, nl: int) -> tuple[int, int]:
    """
    Calculate dimension parameters k2 and k3 for SoyDNGP convolutional network
    :param k1: original input feature dimension after reshaping
    :param nl: number of network layers
    :return: k2 (feature map size), k3 (total feature dimension after convolution)
    """
    b = [32, 32, 64, 64, 64, 128, 128, 256, 256, 512, 512, 1024, 1024, 1024]
    k2 = k1
    if nl >= 11:
        tmp = (k1 - 2) // 2 + 1
        tmp = (tmp - 1) // 2 + 1
        tmp = tmp // 2
        tmp = tmp // 2
        k2 = (tmp - 1) // 2 + 1
    elif 9 <= nl < 11:
        tmp = (k1 - 2) // 2 + 1
        tmp = (tmp - 1) // 2 + 1
        tmp = tmp // 2
        k2 = tmp // 2
    elif 7 <= nl < 9:
        tmp = (k1 - 2) // 2 + 1
        tmp = (tmp - 1) // 2 + 1
        k2 = tmp // 2
    elif 3 <= nl < 7:
        tmp = (k1 - 2) // 2 + 1
        k2 = (tmp - 1) // 2 + 1
    elif nl == 2:
        k2 = (k1 - 2) // 2 + 1
    k3 = b[nl] * k2 * k2
    return k2, k3


def steps_to_two(n: int) -> int:
    """
    Map original feature dimension to the number of convolutional layers for SoyDNGP
    by repeatedly dividing the dimension by 2 until it is less than or equal to 2
    :param n: input feature dimension
    :return: predefined number of network layers
    """
    l = [2, 3, 7, 9, 11]
    if n < 2:
        return None
    count = 0
    while n > 2:
        n //= 2
        count += 1
    if count > 4:
        return 13
    return l[count - 1] if n <= 2 else 13


def pearsonr_by_columns_df(snp_df: pd.DataFrame, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate Pearson correlation coefficient and two-tailed p-value for each feature against phenotype
    :param snp_df: omics feature matrix dataframe
    :param y: 1D array of phenotypic values
    :return: correlation array, p-value array, boolean mask for valid finite values
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

    # Calculate p-value based on t-distribution
    df = max(n - 2, 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = r * np.sqrt(df / (1 - r ** 2))
    pval = 2 * stats.t.sf(np.abs(t), df)
    valid_mask = np.isfinite(pval)
    # Fill invalid p-values with 1.0 to exclude these features
    pval[~valid_mask] = 1.0
    return r.astype(np.float32), pval.astype(np.float64), valid_mask


# ===================== Data Processing Functions =====================
def load_raw_data(data_name: str, dir='') -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load phenotypic trait table and omics feature matrix from local files
    :param data_name: name of omics dataset
    :param dir: folder suffix for special datasets
    :return: trait dataframe, omics feature dataframe
    """
    if data_name == 'SE':
        traits = pd.read_csv(f'{ROOT_PATH}/datasets/250/traits.csv', index_col=0, low_memory=False)
        data = pd.read_csv(f'{ROOT_PATH}/datasets/250/{data_name}.csv', index_col=0, low_memory=False)
        data = data.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    else:
        traits = pd.read_csv(f'{ROOT_PATH}/datasets{dir}/traits.csv', index_col=0, low_memory=False)
        data = pd.read_csv(f'{ROOT_PATH}/datasets{dir}/{data_name}.csv', index_col=0, low_memory=False)
        data = data.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return traits, data


def load_all_data(data_name: str):
    """
    Load multiple omics datasets and store feature matrices in a list for subsequent horizontal concatenation
    :param data_name: concatenated dataset names separated by "+"
    :return: phenotypic DataFrame, list containing each individual omics feature DataFrame
    """
    data_names = data_name.split("+")
    full_data = []
    # Assign corresponding folder path for small RNA datasets
    if any(x in data_names for x in ['CE', 'IE', 'SE', 'TRE']):
        dir = '/250'
    else:
        dir = ''
    for name in data_names:
        traits, data = load_raw_data(name, dir)
        full_data.append(data)
    return traits, full_data


def select_features(ori_data: pd.DataFrame, y: np.ndarray) -> pd.DataFrame:
    """
    Feature screening: retain features with p-value < threshold, then sort features
    Sort rule: ascending p-value first, then descending absolute correlation coefficient
    :param ori_data: original omics feature dataframe
    :param y: phenotypic label array
    :return: filtered and sorted feature dataframe
    """
    r, pval, valid_mask = pearsonr_by_columns_df(ori_data, y)
    cols = ori_data.columns.to_numpy()

    cols_valid = cols[valid_mask]
    r_valid = r[valid_mask]
    p_valid = pval[valid_mask]
    absr_valid = np.abs(r_valid)

    # Filter features with statistically significant correlation
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
            order = np.lexsort((-absr_keep, p_keep))
        else:
            order = np.argsort(-absr_keep)
        cols_sel = cols_keep[order]
    else:
        cols_sel = np.array([], dtype=object)
       
    return ori_data.loc[:, cols_sel]

def cotcant_data(data_name: str, data_list,y):
    """
    Perform feature screening on each single omics dataset and horizontally concatenate filtered feature matrices
    :param data_name: combined dataset name separated by "+"
    :param data_list: list of original omics feature DataFrames
    :param y: phenotypic label array for feature correlation calculation
    :return: merged multi-omics feature matrix after feature selection
    """
    traits = None
    data_names = data_name.split("+")
    full_data = []
    for i,name in enumerate(data_names):
        data = select_features(data_list[i], y)
        full_data.append(data)
    return pd.concat(full_data, axis=1)

def split_and_scale_data(feat_data: np.ndarray, label: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    """
    Split dataset into training set (80%) and test set (20%), perform standardization
    :param feat_data: raw feature matrix
    :param label: phenotypic label array
    :return: standardized train features, test features, train labels, test labels, scaler instance
    """
    X_tr, X_te, y_tr, y_te = train_test_split(
        feat_data, label, test_size=0.2, random_state=42
    )
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_te = scaler.transform(X_te)
    return X_tr, X_te, y_tr, y_te, scaler


# ===================== Traditional Machine Learning Pipeline =====================
def run_ml_pipeline(args, trait_name: str, X_tr, X_te, y_tr, y_te):
    """
    Train and evaluate traditional ML models: Random Forest, SVR, Ridge Regression
    Save trained model via joblib and calculate prediction performance
    :param args: command line arguments
    :param trait_name: target phenotypic trait
    :param X_tr: training feature matrix
    :param X_te: test feature matrix
    :param y_tr: training labels
    :param y_te: test labels
    :return: Pearson correlation coefficient on test set
    """
    ckpt_file = f"{args.model}/{args.data}_{trait_name}_{args.model}.joblib"
    ckpt_full_path = Path(f'{ROOT_PATH}/checkpoints') / ckpt_file
    # Load existing checkpoint and directly evaluate model performance
    if ckpt_full_path.is_file():
        model = joblib.load(ckpt_full_path)
        estimator = model['model']
        y_pred = estimator.predict(X_te)
        r2 = r2_score(y_te, y_pred)
        pcc, _ = stats.pearsonr(y_te, y_pred)
        print(f"[{trait_name}] Load pre-trained checkpoint | PCC = {pcc:.4f} | R2 = {r2:.4f}\n")
        return pcc
    
    ml_config = {
        "rf": (RandomForestRegressor(n_estimators=100, max_depth=20, random_state=SEED), "RandomForest", False),
        "svm": (SVR(), "SVM", False),
        "rrblup": (Ridge(alpha=0.1), "Ridge Regression", False)
    }
    estimator, model_name, do_center = ml_config[args.model]

    y_tr_mean = None
    # Optional label centralization
    if do_center:
        y_tr_mean = y_tr.mean()
        estimator.fit(X_tr, y_tr - y_tr_mean)
        y_pred = estimator.predict(X_te) + y_tr_mean
    else:
        estimator.fit(X_tr, y_tr)
        y_pred = estimator.predict(X_te)

    # Save trained machine learning model
    save_dir = os.path.join(f"{ROOT_PATH}/checkpoints", args.model)
    os.makedirs(save_dir, exist_ok=True)
    save_dict = {
        "model": estimator,
        "y_tr_mean": y_tr_mean,
        "do_center": do_center,
        "model_name": model_name
    }
    save_path = os.path.join(save_dir, f"{args.data}_{trait_name}_{args.model}.joblib")
    joblib.dump(save_dict, save_path)

    # Model evaluation metrics
    r2 = r2_score(y_te, y_pred)
    pcc, _ = stats.pearsonr(y_te, y_pred)
    print(f"[{trait_name}] Training finished | Best PCC = {pcc:.4f} | Best R2 = {r2:.4f}\n")
    return pcc


# ===================== General Deep Learning Functions =====================
def build_dataloader(X_tr, X_te, y_tr, y_te) -> tuple[torchData.DataLoader, torchData.DataLoader]:
    """
    Construct PyTorch DataLoader for training and test dataset
    :param X_tr: standardized training features
    :param X_te: standardized test features
    :param y_tr: training phenotypic labels
    :param y_te: test phenotypic labels
    :return: train dataloader, test dataloader
    """
    tr_dataset = OmicsDataSet(X_tr, y_tr)
    te_dataset = OmicsDataSet(X_te, y_te)
    tr_loader = torchData.DataLoader(tr_dataset, batch_size=32, shuffle=True)
    te_loader = torchData.DataLoader(te_dataset, batch_size=32, shuffle=False)
    return tr_loader, te_loader


def dl_model_inference(model, te_loader, criterion) -> tuple[float, float, list, list]:
    """
    Perform inference on test set for deep learning models without gradient computation
    :param model: trained deep learning model
    :param te_loader: test set dataloader
    :param criterion: loss function
    :return: average test loss, Pearson correlation, ground truth list, prediction list
    """
    all_loss = 0.0
    y_true_test = []
    y_pred_test = []
    model.eval()
    with torch.no_grad():
        for data_X, trait_y in te_loader:
            data_X = data_X.to(DEVICE)
            trait_y = trait_y.to(DEVICE)
            outputs = model(data_X)
            all_loss += criterion(outputs.flatten(), trait_y).item()
            y_pred_test.extend(outputs.flatten().cpu().detach().numpy())
            y_true_test.extend(trait_y.cpu().detach().numpy())
    test_loss = all_loss / max(1, len(te_loader))
    pcc, _ = stats.pearsonr(y_true_test, y_pred_test)
    return test_loss, pcc, y_true_test, y_pred_test


def init_dl_model(args, feat_dim: int, k_params: dict = None, num_layers: int = None):
    """
    Initialize specified deep learning model and move model to designated GPU
    :param args: command line arguments
    :param feat_dim: input feature dimension
    :param k_params: convolution dimension parameters for SoyDNGP
    :param num_layers: number of network blocks for SoyDNGP
    :return: initialized model instance
    """

    if args.model == "dnngp":
        return DNNGPModel(feat_dim).to(DEVICE)
    elif args.model == "pnngs":
        return PNNGSModel(feat_dim).to(DEVICE)
    elif args.model == "soydngp":
        return SoyDNGPModel(
            k_params["k1"], k_params["k2"], k_params["k3"],
            num_blocks=num_layers
        ).to(DEVICE)
    return None


def run_dl_pipeline(args, trait_name: str, X_tr, X_te, y_tr, y_te, k_params=None, num_layers=None):
    """
    End-to-end training and evaluation pipeline for deep learning models
    Load pre-trained checkpoint if exists, otherwise train from scratch with early stopping
    :param args: command line arguments
    :param trait_name: target phenotypic trait
    :param X_tr: training feature matrix
    :param X_te: test feature matrix
    :param y_tr: training labels
    :param y_te: test labels
    :param k_params: convolution hyperparameters for SoyDNGP
    :param num_layers: number of network layers for SoyDNGP
    :return: optimal PCC achieved on test set
    """
    tr_loader, te_loader = build_dataloader(X_tr, X_te, y_tr, y_te)
    model = init_dl_model(args, X_tr.shape[1], k_params, num_layers)
    if model is None:
        return 0.0

    criterion = torch.nn.SmoothL1Loss()
    optimizer = torch.optim.AdamW(
        model.parameters(), weight_decay=0.0001, lr=0.0005, eps=1e-8
    )

    # Define checkpoint saving path
    ckpt_file = f"{args.model}/{args.data}_{trait_name}_{args.model}.pt"
    ckpt_full_path = Path(ROOT_PATH) / 'checkpoints' / ckpt_file

    # Load existing checkpoint and directly evaluate model performance
    if ckpt_full_path.is_file():
        ckpt = torch.load(ckpt_full_path, map_location="cpu", weights_only=True)
        model.load_state_dict(ckpt["state_dict"])
        _, pcc, y_true, y_pred = dl_model_inference(model, te_loader, criterion)
        r2 = r2_score(y_true, y_pred)
        print(f"[{trait_name}] Load pre-trained checkpoint | PCC = {pcc:.4f} | R2 = {r2:.4f}\n")
        return pcc

    # Train model from scratch with early stopping strategy
    best_pcc = -1.0
    best_true = None
    best_pred = None

    for epoch in range(1, EPOCHS + 1):
        # Training phase
        model.train()
        train_loss = 0.0
        for data_X, trait_y in tr_loader:
            data_X = data_X.to(DEVICE)
            trait_y = trait_y.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(data_X)
            loss = criterion(outputs.flatten(), trait_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= max(1, len(tr_loader))

        # Test set evaluation phase
        model.eval()
        test_loss, curr_pcc, y_true, y_pred = dl_model_inference(model, te_loader, criterion)

        # Save model with the highest PCC
        if curr_pcc > best_pcc:
            best_pcc = float(curr_pcc)
            best_true = np.array(y_true, dtype=np.float32)
            best_pred = np.array(y_pred, dtype=np.float32)
            early_stop_count = 0

            save_dir = os.path.join(f'{ROOT_PATH}/checkpoints', args.model)
            os.makedirs(save_dir, exist_ok=True)
            save_name = f"{args.data}_{trait_name}_{args.model}.pt"
            save_path = os.path.join(save_dir, save_name)
            torch.save({
                "trait": trait_name,
                "state_dict": model.state_dict()
            }, save_path)

        # Print training log every 50 epochs
        if epoch % 50 == 0:
            print(f"[{trait_name}] Epoch {epoch:04d} | TrainLoss: {train_loss:.8f} | TestLoss: {test_loss:.8f}")

    # Final performance summary after training
    best_r2 = r2_score(best_true, best_pred)
    print(f"[{trait_name}] Training finished | Best PCC = {best_pcc:.4f} | Best R2 = {best_r2:.4f}\n")
    return best_pcc

# ===================== Single Trait Processing Entry =====================
def process_omics_trait(args, trait_name: str, data_list: list, y_label: np.ndarray) -> float:
    """
    Full pipeline for a single phenotypic trait: feature selection -> feature merging -> train/test split -> standardization -> model training & evaluation
    :param args: command line input arguments
    :param trait_name: target phenotypic trait name
    :param data_list: list of original omics feature DataFrames
    :param y_label: one-dimensional phenotypic value array
    :return: test set PCC; return -999 if current trait is skipped
    """
    # Execute feature screening on each omics dataset and merge features
    data_sel = cotcant_data(args.data, data_list,y_label)
    print(f"Processed [{trait_name}] {args.data} dimension: {data_sel.shape} ")

    k_params = {"k1": 0, "k2": 0, "k3": 0}
    num_layers = 0

    # Reshape feature dimension and calculate convolutional hyperparameters for SoyDNGP
    if args.model == 'soydngp':
        feat_num = data_sel.shape[1]
        k_params["k1"] = int(math.sqrt(feat_num))
        rest = feat_num - k_params["k1"] ** 2
        # Truncate redundant features to make total dimension a perfect square
        if rest > 0:
            data_sel = data_sel.iloc[:, :-rest]
        num_layers = steps_to_two(k_params["k1"])
        k_params["k2"], k_params["k3"] = compute_k(k_params["k1"], num_layers)
        print(f"[{trait_name}] Feature dimension after cropping: {data_sel.shape}, layer number: {num_layers}, {k_params}")

        # Skip current trait if calculated feature map size is invalid
        if k_params["k2"] == 0:
            print(f"[{trait_name}] Skip this trait since k2 equals 0\n")
            return -999.0

    # Dataset splitting and standardization
    X_tr, X_te, y_tr, y_te, _ = split_and_scale_data(data_sel.values, y_label)

    # Run corresponding machine learning or deep learning pipeline
    if args.model in ["svm", "rrblup", "rf"]:
        return run_ml_pipeline(args, trait_name, X_tr, X_te, y_tr, y_te)
    else:
        return run_dl_pipeline(args, trait_name, X_tr, X_te, y_tr, y_te, k_params, num_layers)


# ===================== Main Program Entry =====================
def main():
    """Parse command line arguments, load merged multi-omics data and train model for each phenotypic trait"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--trait", "-t", type=str)
    parser.add_argument("--model", type=str, default="rrblup",
                        choices=['soydngp', 'dnngp', 'pnngs', 'svm', 'rrblup', 'rf'])
    parser.add_argument("--data", type=str, default="G+TE+SE",
                        choices=['ori_G', 'G', 'GE', 'TE', 'SE',
                                 'G+CE', 'G+GE', 'G+IE', 'G+SE', 'G+TE', 'G+TE+CE',
                                 'G+TE+IE', 'G+TE+SE', 'G+TE+TRE', 'G+TRE'])
    args = parser.parse_args()

    # Load concatenated multi-omics feature data and phenotypic table
    traits_df, data_list = load_all_data(args.data)
    print(f"Original raw feature dimension: { pd.concat(data_list, axis=1).shape}")
    trait_list = traits_df.columns.tolist() if args.trait == None else [args.trait]
    all_pccs = []

    # Train prediction model for the specified single trait
    for trait in trait_list:
        y_label = traits_df[trait].values.astype(np.float32)
        pcc_val = process_omics_trait(args, trait, data_list, y_label)
        if pcc_val != -999.0:
            all_pccs.append(pcc_val)


if __name__ == "__main__":
    main()
