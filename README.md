# MirGP: A multi-omics-based plant phenotype prediction model incorporating small RNAs
## Project Introduction
This project realizes the prediction of complex crop agronomic traits based on three types of omics data, including genomics, transcriptomics and small RNA omics. It supports two categories of regression prediction models: traditional machine learning and deep learning.
## Built-in Models
- Deep Learning Models: DNNGP, PNNGS, SoyDNGP, MirGP
- Traditional Machine Learning Models: Random Forest (RF), Support Vector Regression (SVR), Ridge Regression RR-BLUP
## Overall Workflow
Multi-omics data splicing → Pearson correlation feature screening → dataset splitting and standardization → model training / weight loading and inference → prediction accuracy evaluation (PCC, R²) → automatic saving of optimal model weights.

---
### 1. Environment Dependencies
#### 1.1 Python Version
```text
Python >= 3.10
``` 
#### 1.2 Install Dependencies
```text
conda create -n mirgpEnv python=3.10
conda activate mirgpEnv
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
``` 
#### 1.3 Hardware Requirements
- Recommended: NVIDIA GPU with CUDA acceleration for fast model training
- Minimum: CPU (available but with slow training speed)

---
### 2. Project File Structure
```text
sRNA-MirGP
├── config/
|     └── defaults.yaml     # Hyperparameter configuration file for MirGP
├── datasets/               # Raw multi-omics datasets (named with unified abbreviations)
├── checkpoints/            # Saved model weight directory
│    ├── [model_name]/             
│    └── mirgp/              # Weights for self-developed MirGP model
│         ├── omics2/        # Weights for two-omics joint modeling
│         └── omics3/        # Weights for three-omics joint modeling
├── model/                   # Definition scripts for all deep learning models
│   ├── DNNGP.py
│   ├── PNNGS.py
│   ├── SoyDNGP.py
│   ├── MirGP2.py           # MirGP model structure for two omics
│   └── MirGP3.py           # MirGP model structure for three omics
├── main_other_model.py     # Entry script for machine learning & other DL models
├── main_MirGP_omics2.py    # Running script for MirGP two-omics prediction
├── main_MirGP_omics3.py    # Running script for MirGP three-omics prediction
├── requirements.txt        # List of third-party dependent packages
└── README.md               # Project deployment and usage documentation
``` 
---

### 3. Supported Omics Data Types
Omics Type
Parameter Abbreviation
Full Name Description
Genomic SNP
G, ori_G
Original genotype SNP features
Transcriptome mRNA
GE, TE

mRNA gene expression features
Small RNA Omics
CE, IE, SE, TRE
CE: cluster_expressionIE: isomiR_expressionTRE: tRFs_expression & miRNA_expression
Multi-omics Splicing Rule
Connect different omics abbreviations with + without spaces. Example: G+TE+TRE for three-omics joint modeling.

---
### 4. Running Commands
Command Line Parameter Explanation
- --model: Select the prediction model
- --data: Select single or combined multi-omics data
- --trait / -t: Specify the target agronomic trait for prediction
#### 4.1 Deep Learning Model Running Examples
##### MirGP Two-omics joint prediction
```text
python main_MirGP_omics2.py --config /config/G+SE.yaml --trait Plantheight --seq SE --seq_data SE
```
##### MirGP Three-omics joint prediction
```text
python main_MirGP_omics3.py --config /config/G+SE+TE.yaml -t Plantheight --rna_seq TE --srna_seq SE
```
##### SoyDNGP Three-omics prediction
```text
python main_other_model.py -t Plantheight --model soydngp --data G+TE+TRE
```
##### DNNGP Genomics + Transcriptomics
```text
python main_other_model.py --t Plantheight --model dnngp --data G+TE
```
##### PNNGS Single transcriptome prediction
```text
python main_other_model.py  -t Plantheight --model pnngs --data TE
```
#### 4.2 Traditional Machine Learning Running Examples
##### Random Forest (RF)
```text
python main_other_model.py -t Plantheight --model rf --data G+TE
```
##### Support Vector Regression (SVR)
```text
python main_other_model.py -t Plantheight --model svm --data G+TE
```
##### Ridge Regression RR-BLUP
```text
python main_other_model.py -t Plantheight --model rrblup --data G
```


---
### 5. Core Code Pipeline
1. Data Loading and Splicing: Automatically read single or multi-omics features and complete sample alignment.
2. Pearson Correlation Feature Screening: Retain features with P-value < 0.05 and remove noisy invalid features.
3. Data Preprocessing: Split datasets into training set and test set with ratio 8:2; adopt standardization to eliminate dimensional differences.
4. Model Training
  - Deep Learning: AdamW optimizer, SmoothL1 loss function, save the optimal model based on test PCC value.
  - Machine Learning: Train models directly and save complete model files.
5. Automatic Resume Training: The program will load existing weights directly for testing without retraining if historical weight files are detected.

---
### 6. Output Results Description
##### 6.1 Model Saving Path
- Deep learning weights: checkpoints/[model_name]/*.pt
- Machine learning weights: checkpoints/[model_name]/*.joblib
##### 6.2 Evaluation Metrics
- PCC (Pearson Correlation Coefficient): Core prediction metric; higher value means better prediction performance.
- R² (Coefficient of Determination): Reflect the goodness of fit between predicted values and actual phenotypic values.
##### 6.3 Console Log Output
The console will print detailed logs for each target trait, including:
- Feature dimension after correlation screening
- Training loss every 50 epochs
- Optimal PCC result on the test set

---
### 7. Adjustable Core Hyperparameters
Modify the following global parameters directly in source codes as needed:

```text
SEED = 123        # Fixed random seed for experimental reproducibility
EPOCHS = 1000     # Maximum training epochs
PV_THRESH = 0.05  # P-value threshold for Pearson feature screening
```

---
### 8. Common Errors and Solutions
1. GPU Out-of-Memory Error
Solution: Reduce batch size manually or switch to CPU mode for training.
2. SoyDNGP report k2=0 and skip current trait
Reason: Too few valid features after correlation screening.
Solution: Appropriately relax the P-value threshold for feature screening.
3. Program loads old weights directly without retraining
Solution: Delete corresponding .pt weight files under checkpoints folder, then restart training.
### 9. Datasets
If you encounter failures when downloading via Git LFS, an alternative download link for the dataset is provided separately in this project:
https://doi.org/10.6084/m9.figshare.32830973
