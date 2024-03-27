# Guided-PLI
**A Transferability-guided Protein-Ligand Interaction Prediction Method** \
A novel transferability-guided protein-ligand interaction prediction method that effectively fuses multiple modalities while leveraging transferability metrics to guide knowledge transfer during fine-tuning. This method addresses two key challenges in PLI prediction: 1) integrating heterogeneous protein and ligand data modalities, and 2) optimizing the transfer of beneficial pretraining knowledge while avoiding negative transfer.

## Directory Structure
```bash
├── AttentiveFP/           # GAT model for extracting drug features
├── data/                  # PLI task datasets
├── models/                # The Guided-PLI prediction model
├── args.yaml              # Directories and drug molecule parameters
├── config.py              # Configuration file for parameter settings
├── data_handler.py        # PLI data processing tool
├── main.py                # Main program
├── otfrm.py               # The transfer loss defination based on OTFRM
├── README.md              # Readme file
├── requirements.txt       # Environment dependencies
├── train_test.py          # Engine for training and testing the model
├── utils.py               # Collection of utilites
```

## 1. Environment building
[![python >3.9.17](https://img.shields.io/badge/python-3.9.17-brightgreen)](https://www.python.org/) [![torch-1.11.0](https://img.shields.io/badge/torch-1.11.0-orange)](https://github.com/pytorch/pytorch)

```bash
conda create -n GuidedPLI python==3.9.17
conda activate GuidedPLI
cd Guided-PLI
pip install -r requirements.txt
```

## 2. Data Preparation
Place the processed datasets for PDBbind, Kinase, and DUD-E in the `data/` directory. Below is a sample entry for each of the curated datasets:
* PDBbind
|PDB-ID|seq|rdkit_smiles|label|set|
| :----- | :-----: | :-----: | :-----: | -----: |
|11gs|PYTVV...GKQ|CC[C@@H](CSC[C@H]...C(=O)c1ccc(OCC(=O)O)c(Cl)c1Cl|5.82|train|
* Kinase
|PDB-ID|seq|rdkit_smiles|label|set|
| :----- | :-----: | :-----: | :-----: | -----: |
|Q14012|MLGA...HQL|N#Cc1ccc(NC(=O)Nc2ccnc3cc(C(F)(F)F)ccc23)nc1|0.0|train|
* DUD-E
|PDB-ID|seq|rdkit_smiles|label|set|
| :----- | :-----: | :-----: | :-----: | -----: |
|andr|FLNV...HTQ|O=c1cc(-c2ccccc2)[nH]c2cc(-c3ccc(I)cc3)nn12|0|train|

## 3. Pretrained Embedding Generation
To guide the finetuning with transferability from pretraining of MASSA, the embeddings of pretraining datasets from MASSA need to be generated first.
More details please refer to [MASSA repository](https://github.com/SIAT-code/MASSA).

## 4. Fine-tuning on PLI Tasks
Run `main.py` to perform fine-tuning from pre-trained PLMs to downstream PLI prediction. The following example demonstrates the command to fine-tune on the PDBBind task with guidance from transferability:
```bash
python main.py --task=PDBBind --SEED=42 --guide
```
For more input parameter settings, please refer to `config.py`.

## Acknowledgement
The SOFTWARE will be used for teaching or not-for-profit research purposes only. Permission is required for any commercial use of the Software.