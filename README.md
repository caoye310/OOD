# OOD-DTI

This repository is the sourse code for my undergraduate thesis:

Causal Substructure Learning for Generalizable Drug-Target Interaction Prediction

### Environment
To run the code successfully, the following dependencies need to be installed:
```
pytorch             1.13.0
scikit-learn        0.24.2      
numpy               1.23.1
pandas              2.0.3
dgl-cu110           0.6.1
rdkit               2022.03.4
torch_scatter       2.1.1
torch-geometric     2.5.0
torch-sparse        0.6.17
torch-spline-conv   1.2.2
```

### Data
DrugOOD: download link(https://drive.google.com/drive/folders/19EAVkhJg0AgMx7X-bXGOhD4ENLfxJMWC)

### Training & Evaluation
python main.py --dataset sbap_core_potency_assay --lam0 1 --lam1 0.001 --lam2 0.01 --lam3 0.01 --gamma1 0.9 --gamma2 0.8