# Description

A feasibility study of usefulness of fractal-aware datasets in identifying Alzheimer's disease.

Evaluation of ML models in identifying Alzheimer Classification based on fractal data derived from OASIS-1 dataset comprising of structural MRI scans.

### Dataset

Dataset used in this study comprises of a set of Hurst exponents extracted following the procedure in []() from the OASIS-1 data of structural MRI scans.

Because the dataset is unbalanced, several resampling method were used:
- oversampling with SMOTE
- random oversampling
- random undersampling
- combined over-under sampling with SMOTENN

### Models

Models used in the study:
- k-nearest neighbor
- support vector classifier
- gradient boost decision tree

which were trained within a nested cross-validation scheme.

### Results

|    | sampling method   | model   |   accuracy |   F1 score |   MCC |   MCC permutation |
|---:|:------------------|:--------|-----------:|----------:|-----------:|-----------:|
|  0 | over_smote        | lgbm    |      0.545 |     0.397 |      0.149 |      0.01  |
|  1 | none              | svc     |      0.579 |     0.37  |      0.14  |      0.03  |
|  2 | over_smote        | svc     |      0.541 |     0.369 |      0.106 |      0.03  |
|  3 | combine_smotenn   | knn     |      0.253 |     0.228 |      0.081 |      0.02  |
|  4 | over_random       | knn     |      0.464 |     0.369 |      0.099 |      0.069 |

The best classifier is the gradient boosted decision tree lgbm.