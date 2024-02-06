### Description

Our approach focuses on preparing a model for the early detection of Alzheimer's disease. The training dataset comprises of features extracted from structural Magnetic Resonance Imaging (MRI) scans. To extract these features, we utilize [mfmri](https://github.com/Mark-Kac-Center/mfmri), which combines Space-Filling Curves to reduce dimensionality and Hurst exponents as a tool to summarize the fractal properties inherent in the data. 


### Dataset

![OASIS-1 Dataset](dataplot.png)

As the basis we take the [OASIS-1](https://www.oasis-brains.org/) dataset consisting of ~400 scans with healthy subjects and subject affected with the Mild Cognitive Impairment (an early stage of Alzheimer's disease). The output of the *mfmri* pipeline for each three-dimensional scan is a Hurst exponent profile along an axis. The figure presents Hurst profiles aggregated for each patient cohort, providing a first reason for using profiles as inputs to machine-learning models.     


### Training the model

To train the models, we apply several resampling method to address the imbalanced character of the dataset:
- oversampling with SMOTE
- random oversampling
- random undersampling
- combined over-under sampling with SMOTENN

We investigate models of varying complexity:
- Logistic Regression (*sklearn.linear_model.LogisticRegressor*)
- K-Nearest Neighbor (*sklearn.neighbors.KNeighborsClassifier*)
- Support Vector Machine (*sklearn.svm.SVC*)
- Multi-Layer Perceptron (*sklearn.neural_network.MLPClassifier*)
- Random Forest (*sklearn.ensemble.RandomForestClassifier*)
- Gradient Boost Decision Tree (*lgbm.LGBMClassifier*)

Each model is trained within a [nested cross-validation](https://scikit-learn.org/stable/modules/cross_validation.html) scheme. 

Training is conducted in notebook [find_good_models.ipynb](find_good_models.ipynb). We provide training results in the [results.csv](results.csv) file.


### Results

We present the top-five best performing model and resampling technique from the [results.csv](results.csv) file. We measure performance in terms of the Matthews Correlation Coefficient as a measure suitable for imbalanced problems.

| **data_resample**| **model**   |   **test_acc** |   **test_f1** |   **test_mcc** |
|:-----------------|:--------|----------------:|----------------:|-----------:|
| *over_smote*     | *lgbm*  |        *0.56272*|       *0.483093*|  *0.261954*|
| over_smote       | rf      |        0.532747 |        0.456586 |   0.246613 |
| over_smote       | svc     |        0.523497 |        0.434067 |   0.236747 |
| over_random      | mlp     |        0.549491 |        0.4797   |   0.232935 |
| over_random      | rf      |        0.541073 |        0.448499 |   0.228025 |


### Packages used in the study
* scikit-learn
* lgbm
* [imbalanced-learn](https://imbalanced-learn.org/stable/)
