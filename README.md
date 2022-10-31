# Multi-fidelity Surrogate Model for Performance Prediction of Truss Structures Based on Transfer Learning
This project is the open source code of the paper "**Multi-fidelity Surrogate Model for Performance Prediction of Truss Structures Based on Transfer Learning**".
The code is based on the multi-fidelity finite element analysis data of the truss structure as the training dataset for fusion model training.
The code integrates Bayesian hyperparameter optimization, pre-trained network construction based on low-fidelity data, and retraining based on high-fidelity data.
vscode file is used to configure the Matlab plugin, in the Data folder, 10, 25, 72 and 200 rod truss multi-fidelity datasets have been provided for training, and Models are used to save the trained ensemble model structure.

How to build a fusion model:
1) Execute 'mainMFDNNL' to build a pre-trained network under a low-fidelity dataset;
2) perform 'mainMFDNNH' for retraining network construction;
3) The optimal model will be saved in the Model.
