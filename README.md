# Ensemble-SWAN
This repository contains the Python scripts used in the research project on Ensemble-SWAN.. These scripts encompass data preprocessing, descriptive statistics, benchmark models, and the implementation of the Ensemble-SWAN method. Additionally, uncertainty quantification techniques and plotting scripts are included.

Repository Structure
The repository consists of the following files:

Data Preprocessing.ipynb: This Jupyter notebook contains the necessary data preprocessing steps. E.g., obtaining sequences with a minimum length of 3 and a maximum length of 20.

Descriptive Statistics Criteo.ipynb: This Jupyter notebook includes code for generating descriptive statistics, as discussed in Section 3 of the research paper.

ARRN replication.py: This Python script replicates the ARNN (Attentional Recurrent Neural Network) proposed by Ren et al. (2018). The ARNN serves as a benchmark for comparing the performance of SWAN.

SWAN.ipynb: The SWAN Jupyter notebook contains the implementation code for the Stacked Web of Attentional Neurons. Additionally, this file includes the code for generating the oversampled dataset using ADASYN (Adaptive Synthetic Sampling).

Ensemble-SWAN.py: This Python script implements the Ensemble-SWAN method. It also includes code for generating the undersampled datasets and the ensemble uncertainty quantification approach.

UQ MCD Undersampled.ipynb: This Jupyter notebook demonstrates the uncertainty quantification technique, Monte Carlo Dropout (MCD), for one of the sub-NNs of the Ensemble-SWAN model trained on the undersampled dataset.

UQ MCD Oversampled.ipynb: This Jupyter notebook demonstrates the uncertainty quantification technique, Monte Carlo Dropout (MCD), for the SWAN model trained on the oversampled dataset.

Plots SWAN Ensemble-SWAN.ipynb: This Jupyter notebook contains code for generating all the figures presented in Section 5.3 of the research paper.

How to Use
To utilize these scripts, follow the steps below:

Clone the repository: Run git clone https://github.com/pimweterman/Ensemble-SWAN.git in your terminal or download the repository as a ZIP file.

Install the required dependencies: Make sure you have the necessary Python libraries installed. Install the required packages mentioned in each script.

Execute the scripts: Open the Jupyter notebooks (.ipynb) in Jupyter Notebook or JupyterLab, and run each cell to execute the code. For Python scripts (.py), run them using a Python interpreter.

Additional Notes
Some scripts may require a more powerful computer to run due to computational requirements. 

For more detailed explanations and context, please refer to the corresponding sections in the research paper.

References
If you use this code or find it helpful, please consider citing the research paper:

[Include citation information here]

For any questions or issues, please feel free to contact Pim Weterman at 475771pw@student.eur.nl
