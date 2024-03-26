# IPSUR_sequential_design

Author : Paul Lartaud (1s2lartaudpaul@gmail.com)
Requirements:
- Numpy version 
- PyMC
- 
-
-

##General description
This repository provides an example of the IP-SUR sequential design strategy for Bayesian inverse problems introduced in this paper.

A utilitaries directory and a scripts directory are provided. The following files are found:
- `Model.py`: contains the classes used for Gaussian Process (GP) surrogate model. This is based on sklearn base code.
- `Kernel.py`: contains the various classes for kernels used in GP surrogates. This file contains Matérn, RBF and White kernels as well as the Linear Coregionalization Model from Bonilla (CITE) which is used to build multi-output GP. 
- `Dataset.py`: contains the DataSet Class used for GP surrogate models. It includes standardization of the data, and power transforms (Yeo-Jonhson and Box-Cox).
- `Sequential_design.py`: contains the various functions and classes used to implement the sequential design strategies described in the paper (IP-SUR, CSQ, Naive and Bayes risk minimization).
- `IPSUR_banana.py`: contains the script for the banana target posterior distribution, which serves as the first test case in the paper.
- `IPSUR_bimodal.py`: contains the script for the bimodal target posterior distribution, which is the second test case.

An additional repository Resources contains the initial conditions used in the numerical applications of the paper. It contains the starting GP surrogates, the direct model observations and their covariance, and the well-trained GP. 
This code is only a demonstrator and will be improved in the future. Though this is not yet the case here, it can be easily adapted to specialized package for GP surrogate models to improve the computational efficiency and extend the possible methods. 
If you encounter any problem regarding the code or the paper, please feel free to reach me. 

##Citation
Please cite the paper if you use this code or the methods discussed in it.
