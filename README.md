# PDTreeDL
Phylodynamics + Tree Encoding + Deep Learning

------------
In phylodynamic framework, the evolutionary relationship of disease agents which can be obtained form genomic data and represented as trees provides insights on their spatio-temporal population dynamics. In this project, we aim to consolidate the usage of a tree encoding technique with statistical and deep learning frameworks to estimate epidemiological parameters from genomic and surveillance data. Future tasks also include superspreader detection and integration of multiple sources of information into phylodynamics frameworks.

The code provided in ``code/param_inf.py`` uses tree encodings simulated from birth death epidemiological model as inputs, and builds a convolutional variational autoencoder-based neural networks to infer three parameters of interest: *basic production number (R_nought)*, *infectious time*, and *transmission rate*. The tree encoding with branch length information shows promising predictive power for the parameters. 

``code/latent_space_vis_2d.ipynb`` and ``code/latent_space_vis_3d.ipynb`` contains demo visualizations of the 2D and 3D latent space representation of tree encodings learned from variational autoencoder, colored by the corresponding parameter values.

``code/beta_splitting_output_vis.ipynb`` contains performance plots of beta-splitting classification using DL. 

Output figures are [here](plots/).
