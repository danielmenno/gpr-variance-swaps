# Master Thesis
This code is part of my master thesis titled "Leveraging Variance Swaps for non-parametric Option Price Surface Modelling", undertaked at ETH Zürich in 2020/2021.

There are two main parts:

- Simulated environment: option prices are generated through SVI and SSVI parametrisations defined in vol_surfaces.py. In the notebook SVI.ipynb the GPR metholody is applied to learn the dependence of the pricing functional on the variance swap given different implied volaility surfaces

- Market data: in the notebook VIX.ipynb the Gaussian Process Regression methodology is tested on S&P500 call option prices using historical level of the VIX index

The definition of the GPR model and the learning procedure are defined through subclassing from Tensorflow Probability and can be found in gp_models.py
