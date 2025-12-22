# Interpretable Multivariate Conformal Prediction with Fast Transductive Standardization

This repository contains all codes and data needed to reproduce code in paper:

>  "Interpretable Multivariate Conformal Prediction with Fast Transductive Standardization"  
>  Yunjie Fan, Matteo Sesia 
>  [arXiv preprint](https://arxiv.org/abs/2512.15383)

## Abstract

We propose a conformal prediction method for constructing tight simultaneous prediction intervals for multiple, potentially related, numerical outputs given a single input. This method can be combined with any multi-target regression model and guarantees finite-sample coverage. It is computationally efficient and yields informative prediction intervals even with limited data. The core idea is a novel \emph{coordinate-wise} standardization procedure that makes residuals across output dimensions directly comparable, estimating suitable scaling parameters using the calibration data themselves. This does not require modeling of cross-output dependence nor auxiliary sample splitting. Implementing this idea requires overcoming technical challenges associated with transductive or full conformal prediction. Experiments on simulated and real data demonstrate this method can produce tighter prediction intervals than existing baselines while maintaining valid simultaneous coverage.

## Required Packages

- `numpy` 
- `scipy` 
- `scikit-learn` 
- `pandas` 
- `ucimlrepo` 

## Reproducibility Instructions

The `utility` package contains the following scripts:
- `rectangle.py`: a script that defines the general use of hyper-rectangles;
- `res_rescaled.py`: a script that contains implementation of our **TSCP** described in the paper, including TSCP-GWC and TSCP-LWC; 
- `data_splitting.py`: a script that contains implementation of Naive, TSCP-S, Pop. Oracle, and Point CHR;
- `unscaled.py`: a script that contains implementation of Bonferroni and Unscaled Max;
- `copula.py`: a script that contains implementation of Emp. Copula;

- `data_generator.py`: a script that contains the data generator used in the experiments described in the paper;
- `exps.py`: a script that contains implementation (including how to generate or load the data) of all of our simulated and real data experiments described in the paper.

The `real_exps` folder and `syn_exps` folder contain all of our results and figure summaries of simulated and real experiments. 