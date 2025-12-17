# Interpretable Multivariate Conformal Prediction with Fast Transductive Standardization

This repository contains all codes and data needed to reproduce code in paper:

>  "Interpretable Multivariate Conformal Prediction with Fast Transductive Standardization"  
>  Yunjie Fan, Matteo Sesia 
>  [arXiv preprint]

## Abstract

We propose a conformal prediction method for constructing tight simultaneous prediction intervals for multiple, potentially related, numerical outputs given a single input. This method can be combined with any multi-target regression model and guarantees finite-sample coverage. It is computationally efficient and yields informative prediction intervals even with limited data. The core idea is a novel \emph{coordinate-wise} standardization procedure that makes residuals across output dimensions directly comparable, estimating suitable scaling parameters using the calibration data themselves. This does not require modeling of cross-output dependence nor auxiliary sample splitting. Implementing this idea requires overcoming technical challenges associated with transductive or full conformal prediction. Experiments on simulated and real data demonstrate this method can produce tighter prediction intervals than existing baselines while maintaining valid simultaneous coverage.


Detailed example to be published soon.