
# gpc

<!-- badges: start -->
[![Build Status](https://travis-ci.com/dannyjameswilliams/gpc.svg?branch=master)](https://travis-ci.com/dannyjameswilliams/gpc)   [![codecov](https://codecov.io/gh/dannyjameswilliams/gpc/branch/master/graph/badge.svg)](https://codecov.io/gh/dannyjameswilliams/gpc)
<!-- badges: end -->

A [fast, sparse,](https://papers.nips.cc/paper/2240-fast-sparse-gaussian-process-methods-the-informative-vector-machine.pdf) parallel implementation of [Pseudo-Marginal Inference for
Gaussian Process Classification with Large Datasets](https://github.com/jakespiteri/GPclassification/blob/master/report/main.pdf), based on [Pseudo-Marginal Bayesian Inference for Gaussian Processes](https://www.researchgate.net/publication/262954130_Pseudo-Marginal_Bayesian_Inference_for_Gaussian_Processes).

### Authors:

Daniel Williams `daniel.williams@bristol.ac.uk`

Dom Owens `dom.owens@bristol.ac.uk`

Jake Spiteri `jake.spiteri@bristol.ac.uk`

## Installation

You can install the released version of `gpc` from [github](https://github.com/dannyjameswilliams/gpc) with:

``` r
library(devtools)
install_github("dannyjameswilliams/gpc", build_vignettes = TRUE)
```
Alternatively, the package can be installed faster without building the vignettes by changing the second argument to `FALSE`.

## Package Contents

The package contains software to efficiently fit a Gaussian process classification (gpc) mdoel, using `Rcpp` and `RcppParallel`. We also have included the e-mail spam dataset used for classification.

To see an example of the code, as well as a step-by-step tutorial for its implementation, see the vignette `using_gpc`, provided as an HTML document you can view in this repository by [clicking here](https://htmlpreview.github.io/?https://github.com/dannyjameswilliams/gpc/blob/master/using_gpc.html), or by running
```r
vignette(package="gpc")
```
once the package has installed, provided the argument `build_vignettes=TRUE` was specified.
