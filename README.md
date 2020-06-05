
# gpc

<!-- badges: start -->
<!-- badges: end -->

A [fast, sparse,](https://papers.nips.cc/paper/2240-fast-sparse-gaussian-process-methods-the-informative-vector-machine.pdf) parallel implementation of [Pseudo-Marginal Inference for
Gaussian Process Classification with Large Datasets](https://github.com/jakespiteri/GPclassification/blob/master/report/main.pdf), based on [Pseudo-Marginal Bayesian Inference for Gaussian Processes](https://www.researchgate.net/publication/262954130_Pseudo-Marginal_Bayesian_Inference_for_Gaussian_Processes).

### Authors:

Daniel Williams `daniel.williams@bristol.ac.uk`

Dom Owens `dom.owens@bristol.ac.uk`

Jake Spiteri `jake.spiteri@bristol.ac.uk`

## Installation

You can install the released version of gpc from [github](https://github.com/dannyjameswilliams/gpc) with:

``` r
library(devtools)
devtools::install_github(https://github.com/dannyjameswilliams/gpc)
```

## Example

This is a basic example which demonstrates using the core `gpc` function on bundled data

``` r
library(gpc)
load("data/spamAnalysis.Rdata") #load spam data analysis

fit <- gpc(y = y_train,X= as.matrix(X_train), 
  nsteps= 100,nburn= 0,nchains= 1,nimp= 100,init_theta= c(1,1),kernel= f,
  print_every = 10, debug = FALSE) #fit model
  
print(fit) #print method
plot(fit) #plot method
```

