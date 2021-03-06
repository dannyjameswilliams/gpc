% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/fit_model.R
\name{gpc}
\alias{gpc}
\title{Fit Gaussian Process Classification Model}
\usage{
gpc(
  y,
  X,
  nsteps,
  nburn,
  nchains,
  nimp,
  init_theta,
  kernel = "gaussian",
  prior = prior_default(),
  proposal = proposal_default(),
  init_f = rep(0, length(y)),
  init_marginal_lik = rep(0, length(y)),
  print_every = 25,
  debug = FALSE
)
}
\arguments{
\item{y}{binary output vector in {-1, +1}}

\item{X}{predictor matrix}

\item{nsteps}{number of iterations to run the MCMC sampler for}

\item{nburn}{number of 'burn in'}

\item{nchains}{number of chains to run, i.e. number of times to run the MCMC sampler}

\item{nimp}{number of samples in importance sampling to approximate the marginal likelihood}

\item{init_theta}{initial conditions for hyperparameter vector}

\item{kernel}{function taking two inputs to calculate the kernel matrix}

\item{prior}{list of two functions, to sample and give the density of the prior (see details)}

\item{proposal}{list of two functions, to sample and give the density of the proposal distribution (see details)}

\item{init_f}{initial condition}

\item{init_marginal_lik}{initial condition for marginal likelihood}

\item{print_every}{number of steps until progress is printed to the console}

\item{debug}{logical; if \code{TRUE} then debugging information is printed to the console at each step}
}
\value{
An object of type '\code{gpc}', containing elements of the model fit and samples given for all chains for
\describe{
\item{theta_samples}{accepted samples of the hyperparameter vector}
\item{f_samples}{samples of the latent variable}
\item{pm_samples}{samples of the log sum of pseudo marginal approximation evaluations}
}
}
\description{
Wrapper function for the Rcpp implementation of fitting a Gaussian process classification model
}
\details{
This is a wrapper function for \code{\link{algo_1}}, which can take default arguments and calls
the \code{\link{algo_1}} in a loop for the required number of chains.

The arguments \code{prior} and \code{proposal} must be lists, detailed below
\describe{
\item{\code{prior}}{must have elements \code{density} and \code{sample}, each only taking one argument and returning a single value}
\item{\code{grad}}{must have elements \code{density} and \code{sample}, containing one and two arguments respectively, returning a single value}
}
}
