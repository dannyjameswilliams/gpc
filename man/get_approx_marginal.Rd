% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/RcppExports.R
\name{get_approx_marginal}
\alias{get_approx_marginal}
\title{Get Pseudo Marginal Likelihood}
\usage{
get_approx_marginal(y, K, nimp, theta, laplace_approx)
}
\arguments{
\item{y}{binary output vector in -1, +1}

\item{K}{Gram matrix}

\item{nimp}{number of samples in importance sampling to approximate the marginal likelihood}

\item{theta}{hyperparameter vector}

\item{laplace_approx}{list containing \code{f_hat} and \code{sigma_hat}, output from \code{\link{laplace_approx}}}
}
\value{
a single value, the log sum of the pseudo weights.
}
\description{
Unbiased estimation of \eqn{p(y | \theta)} using importance sampling.
}
\details{
Using the approximating distribution \eqn{q(f | y, \theta)}, the unbiased estimate of the marginal can be given as
\deqn{
\tilde{p} (y | \theta) \approx 1 / N_{imp} \sum^{N_{imp}}_{i=1} p(y | f_i) p(f_i | \theta) / q(f | y, \theta)
}
In this case, the approximating distribution \eqn{q(f | y, \theta)} is given by the laplace approximation,
which is calculated in \code{\link{laplace_approx}}, \eqn{p(y | f)} is the likelihood and \eqn{p(f | \theta)} is the prior density.
}
