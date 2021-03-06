% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/fit_model.R
\name{prior_default}
\alias{prior_default}
\alias{proposal_default}
\alias{default_distributions}
\title{Default proposal and prior distributions}
\usage{
prior_default(shape = 5, rate = 5)

proposal_default(sd = 1)
}
\arguments{
\item{shape}{shape parameter of the gamma distribution (prior)}

\item{rate}{rate of the gamma distribution (prior)}

\item{sd}{standard deviation of the Gaussian distribution (proposal)}
}
\description{
Pre-set distributions to sample and take density from the prior/proposal
}
\details{
These functions both appear as arguments to \code{\link{gpc}}. These can be called again as arguments
if the user wants to define different parameters of the distributions.

These can also be used to create different proposals, you need to copy the format
of these functions.

The default priors are a gamma distribution with \code{rate} and \code{shape} equal to 5.
The default proposal distribution is a Gaussian distribution with standard deviation 1.
}
