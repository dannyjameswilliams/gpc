# Generated by using Rcpp::compileAttributes() -> do not edit by hand
# Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#' Rcpp Fit Gaussian Process Classification using MCMC
#'
#' Pseudo marginal approach to fitting a Gaussian process classification model using Markov Chain Monte Carlo (MCMC)
#'
#' @param y binary output vector in {-1, +1}
#' @param X predictor matrix
#' @param nsteps number of iterations to run the MCMC sampler for
#' @param nburn number of 'burn in'
#' @param nimp number of samples in importance sampling to approximate the marginal likelihood
#' @param init_theta initial conditions for hyperparameter vector
#' @param init_marginal_lik initial condition for marginal likelihood
#' @param init_f initial condition for latent variables f
#' @param prior_density R function returning the prior density at a single point
#' @param kernel R function taking two inputs to calculate the kernel matrix
#' @param prop_dist_sample R function giving a sample from the proposal distribution
#' @param prop_dist_density R function giving the density from the proposal distribution
#' @param print_every number of steps until progress is printed to the console
#' @param debug logical; if \code{TRUE} then debugging information is printed to the console at each step
#' @param chain_no chain that is currently running, only used for printing
#' @param kernel_pass string; contains information as to whether the \code{kernel} argument is a function or a string. If a string detailing the kernel to use, a parallel construction of the gram matrix is implemented
#'
#' @return a single chain consisting of samples of the hyperparameter vector and samples of the latent variable \eqn{f}
#'
#' @details
#' Combination of \code{\link{laplace_approx}}, \code{\link{get_approx_marginal}} and \code{\link{ell_ss_sample}}, coming
#' together to fit a Gaussian process classification model using MCMC.
#'
#' This function retrieves output from all of the above functions, calculates the acceptance probability and accepts or rejects a sample.
#' The accepted samples are saved in a matrix.
#'
algo_1 <- function(y, X, nsteps, nburn, nimp, init_theta, init_marginal_lik, init_f, prior_density, kernel, prop_dist_sample, prop_dist_density, print_every, debug, chain_no, kernel_pass) {
    .Call(`_gpc_algo_1`, y, X, nsteps, nburn, nimp, init_theta, init_marginal_lik, init_f, prior_density, kernel, prop_dist_sample, prop_dist_density, print_every, debug, chain_no, kernel_pass)
}

#' Get Pseudo Marginal Likelihood
#'
#' Unbiased estimation of \eqn{p(y | \theta)} using importance sampling.
#'
#' @param y binary output vector in -1, +1
#' @param K Gram matrix
#' @param nimp number of samples in importance sampling to approximate the marginal likelihood
#' @param theta hyperparameter vector
#' @param laplace_approx list containing \code{f_hat} and \code{sigma_hat}, output from \code{\link{laplace_approx}}
#'
#' @return a single value, the log sum of the pseudo weights.
#'
#' @details
#' Using the approximating distribution \eqn{q(f | y, \theta)}, the unbiased estimate of the marginal can be given as
#' \deqn{
#' \tilde{p} (y | \theta) \approx 1 / N_{imp} \sum^{N_{imp}}_{i=1} p(y | f_i) p(f_i | \theta) / q(f | y, \theta)
#' }
#' In this case, the approximating distribution \eqn{q(f | y, \theta)} is given by the laplace approximation,
#' which is calculated in \code{\link{laplace_approx}}, \eqn{p(y | f)} is the likelihood and \eqn{p(f | \theta)} is the prior density.
#'
get_approx_marginal <- function(y, K, nimp, theta, laplace_approx) {
    .Call(`_gpc_get_approx_marginal`, y, K, nimp, theta, laplace_approx)
}

#' Get Pseudo Marginal Likelihood (Parallel)
#'
#' Unbiased estimation of \eqn{p(y | \theta)} using importance sampling, implemented in parallel.
#'
#' @param y binary output vector in -1, +1
#' @param K Gram matrix
#' @param nimp number of samples in importance sampling to approximate the marginal likelihood
#' @param theta hyperparameter vector
#' @param laplace_approx list containing \code{f_hat} and \code{sigma_hat}, output from \code{\link{laplace_approx}}
#'
#' @return a single value, the log sum of the pseudo weights.
#'
#' @details
#' This performs the same operations as \code{\link{get_approx_marginal}}, but implemented in parallel with \code{RcppParallel}.
#'
#' Using the approximating distribution \eqn{q(f | y, \theta)}, the unbiased estimate of the marginal can be given as
#' \deqn{
#' \tilde{p} (y | \theta) \approx 1 / N_{imp} \sum^{N_{imp}}_{i=1} p(y | f_i) p(f_i | \theta) / q(f | y, \theta)
#' }
#' In this case, the approximating distribution \eqn{q(f | y, \theta)} is given by the laplace approximation,
#' which is calculated in \code{\link{laplace_approx}}, \eqn{p(y | f)} is the likelihood and \eqn{p(f | \theta)} is the prior density.
#'
get_approx_marginal_par <- function(y, K, nimp, theta, laplace_approx) {
    .Call(`_gpc_get_approx_marginal_par`, y, K, nimp, theta, laplace_approx)
}

#' Elliptical Slice Sampling
#'
#' Elliptical method to sample latent variables \code{f}
#'
#' @param y binary output vector in -1, +1
#' @param f latent variable for Gaussian process
#' @param K Gram matrix
#'
#' @return new proposal for \eqn{f}
#'
#' @details
#' Sampling of latent variables \code{f} constrained to an ellipse centred at the current mean
#' \eqn{f}, with eccentricity of the ellipse determined by the covariance matrix \eqn{\Sigma}.
#'
ell_ss_sample <- function(y, f, K) {
    .Call(`_gpc_ell_ss_sample`, y, f, K)
}

#' Make Gram Matrix
#'
#' Compute the Gram (covariance) matrix
#'
#' @param x input matrix 1
#' @param y input matrix 2
#' @param k R function to apply covariance function for elements of \code{x} and \code{y}
#' @param theta hyperparameter vector to input to \code{k}
#'
#' @details
#' The kernel function \code{k} needs to have arguments corresponding to  \code{x}, \code{y}, \code{theta} in that order.
#'
#' \code{build_K} computes the gram matrix for a given \code{k}. For a faster, parallel approach,
#' \code{make_gram_par} uses a pre-determined Gaussian covariance function.
#'
build_K <- function(x, y, k, theta) {
    .Call(`_gpc_build_K`, x, y, k, theta)
}

#' Make Gram Matrix (Gaussian covariance)
#'
#' Compute the Gram (covariance) matrix in parallel with a Gaussian covariance function
#'
#' @param x input matrix 1
#' @param y input matrix 2
#' @param theta hyperparameter vector
#'
#' @details
#' This is a faster, parallel approach to build the Gram matrix using a pre-determined Gaussian covariance function.
#' \code{build_K} computes the gram matrix for a given \code{k}, but is slower and not in parallel.
make_gram_par <- function(x, y, theta) {
    .Call(`_gpc_make_gram_par`, x, y, theta)
}

#' Laplace Approximation
#'
#' Laplace approximation of the posterior of the latent Variables \eqn{f}.
#'
#' @param y binary output vector in -1, +1
#' @param K Gram matrix
#'
#' @return list containing two elements:
#' \item{\code{f_hat}}{mean of the approximate Gaussian distribution}
#' \item{\code{sigma_hat}}{covariance matrix of the approximate Gaussian distribution}
#'
#' @details
#' Obtain a Laplace approximate of the posterior of \deqn{
#' p(f | y, \theta)
#' }
#' by approximating
#' the distribution with a Gaussian \eqn{N(f | \mu_q, \Sigma_q)}.
#'
#' This amounts to an iterative procedure, iterating a Newton-Raphson formula:
#' \deqn{
#' f_{new} = f - (\nabla_f \nabla_f \Psi(f))^{-1} \nabla_f \Psi(f),
#' }
#' where \deqn{
#' \Psi(f) = log p(y | f) + log p (f | \theta) + const.
#' }
#' Beginning from \eqn{f=0}, this
#' is iterated until convergence. That is, when the distance between \eqn{f} and \eqn{f_{new}} becomes
#' sufficiently small.
laplace_approx <- function(y, K) {
    .Call(`_gpc_laplace_approx`, y, K)
}

#' @keywords internal
NULL

#' @keywords internal
dmvnorm <- function(x, mean, sigma, logd) {
    .Call(`_gpc_dmvnorm`, x, mean, sigma, logd)
}

#' Rcpp Predict Gaussian Process Classification using MCMC
#'
#' Pseudo marginal approach to fitting a Gaussian process classification model using Markov Chain Monte Carlo (MCMC)
#'
#' @param y binary output vector in {-1, +1}
#' @param X predictor matrix
#' @param newdata new data to predict from, same dimension as \code{X}
#' @param kernel R function taking two inputs to calculate the kernel matrix
#' @param fit list containing samples from the mcmc chains for \code{f} and \code{theta}
#' @param nchains number of chains used in fitting model
#' @param kernel_pass string containing information on what the kernel type is (decides parallel operations)
#' @param print_every number of steps until progress is printed to the console
#'
#' @return fitted probabilities corresponding to the positive and negative class for each data point in \code{newdata}
#'
predict_gp <- function(y, X, newdata, kernel, fit, nchains, kernel_pass, print_every) {
    .Call(`_gpc_predict_gp`, y, X, newdata, kernel, fit, nchains, kernel_pass, print_every)
}

#' @keywords internal
NULL

#' @keywords internal
NULL

#' @keywords internal
NULL

#' @keywords internal
chol_plus_diag <- function(A, type) {
    .Call(`_gpc_chol_plus_diag`, A, type)
}

