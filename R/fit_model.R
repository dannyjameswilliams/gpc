#' @importFrom stats dgamma rgamma
#' @keywords internal
prior_default = function(shape = 5, rate = 5){
  prior_dens = function(x) dgamma(x, shape, rate)
  prior_samp = function(x) rgamma(1, shape, rate)
  return(list(density = prior_dens, sample = prior_samp))
}
#' @importFrom stats dnorm rnorm
#' @keywords internal
proposal_default = function(sd = 1){
  prop_dist_samp = function(x) rnorm(1, x, sd)
  prop_dist_dens = function(x, theta) dnorm(x, theta, sd)
  return(list(density = prop_dist_dens, sample = prop_dist_samp))
}



#' Fit Gaussian Process Classification Model
#'
#' Wrapper function for the Rcpp implementation of fitting a Gaussian process classification model
#'
#' @param y binary output vector in {-1, +1}
#' @param X predictor matrix
#' @param nsteps number of iterations to run the MCMC sampler for
#' @param nburn number of 'burn in'
#' @param nchains number of chains to run, i.e. number of times to run the MCMC sampler
#' @param nimp number of samples in importance sampling to approximate the marginal likelihood
#' @param init_theta initial conditions for hyperparameter vector
#' @param kernel function taking two inputs to calculate the kernel matrix
#' @param prior list of two functions, to sample and give the density of the prior (see details)
#' @param proposal list of two functions, to sample and give the density of the proposal distribution (see details)
#' @param init_f initial condition
#' @param init_marginal_lik initial condition for marginal likelihood
#' @param print_every number of steps until progress is printed to the console
#' @param debug logical; if \code{TRUE} then debugging information is printed to the console at each step
#'
#' @details
#' This is a wrapper function for \code{\link{algo_1}}, which can take default arguments and calls
#' the \code{\link{algo_1}} in a loop for the required number of chains.
#'
#' The arguments \code{prior} and \code{proposal} must be lists, detailed below
#' \describe{
#' \item{\code{prior}}{must have elements \code{density} and \code{sample}, each only taking one argument and returning a single value}
#' \item{\code{grad}}{must have elements \code{density} and \code{sample}, containing one and two arguments respectively, returning a single value}
#' }
#'
#' @return
#' An object of type '\code{gpc}', containing elements of the model fit and samples given for all chains for
#' \describe{
#' \item{theta_samples}{accepted samples of the hyperparameter vector}
#' \item{f_samples}{samples of the latent variable}
#' \item{pm_samples}{samples of the log sum of pseudo marginal approximation evaluations}
#' }
#' @export
gpc = function(y, X, nsteps, nburn, nchains, nimp, init_theta, kernel = "gaussian",
               prior = prior_default(), proposal = proposal_default(),
               init_f = rep(0, length(y)), init_marginal_lik = rep(0, length(y)),
               print_every = 25, debug=FALSE){

  if(!is.matrix(X)) X = as.matrix(X)
  p = length(init_theta)
  n = length(y)
  d = ncol(X)

  # match kernel function
  if(typeof(kernel) == "character"){
    if(grep(kernel, "gaussian", ignore.case=TRUE) == 1){
      kernel = function(x) x    # just to match to C inputs
      kernel_pass = "gaussian"
    }
  } else if (typeof(kernel) == "closure"){
    kernel_pass = "function"
  }

  cat("Fitting a GP classification model. \n")
  cat("Running with", nchains, "chain,", nsteps, "steps. \n")
  cat("Data length n = ", n, ", dimension p = ", d, ".\n", sep="")
  cat("Number of hyperparameters: ", p, "\n")
  cat("_____________________________________ \n")

  samples_mat = matrix(NA, nrow = nsteps-nburn, ncol = p + n + 1)
  colnames(samples_mat) = c(paste0("theta[",1:p,"]"), "pseudo",
                            paste0("f[", 1:n,"]"))
  samples_out = rep(list(samples_mat), nchains)

  for(chain in 1:nchains){
    cat("Chain", chain, "/", nchains, "starting... \n")
    run = algo_1(y, X, nsteps, nburn, nimp, init_theta, init_marginal_lik, init_f,
                 prior$density,  kernel, proposal$sample, proposal$density,
                 print_every, debug, chain_no = chain, kernel_pass = kernel_pass)
    samples_out[[chain]][,1:p] = run$theta_samples
    samples_out[[chain]][,p+1] = run$pm_samples
    samples_out[[chain]][,(p+2):(p+n+1)] = run$f_samples
  }

  names(samples_out) = paste0("chain", 1:nchains)
  samples_out$n = n
  samples_out$p = p
  samples_out$d = d
  samples_out$nchains = nchains
  samples_out$y = y
  samples_out$X = X
  samples_out$kernel = kernel
  samples_out$prior = prior
  samples_out$proposal = proposal
  samples_out$print_every = print_every
  samples_out$kernel_pass = kernel_pass

  class(samples_out) = "gpc"

  return(samples_out)
}


#' Plot Gaussian Process Classification Model
#'
#' @param x object of type '\code{gpc}', output from \code{\link{gpc}}
#' @param f logical; if \code{TRUE}, plot posteriors of latent variable \code{f}
#' @param ... further arguments to be passed to plot
#'
#' @return
#' trace and density plot of the model fit
#'
#' @importFrom graphics par plot lines
#' @importFrom stats density
#' @importFrom grDevices hcl
#'
#' @rawNamespace S3method(plot, gpc)
#'
#' @export
plot.gpc = function(x, f = FALSE, ...){
  theta_var = paste0("theta[",1:x$p,"]")
  pm_var = "pseudo"
  f_var = paste0("f[", 1:x$n, "]")
  chain_var = paste0("chain",1:x$nchains)
  nsteps = nrow(x$chain1)





  chain_col_f <- function(n) {
    hues = seq(15, 375, length = n + 1)
    hcl(h = hues, l = 65, c = 100)[1:n]
  }
  chain_cols = chain_col_f(x$nchains)
  if(f){

    par(mfrow=c(min(x$n + 1, 4), 2))

    all_f = x[[chain_var[1]]][,f_var]
    if(x$nchains > 1){
      for(i in 2:x$nchains) all_f = rbind(all_f, x[[chain_var[i]]][,f_var])
    }


    if(x$n > 4) par(ask=TRUE)
    for(t in 1:x$n){
      yr = range(all_f[,t])
      for(i in 1:x$nchains){

        if(i==1) plot(1:nsteps, x[[chain_var[i]]][,f_var[t]], type="l", col = chain_cols[i],
                      xlab = "Iterations", ylab = f_var[t], main = paste(f_var[t], "trace"),
                      ylim = yr)
        if(i!=1) lines(1:nsteps, x[[chain_var[i]]][,f_var[t]], col = chain_cols[i])
      }
      theta_dens = x[[chain_var[1]]][,f_var[t]]
      for(i in 2:x$nchains) theta_dens = rbind(theta_dens, x[[chain_var[i]]][,f_var[t]])
      plot(density(theta_dens), xlab = "Density",
           ylab = f_var[t], main = paste(f_var[t], "density"))
    }
    if(x$n > 4) par(ask=FALSE)
  } else {

    par(mfrow=c(min(x$p + 1, 4), 2))

    # combine theta across all string
    all_theta = x[[chain_var[1]]][,theta_var]
    if(x$nchains > 1){
      for(i in 2:x$nchains) all_theta = rbind(all_theta, x[[chain_var[i]]][,theta_var])
    }

    # if more than 4 theta/pm, add ask = TRUE
    if(x$p + 1 > 4) par(ask=TRUE)

    # loop over theta values and plot trace + density for each one
    for(t in 1:x$p){
      yr = range(all_theta[,t])
      for(i in 1:x$nchains){

        if(i==1) plot(1:nsteps, x[[chain_var[i]]][,theta_var[t]], type="l", col = chain_cols[i],
             xlab = "Iterations", ylab = theta_var[t], main = paste(theta_var[t], "trace"),
             ylim = yr)
        if(i!=1) lines(1:nsteps, x[[chain_var[i]]][,theta_var[t]], col = chain_cols[i])
      }
      theta_dens = x[[chain_var[1]]][,theta_var[t]]
      for(i in 2:x$nchains) theta_dens = rbind(theta_dens, x[[chain_var[i]]][,theta_var[t]])
      plot(density(theta_dens), xlab = "Density",
           ylab = theta_var[t], main = paste(theta_var[t], "density"))
    }

    # do the same for pseudo marginal
    all_pm = x[[chain_var[1]]][,pm_var]
    if(x$nchains > 1){
      for(i in 2:x$nchains) all_pm = rbind(all_pm, x[[chain_var[i]]][,pm_var])
    }

    yr = range(all_pm)
    for(i in 1:x$nchains){
      if(i==1) plot(1:nsteps, x[[chain_var[i]]][,pm_var], type="l", col = chain_cols[i],
                    xlab = "Iterations", ylab = pm_var, main = paste("log", pm_var, "trace"),
                    ylim = yr)
      if(i!=1) lines(1:nsteps, x[[chain_var[i]]][,pm_var], col = chain_cols[i])
    }
    theta_dens = x[[chain_var[1]]][,pm_var]
    for(i in 2:x$nchains) theta_dens = rbind(theta_dens, x[[chain_var[i]]][,pm_var])
    plot(density(theta_dens), xlab = "Density",
         ylab = pm_var, main = paste("log", pm_var, "density"))

    if(x$p + 1 > 4) par(ask=FALSE)
  }

  par(mfrow=c(1,1))

}

#' Print Gaussian Process Classification Model
#'
#' @param x object of type '\code{gpc}', output from \code{\link{gpc}}
#' @param ... further arguments to be passed to print
#'
#' @importFrom stats median var quantile
#'
#' @rawNamespace S3method(print, gpc)
#'
#' @export
print.gpc = function(x, ...){
  nsteps = nrow(x$chain1)
  cat("GP classification model fit with pseudo marginal likelihood. \n")
  cat("Run with", x$nchains, "chains and", nsteps, "steps. \n")
  cat("Data length n = ", x$n, ", number of hyperparameters: ", x$p, "\n")
  cat("____________________________________________________________ \n")

  # combine chains
  chain_var = paste0("chain",1:x$nchains)
  all_vars = x[[chain_var[1]]]
  if(x$nchains > 1){
    for(i in 2:x$nchains) all_vars = rbind(all_vars, x[[chain_var[i]]])
  }
  means = colMeans(all_vars)
  medians = apply(all_vars, 2, median)
  vars = apply(all_vars, 2, var)
  pct5 = apply(all_vars, 2, quantile, probs = 0.05)
  pct95 = apply(all_vars, 2, quantile, probs = 0.95)

  statistics = rbind(means, medians, vars, pct5, pct95)

  colnames(statistics) = colnames(all_vars)
  rownames(statistics) = c("Mean", "Median", "Variance", "5% Quantile", "95% Quantile")

  print(statistics)
}

#' Predict from Gaussian Process Classification Model
#'
#' @param object object of type '\code{gpc}', output from \code{\link{gpc}}
#' @param newdata new data matrix, of same dimension as \code{X} used for fitting
#' @param ... further arguments to pass to \code{predict}
#'
#' @return
#' a vector of probabilities corresponding to the positive and negative class
#'
#' @rawNamespace S3method(predict, gpc)
#' @export
predict.gpc = function(object, newdata, ...){
  if(!is.matrix(newdata)) newdata = as.matrix(newdata)
  # combine all chains
  chain_var = paste0("chain",1:object$nchains)
  f_all = object$chain1[, paste0("f[", 1:object$n, "]")]
  theta_all = object$chain1[, paste0("theta[", 1:object$p, "]")]
  if(object$nchains > 1){
    for(i in 2:object$nchains) {
      f_all = rbind(f_all, object[[chain_var[i]]][, paste0("f[", 1:object$n, "]")])
      theta_all = rbind(theta_all, object[[chain_var[i]]][, paste0("theta[", 1:object$p, "]")])
    }
  }
  fit_all_chains = list(f_samples = f_all, theta_samples = theta_all)
  pred = predict_gp(object$y, object$X, newdata,
                    object$kernel, fit_all_chains,
                    object$nchains, object$kernel_pass,
                    object$print_every)
  return(pred)
}
