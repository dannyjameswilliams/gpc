#' IVM Subset Selection
#'
#' Implementation of IVM subset selection for Gaussian process classification
#'
#' @param X predictor matrix
#' @param kernel kernel function for the covariance in \code{X}
#' @param theta hyperparameter vector
#' @param nsub desired subset size
#' @param noise Gaussian variance (noise) in observations
#'
#' @return a vector of indices of \code{y} giving the locations for the subset
#'
#' @details
#' This performs the 'Information Vector Machine' (IVM) algorithm to choose a
#' subset of data to reduce computation time when fitting a Gaussian Process Classification
#' model.
#'
#' @export
ivm_subset_selection = function(X, kernel, theta, nsub, noise = 0.001){

  X = as.matrix(X)
  n = nrow(X)
  active_set = c()
  inactive_set = 1:n

  A = rep(NA, n)
  for(i in 1:n) A[i] = kernel(X[i,], X[i,], theta)

  M = matrix(0, nsub, n)
  for(k in 1:nsub){

    delta = nu = rep(NA, length(inactive_set))
    for(jj in 1:length(inactive_set)){
      j = inactive_set[jj]
      nu[jj] = 1/(A[j] + noise)
      delta[jj] = -0.5 * log(1-A[j]*nu[jj])
    }
    ii = which.max(delta)
    i = inactive_set[ii]

    k_row = rep(NA, n)
    for(jj in 1:n) k_row[jj] = kernel(X[i,], X[jj,], theta)

    sj = k_row - t(M) %*% M[,i]
    M[k,] = sqrt(nu[ii]) * sj
    A = A - nu[ii]* diag(sj %*% t(sj))

    active_set = append(active_set, i)
    inactive_set = inactive_set[inactive_set!=i]
  }
  return(active_set)
}
