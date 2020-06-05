test_that("basic subset selection", {
  # simulate GP data
  library(mvtnorm)
  set.seed(123)

  n <- 50
  x <- sort(runif(n,-10,125))

  k <- kernel_gaussian <- function(x, y,theta){
    return(theta[1] * exp(-0.5 / theta[2]^2 * sum((x - y)^2)))
  }

  K = outer(x, x, Vectorize(function(x, y) k(x, y, c(1, 1))))

  f <- rmvnorm(1, rep(0, n), K)
  y <- rbinom(n, 1, pnorm(f))
  y[y == 0] <- -1

  f = function(x, y, theta) theta[1] * exp(-0.5 / theta[2]^2 * sum((x - y)^2))

  ivm_subset_selection(x, k, c(1,1), nsub = 25)
})
