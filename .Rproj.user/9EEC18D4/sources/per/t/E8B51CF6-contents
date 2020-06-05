test_that("gpc works", {

  # simulate GP data
  library(mvtnorm)
  set.seed(123)

  n <- 50
  x <- sort(runif(n,-10,125))

  k <- kernel_gaussian <- function(x, y, sigma, gamma){
    return(sigma * exp(-0.5 / gamma^2 * sum((x - y)^2)))
  }

  K = outer(x, x, Vectorize(function(x, y) k(x, y, 1, 1)))

  f <- rmvnorm(1, rep(0, n), K)
  y <- rbinom(n, 1, pnorm(f))
  y[y == 0] <- -1

  f = function(x, y, theta) theta[1] * exp(-0.5 / theta[2]^2 * sum((x - y)^2))

  fit = gpc(y, x, 50, 25, 2, 50, c(1,1), f)
  fit = gpc(y, x, 50, 25, 2, 50, c(1,1), "gaussian")
  plot(fit)
  print(fit)
  predict(fit, x)
})
