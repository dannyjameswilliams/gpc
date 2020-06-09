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


test_that("fit gpc (with different f and not)", {

  fit = gpc(y, x, 50, 25, 2, 50, c(1,1), f, debug=TRUE)
  predict(fit, x)
  fit = gpc(y, x, 50, 25, 2, 50, c(1,1), "gaussian")
  plot(fit)
  print(fit)
  predict(fit, x)
})

test_that("S3 methods work", {
  fit = gpc(y, x, 50, 25, 2, 50, c(1,1), "gaussian")

  plot(fit)
  plot(fit, f=TRUE)
  print(fit)
  predict(fit, x)

  # also make a random data chain for theta > 4
  random_theta = matrix(rnorm(nrow(fit$chain1)*5), nrow(fit$chain1), 5)
  colnames(random_theta) = paste0("theta[",3:7,"]")
  fit$chain1 = cbind(fit$chain1[,1:2], random_theta, fit$chain1[,3:ncol(fit$chain1)])
  fit$chain2 = cbind(fit$chain2[,1:2], random_theta, fit$chain2[,3:ncol(fit$chain2)])
  fit$p = fit$p + 5
  plot(fit)
})

test_that("approx marginal non parallel", {

  la = gpc::laplace_approx(y, K)
  gpc::get_approx_marginal(y, K, 200, c(1,1), la)
})

test_that("dmvnorm",{
  gpc:::dmvnorm(K, rep(0, nrow(K)), diag(nrow(K)), TRUE)
})

test_that("bad chol", {
  A = matrix(1:25, 5, 5)
  gpc:::chol_plus_diag(A, "lower")
})

test_that("bad laplace approximation", {
  y = rep(1e5, length(y))
  laplace_approx(y, K)
})




