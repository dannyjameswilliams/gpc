#include "RcppArmadillo.h"
#include "functions.h"

using namespace arma;

//' Laplace Approximation
//'
//' Laplace approximation of the posterior of the latent Variables \eqn{f}.
//'
//' @param y binary output vector in -1, +1
//' @param K Gram matrix
//'
//' @return list containing two elements:
//' \item{\code{f_hat}}{mean of the approximate Gaussian distribution}
//' \item{\code{sigma_hat}}{covariance matrix of the approximate Gaussian distribution}
//'
//' @details
//' Obtain a Laplace approximate of the posterior of \deqn{
//' p(f | y, \theta)
//' }
//' by approximating
//' the distribution with a Gaussian \eqn{N(f | \mu_q, \Sigma_q)}.
//'
//' This amounts to an iterative procedure, iterating a Newton-Raphson formula:
//' \deqn{
//' f_{new} = f - (\nabla_f \nabla_f \Psi(f))^{-1} \nabla_f \Psi(f),
//' }
//' where \deqn{
//' \Psi(f) = log p(y | f) + log p (f | \theta) + const.
//' }
//' Beginning from \eqn{f=0}, this
//' is iterated until convergence. That is, when the distance between \eqn{f} and \eqn{f_{new}} becomes
//' sufficiently small.
// [[Rcpp::export(name="laplace_approx")]]
Rcpp::List laplace_approx(const arma::vec& y, const arma::mat& K)
{
  int n = y.n_elem;
  arma::mat W(n, n, fill::zeros);
  arma::mat Wsqrt(n, n, fill::zeros);
  arma::mat L(n, n, fill::zeros);
  arma::mat I(n, n, fill::eye);
  arma::mat Sigma(n, n, fill::zeros);
  arma::mat B(n, n, fill::zeros);

  arma::vec f(n); f.fill(99);
  arma::vec fnew(n); fnew.fill(0);
  //Rcpp::Rcout << "Laplace Approxiarma::mation running... \n";

  int ii = 0;
  double tol = 1e-6;
  while(mean(pow(f - fnew, 2)) > tol)
  {
    f = fnew;
    W.diag() = - d2_log_lik(y, f);
    Wsqrt = sqrt(W);
    B = I + Wsqrt * K * Wsqrt;

    L = chol_plus_diag(B, "lower");
    arma::mat Lt = L.t();
    arma::vec b =  W * f + d_log_lik(y, f);
    arma::vec inv1 = solve(trimatl(L), Wsqrt * K * b);
    arma::vec a = b - Wsqrt * solve(trimatu(Lt), inv1);
    fnew = K * a;

    // Should converge in only a few iterations
    if(ii > 9){
      break;
    }
    ii += 1;
  }

  Sigma = K - K * Wsqrt * inv_sympd(B) * Wsqrt * K;

  return Rcpp::List::create(
    Rcpp::Named("f_hat") = fnew,
    Rcpp::Named("sigma_hat") = Sigma
  );
}
