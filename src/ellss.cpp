#include "RcppArmadillo.h"
#include "functions.h"

using namespace arma;

//' Elliptical Slice Sampling
//'
//' Elliptical method to sample latent variables \code{f}
//'
//' @param y binary output vector in -1, +1
//' @param f latent variable for Gaussian process
//' @param K Gram matrix
//'
//' @return new proposal for \eqn{f}
//'
//' @details
//' Sampling of latent variables \code{f} constrained to an ellipse centred at the current mean
//' \eqn{f}, with eccentricity of the ellipse determined by the covariance matrix \eqn{\Sigma}.
//'
// [[Rcpp::export(name="ell_ss_sample")]]
arma::vec ell_ss_sample(const arma::vec& y, const arma::vec& f, arma::mat& K){
  int n = K.n_rows;
  arma::vec zrs(n, fill::zeros);
  double pi = 3.141592653589793116;

  arma::mat z = rmvnorm(1, zrs, K);

  double u = R::rexp(1);
  double eta = sum(log_lik(y, f)) - u;

  double alpha = R::runif(0, 2*pi);
  double alpha_min = alpha - 2*pi;
  double alpha_max = alpha;

  arma::vec f_prop = f * cos(alpha) + z.t() * sin(alpha);

    while(sum(log_lik(y, f_prop)) < eta)
  {

    if(alpha < 0){
      alpha_min = 0;
    }
    if(alpha >= 0){
      alpha_max = 0;
    }
    alpha = R::runif(alpha_min, alpha_max);
    f_prop = f * cos(alpha) + z.t() * sin(alpha);

    R_CheckUserInterrupt();
  }

  return f_prop;
}
