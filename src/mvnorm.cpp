#include "RcppArmadillo.h"
#include "functions.h"

using namespace arma;

//' @keywords internal
// [[Rcpp::export(name="rmvnorm_cpp")]]
arma::mat rmvnorm(int n, arma::vec& mu, arma::mat& sigma) {
  int ncols = sigma.n_cols;
  arma::mat Y = arma::randn(n, ncols);
  return arma::repmat(mu, 1, n).t() + Y * chol_plus_diag(sigma, "upper");
}
//' @keywords internal
// [[Rcpp::export(name="dmvnorm_cpp")]]
arma::vec dmvnorm(arma::mat const &x,
  arma::vec const &mean,
  arma::mat sigma,
  bool const logd) {
  using arma::uword;
  uword const n = x.n_rows,
    xdim = x.n_cols;
  arma::vec out(n);

  // cholesky decomp

  arma::mat const rooti = arma::inv(trimatu(chol_plus_diag(sigma, "upper")));
  double const rootisum = arma::sum(log(rooti.diag())),
    constants = -(double)xdim/2.0 * log2pi,
    other_terms = rootisum + constants;

  arma::rowvec z;
  for (uword i = 0; i < n; i++) {
    z = (x.row(i) - mean.t());
    inplace_tri_mat_mult(z, rooti);
    out(i) = other_terms - 0.5 * arma::dot(z, z);
  }

  if (logd)
    return out;
  return exp(out);
}
