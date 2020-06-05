#include "RcppArmadillo.h"
#include "functions.h"

using namespace arma;
//' @keywords internal
// [[Rcpp::export(name="log_lik")]]
arma::vec log_lik(const arma::vec& y, const arma::vec& f)
{
  int n = y.n_elem;
  arma::vec out(n);
  for(int ii = 0; ii < n; ii++)
  {
    out[ii] = R::pnorm(y[ii] * f[ii], 0, 1, true, true);
  }
  return out;
}
//' @keywords internal
// [[Rcpp::export(name="d_log_lik")]]
arma::vec d_log_lik(const arma::vec& y, const arma::vec& f)
{
  int n = y.n_elem;
  arma::vec out(n);
  for(int ii = 0; ii < n; ii++)
  {
    //out[ii] = y[ii] * R::dnorm(f[ii], 0, 1, false) / R::pnorm(y[ii] * f[ii], 0, 1, true, false);
    out[ii] = y[ii] * exp(R::dnorm(f[ii], 0, 1, true) - R::pnorm(y[ii] * f[ii], 0, 1, true, true));
  }
  return out;
}
//' @keywords internal
// [[Rcpp::export(name="d2_log_lik")]]
arma::vec d2_log_lik(const arma::vec& y, const arma::vec& f)
{
  int n = y.n_elem;
  arma::vec out(n);
  for(int ii = 0; ii < n; ii++)
  {
    //out[ii] = -pow(R::dnorm(f[ii], 0, 1, false), 2)/pow(R::pnorm(y[ii]*f[ii], 0, 1, true, false),2) -
    //            y[ii] * f[ii] * R::dnorm(f[ii], 0, 1, false) / R::pnorm(y[ii] * f[ii], 0, 1, true, false);
    out[ii] = -exp(2 * R::dnorm(f[ii], 0, 1, true) - 2 * R::pnorm(y[ii]*f[ii], 0, 1, true, true) ) -
      y[ii] * f[ii] * exp( R::dnorm(f[ii], 0, 1, true) - R::pnorm(y[ii] * f[ii], 0, 1, true, true) );
  }
  return out;
}

