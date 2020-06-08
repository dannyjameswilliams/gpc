#include "RcppArmadillo.h"
#include "functions.h"

using namespace arma;

Rcpp::NumericMatrix mat_to_rcpp(arma::mat& X)
{
  Rcpp::NumericMatrix A = Rcpp::as<Rcpp::NumericMatrix>(Rcpp::wrap(X));
  return(A);
}

arma::mat rcpp_to_mat(Rcpp::NumericMatrix X)
{
  arma::mat A = Rcpp::as<arma::mat>(X);
  return(A);
}
Rcpp::NumericVector vec_to_rcpp(arma::vec& x)
{
  Rcpp::NumericVector a = Rcpp::as<Rcpp::NumericVector>(Rcpp::wrap(x));
  return(a);
}
arma::vec rcpp_to_vec(Rcpp::NumericVector x)
{
  arma::vec A = Rcpp::as<arma::vec>(x);
  return(A);
}
