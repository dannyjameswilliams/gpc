#include "RcppArmadillo.h"
#include "functions.h"

using namespace arma;

// [[Rcpp::depends(RcppArmadillo)]]
//' @keywords internal
arma::mat chol_plus_diag(arma::mat& A, Rcpp::String type)
{
  bool success = false;
  arma::mat L;
  while(success == false)
  {
    if(type == "upper") {
      success = chol(L, A, "upper");
    } else {
      success = chol(L, A, "lower");
    }

    if(success == false)
    {
      A.diag() += 1e-6;
    }
    R_CheckUserInterrupt();
  }
  return(L);
}
//' @keywords internal
double prod(vec a)
{
  int n = a.n_elem;
  double tmp = 1;
  for(int i = 0; i < n; i++)
  {
    tmp = tmp*a[i];
  }
  return(tmp);
}
//' @keywords internal
double calculate_A(const double& p_tilde_prop,const vec& prior_densities_prop,const double& p_tilde,
  const vec& prior_densities, const vec& prop_densities_1, const vec& prop_densities_2)
{
  double t1 = p_tilde_prop + sum(log((prior_densities_prop)));
  double t2 = p_tilde + sum(log((prior_densities)));
  double t3 = sum(log(prop_densities_1[1]));
  double t4 = sum(log(prop_densities_2[1]));

  double acceptance_prob = (t1 - t2) + (t3 - t4);
  return acceptance_prob;
}

//' @keywords internal
double log_sum(arma::vec a)
{
  double max_a = max(a);
  double out = max_a + log(sum(exp(a - max_a)));
  return(out);
}
//' @keywords internal
void inplace_tri_mat_mult(arma::rowvec &x, arma::mat const &trimat){
  arma::uword const n = trimat.n_cols;

  for(unsigned j = n; j-- > 0;){
    double tmp(0.);
    for(unsigned i = 0; i <= j; ++i)
      tmp += trimat.at(i, j) * x[i];
    x[j] = tmp;
  }
}
