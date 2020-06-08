#include "RcppArmadillo.h"
#include "RcppParallel.h"
#include "functions.h"

// [[Rcpp::depends(RcppParallel)]]

//' Get Pseudo Marginal Likelihood
//'
//' Unbiased estimation of \eqn{p(y | \theta)} using importance sampling.
//'
//' @param y binary output vector in -1, +1
//' @param K Gram matrix
//' @param nimp number of samples in importance sampling to approximate the marginal likelihood
//' @param theta hyperparameter vector
//' @param laplace_approx list containing \code{f_hat} and \code{sigma_hat}, output from \code{\link{laplace_approx}}
//'
//' @return a single value, the log sum of the pseudo weights.
//'
//' @details
//' Using the approximating distribution \eqn{q(f | y, \theta)}, the unbiased estimate of the marginal can be given as
//' \deqn{
//' \tilde{p} (y | \theta) \approx 1 / N_{imp} \sum^{N_{imp}}_{i=1} p(y | f_i) p(f_i | \theta) / q(f | y, \theta)
//' }
//' In this case, the approximating distribution \eqn{q(f | y, \theta)} is given by the laplace approximation,
//' which is calculated in \code{\link{laplace_approx}}, \eqn{p(y | f)} is the likelihood and \eqn{p(f | \theta)} is the prior density.
//'
// [[Rcpp::export(name="get_approx_marginal")]]
double get_approx_marginal(const arma::vec y, const arma::mat& K, const int nimp, const arma::vec theta,
  const Rcpp::List laplace_approx)
{
  int ny = y.n_elem;
  const arma::vec zero_vec(ny, arma::fill::zeros);

  // Extract output from laplace_approx list
  arma::mat Lt = laplace_approx["sigma_hat"];
  arma::vec mu = laplace_approx["f_hat"];
  arma::mat f_pseudo = rmvnorm(nimp, mu, Lt);
  arma::mat laplace_densities = dmvnorm(f_pseudo, mu, Lt);
  arma::mat prior_densities = dmvnorm(f_pseudo, zero_vec, K);

  // construct arma::vector which we will log exp sum over
  arma::vec log_pseudo_weights(nimp);
  for(int ii = 0; ii < nimp; ii++)
  {
    arma::vec log_p_y_giv_theta(ny);

    // Use laplace approxiarma::mates for sample
    arma::mat f_temp = f_pseudo.row(ii);
    //arma::vec laplace_density = dmvnorm(f_temp, mu, Lt);
    //arma::vec prior_density = dmvnorm(f_temp, zero_vec, K);

    // Sum over log probabilities of each element in importance sampling estiarma::mator
    double log_p_y_giv_f = 0;
    for(int jj = 0; jj < ny; jj ++)
    {
      log_p_y_giv_f += R::pnorm(y(jj) * f_temp(jj), 0, 1, true, true);
    }
    double log_p_f_giv_theta = log(prior_densities(ii));
    double log_q_f_giv_y_theta = log(laplace_densities(ii));

    //Rcpp::Rcout << laplace_density;

    // Combine together based on formula, log_exp_sum comes later
    log_pseudo_weights(ii) = log_p_y_giv_f + log_p_f_giv_theta - log_q_f_giv_y_theta - log(nimp);
  }

  // Retrieve log prob by log exp sum
  double out = log_sum(log_pseudo_weights);
  return(out);
}

struct marginal_loop : public RcppParallel::Worker
{
  // Set input and output variables
  const RcppParallel::RVector<double> y;
  const RcppParallel::RMatrix<double> f_pseudo;
  const RcppParallel::RMatrix<double> laplace_densities;
  const RcppParallel::RMatrix<double> prior_densities;
  RcppParallel::RVector<double> log_pseudo_weights;

  // Create object
  marginal_loop(const Rcpp::NumericVector y_,
    const Rcpp::NumericMatrix f_pseudo_,
    const Rcpp::NumericMatrix laplace_densities_,
    const Rcpp::NumericMatrix prior_densities_,
    Rcpp::NumericVector log_pseudo_weights_)
    : y(y_),
      f_pseudo(f_pseudo_),
      laplace_densities(laplace_densities_),
      prior_densities(prior_densities_),
      log_pseudo_weights(log_pseudo_weights_) {}

  // Create operator for sections of loop
  void operator()(std::size_t begin, std::size_t end) {
    for(std::size_t ii = begin; ii < end; ii++)
    {

      // Use laplace approximates for sample
      RcppParallel::RMatrix<double>::Row f_temp = f_pseudo.row(ii);

      // Sum over log probabilities of each element in importance sampling estiarma::mator
      double log_p_y_giv_f = 0;
      for(std::size_t jj = 0; jj < y.length(); jj ++)
      {
        double yjj = y[jj];
        double f_temp_jj = f_temp[jj];
        log_p_y_giv_f += R::pnorm(yjj * f_temp_jj, 0, 1, true, true);
      }
      double log_p_f_giv_theta = log(prior_densities[ii]);
      double log_q_f_giv_y_theta = log(laplace_densities[ii]);


      // Combine together based on formula, log_exp_sum comes later
      log_pseudo_weights[ii] = log_p_y_giv_f + log_p_f_giv_theta - log_q_f_giv_y_theta;
    }

  }

};

//' Get Pseudo Marginal Likelihood (Parallel)
//'
//' Unbiased estimation of \eqn{p(y | \theta)} using importance sampling, implemented in parallel.
//'
//' @param y binary output vector in -1, +1
//' @param K Gram matrix
//' @param nimp number of samples in importance sampling to approximate the marginal likelihood
//' @param theta hyperparameter vector
//' @param laplace_approx list containing \code{f_hat} and \code{sigma_hat}, output from \code{\link{laplace_approx}}
//'
//' @return a single value, the log sum of the pseudo weights.
//'
//' @details
//' This performs the same operations as \code{\link{get_approx_marginal}}, but implemented in parallel with \code{RcppParallel}.
//'
//' Using the approximating distribution \eqn{q(f | y, \theta)}, the unbiased estimate of the marginal can be given as
//' \deqn{
//' \tilde{p} (y | \theta) \approx 1 / N_{imp} \sum^{N_{imp}}_{i=1} p(y | f_i) p(f_i | \theta) / q(f | y, \theta)
//' }
//' In this case, the approximating distribution \eqn{q(f | y, \theta)} is given by the laplace approximation,
//' which is calculated in \code{\link{laplace_approx}}, \eqn{p(y | f)} is the likelihood and \eqn{p(f | \theta)} is the prior density.
//'
// [[Rcpp::export(name="get_approx_marginal_par")]]
double get_approx_marginal_par(arma::vec& y, const arma::mat& K, const int nimp, const arma::vec& theta,
  const Rcpp::List laplace_approx)
{
  int ny = y.n_elem;
  const arma::vec zero_vec(ny, arma::fill::zeros);

  // Extract output from laplace_approx list
  arma::mat Lt = laplace_approx["sigma_hat"];
  arma::vec mu = laplace_approx["f_hat"];

  // Create samples and densities
  arma::mat f_pseudo = rmvnorm(nimp, mu, Lt);
  arma::mat laplace_densities = dmvnorm(f_pseudo, mu, Lt, false);
  arma::mat prior_densities = dmvnorm(f_pseudo, zero_vec, K, false);

  // Convert matrices to Rcpp for parallel worker
  Rcpp::NumericMatrix f_pseudo_nm = mat_to_rcpp(f_pseudo);
  Rcpp::NumericMatrix laplace_densities_nm = mat_to_rcpp(laplace_densities);
  Rcpp::NumericMatrix prior_densities_nm = mat_to_rcpp(prior_densities);
  Rcpp::NumericVector y_nv = vec_to_rcpp(y);

  // Call parallelFor to loop over nimp samples
  Rcpp::NumericVector log_pseudo_weights(nimp);
  marginal_loop obj(y_nv, f_pseudo_nm, laplace_densities_nm,
    prior_densities_nm, log_pseudo_weights);
  parallelFor(0, nimp, obj);

  // Convert output back to an arma vec
  arma::vec log_pseudo_weights_a = rcpp_to_vec(log_pseudo_weights);

  // Retrieve log prob by log exp sum
  double out = log_sum(log_pseudo_weights - log(nimp));

  return(out);
}

