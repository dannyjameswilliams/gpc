#include "RcppArmadillo.h"
#include "functions.h"

using namespace arma;
//' Rcpp Fit Gaussian Process Classification using MCMC
//'
//' Pseudo marginal approach to fitting a Gaussian process classification model using Markov Chain Monte Carlo (MCMC)
//'
//' @param y binary output vector in {-1, +1}
//' @param X predictor matrix
//' @param nsteps number of iterations to run the MCMC sampler for
//' @param nburn number of 'burn in'
//' @param nimp number of samples in importance sampling to approximate the marginal likelihood
//' @param init_theta initial conditions for hyperparameter vector
//' @param init_marginal_lik initial condition for marginal likelihood
//' @param init_f initial condition for latent variables f
//' @param prior_density R function returning the prior density at a single point
//' @param kernel R function taking two inputs to calculate the kernel matrix
//' @param prop_dist_sample R function giving a sample from the proposal distribution
//' @param prop_dist_density R function giving the density from the proposal distribution
//' @param print_every number of steps until progress is printed to the console
//' @param debug logical; if \code{TRUE} then debugging information is printed to the console at each step
//' @param chain_no chain that is currently running, only used for printing
//' @param kernel_pass string; contains information as to whether the \code{kernel} argument is a function or a string. If a string detailing the kernel to use, a parallel construction of the gram matrix is implemented
//'
//' @return a single chain consisting of samples of the hyperparameter vector and samples of the latent variable \eqn{f}
//'
//' @details
//' Combination of \code{\link{laplace_approx}}, \code{\link{get_approx_marginal}} and \code{\link{ell_ss_sample}}, coming
//' together to fit a Gaussian process classification model using MCMC.
//'
//' This function retrieves output from all of the above functions, calculates the acceptance probability and accepts or rejects a sample.
//' The accepted samples are saved in a matrix.
//'
// [[Rcpp::export(name="algo_1")]]
Rcpp::List algo_1(arma::vec& y, arma::mat& X, const int& nsteps, const int& nburn, const int& nimp,
                  const arma::vec& init_theta, const arma::vec& init_marginal_lik, const arma::vec& init_f,
                  const Rcpp::Function& prior_density,  const Rcpp::Function& kernel,
                  const Rcpp::Function& prop_dist_sample, const Rcpp::Function& prop_dist_density,
                  const int print_every, const bool debug, const int chain_no, Rcpp::String kernel_pass)
{

  int theta_n = init_theta.n_elem;
  int n = y.n_elem;
  int d = X.n_cols;

  double acceptance_prob;
  double acceptance_ratio = 0.0;

  arma::mat K(n, n);
  arma::mat theta_samples(nsteps, theta_n);
  arma::mat f_samples(nsteps, n);


  // Construct gram matrix (and check for kernel fn or string)
  const Rcpp::String kernel_is_function = "function";
  const Rcpp::String kernel_is_gaussian = "gaussian";

  if(kernel_pass == kernel_is_function){
    K = build_K(X, X, kernel, init_theta);
  } else if(kernel_pass == kernel_is_gaussian){
    K = make_gram_par(X, X, init_theta);
  }

  // Initial step before the loop

  Rcpp::List l_approx = laplace_approx(y, K);
  double p_tilde = get_approx_marginal_par(y, K, nimp, init_theta, l_approx);

  // for(int i=0; i<n; i++){
  //   p_tilde(i) = 1e-100;
  // }

  arma::vec theta = init_theta;
  arma::vec f = init_f;
  arma::vec a_probs(nsteps, fill::zeros);

  for(int i = 0; i < nsteps; i++)
  {

    // New proposal for each theta
    arma::vec theta_prop(theta_n);
    arma::vec prior_dens_theta_prop(theta_n);
    for(int t = 0; t < theta_n; t++)
    {
      theta_prop(t) = Rcpp::as<double>(prop_dist_sample(theta(t)));
      prior_dens_theta_prop(t) = Rcpp::as<double>(prior_density(theta_prop(t)));
    }

    // Instantly reject theta if it is not in the prior
    bool theta_prop_in_prior = all(prior_dens_theta_prop != 0);
    if(theta_prop_in_prior)
    {
      // Laplace Approximation and Pseudo Likelihood
      if(kernel_pass == kernel_is_function){
        K = build_K(X, X, kernel, theta_prop);
      } else if(kernel_pass == kernel_is_gaussian){
        K = make_gram_par(X, X, theta_prop);
      }

      l_approx = laplace_approx(y, K);
      double p_tilde_prop = get_approx_marginal_par(y, K, nimp, theta_prop, l_approx);

      bool is_p_tilde_prop_nan = Rcpp::all(Rcpp::is_nan(Rcpp::NumericVector::create(p_tilde_prop)));
      if(is_p_tilde_prop_nan) p_tilde_prop = p_tilde*2;

      // Acceptance Probability
      arma::vec prior_densities(theta_n);
      arma::vec prior_densities_prop(theta_n);
      arma::vec prop_densities_1(theta_n);
      arma::vec prop_densities_2(theta_n);
      for(int t = 0; t < theta_n; t++)
      {
        prior_densities(t) = Rcpp::as<double>(prior_density(theta(t)));
        prior_densities_prop(t) = Rcpp::as<double>(prior_density(theta_prop(t)));
        prop_densities_1(t) = Rcpp::as<double>(prop_dist_density(theta(t), theta_prop(t)));
        prop_densities_2(t) = Rcpp::as<double>(prop_dist_density(theta_prop(t), theta(t)));
      }

      double A = calculate_A(p_tilde_prop, prior_densities_prop, p_tilde,
                             prior_densities, prop_densities_1, prop_densities_2);
      a_probs(i) = A;

      // output diagnostics
      if(debug){
        Rcpp::Rcout << "theta proposal: " << theta_prop << "\n";
        Rcpp::Rcout << "p_tilde_prop: " << p_tilde_prop << "\n" <<
          "prior_densities_prop: " << sum(log((prior_densities_prop))) << "\n" <<
            "p_tilde: " << p_tilde << "\n" <<
              "prior_densities: " << sum(log((prior_densities))) << "\n" <<
                "prop_densities_1: " << sum(log((prop_densities_1))) << "\n" <<
                  "prop_densities_2: " << sum(log((prop_densities_2))) << "\n" <<
                    "A: " << A << "\n" <<
                      "theta proposal: " << theta_prop(0) << ", " << theta_prop(1) << "\n";
      }

      // find minimum by creating new arma::vector and taking min
      arma::vec acceptance_vec(2);
      acceptance_vec(0) = 0;
      acceptance_vec(1) = A;
      acceptance_prob = acceptance_vec.min();

      // Accept/Reject
      double u = log(R::runif(0, 1));
      if(debug){
        Rcpp::Rcout << "acceptance prob: " << acceptance_prob << "\n" <<
          "u: " << u;
      }
      if(u < acceptance_prob)
      {
        if(debug) {Rcpp::Rcout << " ... accepted! \n";}
        theta = theta_prop;
        p_tilde = p_tilde_prop;
        acceptance_ratio += 1;
      } else {
        if(debug) {Rcpp::Rcout << " ... rejected :( \n";}
      }
    } else {
      if(debug) {Rcpp::Rcout << "\n theta proposal missed the prior... rejected :( \n";}
    }

    // Save whichever theta won into theta_samples
    for(int t = 0; t < theta_n; t++)
    {
      theta_samples(i, t) = theta(t);
    }

    // Sample f with ELL SS
    arma::vec ellss = ell_ss_sample(y, f, K);
    for(int j = 0; j < n; j++)
    {
      f(j) = ellss(j);
      f_samples(i, j) = ellss(j);
    }
    // Save approximate marginal likelihood as well

    // Display progress to console (platform independent)
    if(i == 0 or ((i+1) % print_every) == 0){
      Rcpp::Rcout  <<  "Chain " << chain_no << " Running: " << i+1 << "/" << nsteps << " iterations completed. "
                   << "|| Acceptance ratio: " << acceptance_ratio / (i+1) << "\n";
    }
    R_CheckUserInterrupt();

  }
  Rcpp::Rcout << "\nChain completed successfully. \n";

  // Reduce to final (nsteps - nburn) samples
  f_samples = f_samples.rows(nburn, nsteps-1);
  theta_samples = theta_samples.rows(nburn, nsteps-1);
  acceptance_ratio = sum(a_probs.rows(nburn, nsteps-1)) / nsteps;

  return Rcpp::List::create(
    Rcpp::Named("f_samples") = f_samples,
    Rcpp::Named("theta_samples") = theta_samples,
    Rcpp::Named("acceptance_ratio") = acceptance_ratio,
    Rcpp::Named("probs") = a_probs
  );
}
