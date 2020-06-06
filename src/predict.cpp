#include "RcppArmadillo.h"
#include "functions.h"

using namespace arma;

//' Rcpp Predict Gaussian Process Classification using MCMC
//'
//' Pseudo marginal approach to fitting a Gaussian process classification model using Markov Chain Monte Carlo (MCMC)
//'
//' @param y binary output vector in {-1, +1}
//' @param X predictor matrix
//' @param newdata new data to predict from, same dimension as \code{X}
//' @param kernel R function taking two inputs to calculate the kernel matrix
//' @param fit list containing samples from the mcmc chains for \code{f} and \code{theta}
//' @param nchains number of chains used in fitting model
//' @param kernel_pass string containing information on what the kernel type is (decides parallel operations)
//' @param print_every number of steps until progress is printed to the console
//'
//' @return fitted probabilities corresponding to the positive and negative class for each data point in \code{newdata}
//'
// [[Rcpp::export(name="predict_gp")]]
arma::vec predict_gp(const arma::vec& y, arma::mat& X, arma::mat& newdata,
                     const Rcpp::Function kernel, const Rcpp::List fit,
                     const int nchains, const Rcpp::String kernel_pass,
                     const int print_every){


  arma::mat f_samples = fit["f_samples"];
  arma::mat theta_samples = fit["theta_samples"];

  int nsamples = f_samples.n_rows;
  int n_new = newdata.n_rows;
  int n = y.n_elem;

  arma::mat K(n, n);
  arma::mat k_star(n, n_new);
  arma::mat k_star_star(n_new, n_new);

  // Display properties
  Rcpp::Rcout << "Predicting from GP classification model. \n" <<
    "Averaging across all " << nsamples << " steps, " << nchains << " chain(s). \n" <<
      "Fit Data length n = " << n << ", dimension p = " << X.n_cols << ".\n" <<
        "New data length n = " << n_new << ", dimension p = " << newdata.n_cols << ".\n" <<
          "_____________________________________ \n";

  arma::vec pred(n_new, fill::zeros);
  arma::vec theta = theta_samples.row(0).t();

  // Construct gram matrix (and check for kernel fn or string)
  const Rcpp::String kernel_is_function = "function";
  const Rcpp::String kernel_is_gaussian = "gaussian";

  if(kernel_pass == kernel_is_function){
    K = build_K(X, X, kernel, theta);
    k_star = build_K(X, newdata, kernel, theta);
    k_star_star = build_K(newdata, newdata, kernel, theta);
  } else if(kernel_pass == kernel_is_gaussian){
    K = make_gram_par(X, X, theta);
    k_star = make_gram_par(X, newdata, theta);
    k_star_star = make_gram_par(newdata, newdata, theta);
  }

  for(int i = 1; i < nsamples; i++)
  {

    arma::mat sigma_q(n, n, fill::zeros);
    arma::vec theta = theta_samples.row(i).t();
    if(any(theta != theta_samples.row(i-1).t()))
    {
      if(kernel_pass == kernel_is_function){
        K = build_K(X, X, kernel, theta);
        k_star = build_K(X, newdata, kernel, theta);
        k_star_star = build_K(newdata, newdata, kernel, theta);
      } else if(kernel_pass == kernel_is_gaussian){
        K = make_gram_par(X, X, theta);
        k_star = make_gram_par(X, newdata, theta);
        k_star_star = make_gram_par(newdata, newdata, theta);
      }
    }

    arma::vec mu_q = f_samples.row(i).t();
    sigma_q.diag() = -(d2_log_lik(y, mu_q));

    arma::mat K_inv = inv(K);
    arma::vec m_star = k_star.t() * K_inv * mu_q;
    arma::mat s2_star = k_star_star - k_star.t() * K_inv * k_star + k_star.t() *
      K_inv * sigma_q * K_inv * k_star;

    arma::vec xn = m_star/sqrt(1 + s2_star.diag());
    for(int j = 0; j < xn.n_elem; j++){
      pred(j) = pred[j] + R::pnorm(xn[j], 0, 1, true, false);
    }

    // Display progress to console (platform independent)
    if(i == 0 or ((i+1) % print_every) == 0){
      Rcpp::Rcout  <<  "Prediction Averaging: " << i+1 << "/" << nsamples << " steps completed. \n";
    }
    R_CheckUserInterrupt();
  }
  Rcpp::Rcout << "\n";
  Rcpp::Rcout << "Completed. \n";
  pred = pred / nsamples;

  return(pred);
}
