#ifndef functions_H
#define functions_H

Rcpp::List algo_1(const arma::vec& y, const arma::mat& X, const int& nsteps, const int& nburn, const int& nimp,
  const arma::vec& init_theta, const arma::vec& init_marginal_lik, const arma::vec& init_f,
  const Rcpp::Function& prior_density,  const Rcpp::Function& kernel,
  const Rcpp::Function& prop_dist_sample, const Rcpp::Function& prop_dist_density,
  const int print_every = 25, const bool debug = false);

double get_approx_marginal(const arma::vec y, const arma::mat& K, const int nimp, const arma::vec theta,
  const Rcpp::List laplace_approx);

arma::vec ell_ss_sample(const arma::vec& y, const arma::vec& f, arma::mat& K);

arma::mat build_K(const arma::mat& x, const arma::mat& y, Rcpp::Function k, const arma::vec& theta);

double log_sum(arma::vec a);

Rcpp::List laplace_approx(const arma::vec& y, const arma::mat& K);

arma::vec log_lik(const arma::vec& y, const arma::vec& f);

arma::vec d_log_lik(const arma::vec& y, const arma::vec& f);

arma::vec d2_log_lik(const arma::vec& y, const arma::vec& f);

arma::mat rmvnorm(int n, arma::vec& mu, arma::mat& sigma);

arma::vec dmvnorm(arma::mat const &x,
  arma::vec const &mean,
  arma::mat sigma,
  bool const logd = false);

arma::vec predict_gp(const arma::vec& y, const arma::mat& X, const arma::mat& newdata, const Rcpp::Function kernel,
  const Rcpp::List fit, const int print_every = 25);

arma::mat chol_plus_diag(arma::mat& A, Rcpp::String type="upper");

double prod(arma::vec a);


double calculate_A(const double& p_tilde_prop,const arma::vec& prior_densities_prop,const double& p_tilde,
  const arma::vec& prior_densities, const arma::vec& prop_densities_1, const arma::vec& prop_densities_2);

double log_sum(arma::vec a);

static double const log2pi = std::log(2.0 * M_PI);

void inplace_tri_mat_mult(arma::rowvec &x, arma::mat const &trimat);

Rcpp::NumericMatrix mat_to_rcpp(arma::mat& X);

arma::mat rcpp_to_mat(Rcpp::NumericMatrix X);

Rcpp::NumericVector vec_to_rcpp(arma::vec& x);

arma::vec rcpp_to_vec(Rcpp::NumericVector x);

template <typename InputIterator1>
inline double kernel_gaussian_eval(InputIterator1 begin1, InputIterator1 end1, double magnitude, double lengthscale);

struct kernel_eval;

arma::mat make_gram_par(arma::mat& x, arma::mat& y, const arma::vec& theta);

struct marginal_loop;

double get_approx_marginal_par(arma::vec& y, const arma::mat& K, const int nimp, const arma::vec& theta, const Rcpp::List laplace_approx);

//SEXP start_profiler(SEXP str);

//SEXP stop_profiler();

#endif
