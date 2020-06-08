#include "RcppArmadillo.h"
#include "RcppParallel.h"
#include "functions.h"
#include "cmath"
#include "algorithm"

// [[Rcpp::depends(RcppParallel)]]
// [[Rcpp::plugins(cpp11)]]

using namespace arma;

//' Make Gram Matrix
//'
//' Compute the Gram (covariance) matrix
//'
//' @param x input matrix 1
//' @param y input matrix 2
//' @param k R function to apply covariance function for elements of \code{x} and \code{y}
//' @param theta hyperparameter vector to input to \code{k}
//'
//' @details
//' The kernel function \code{k} needs to have arguments corresponding to  \code{x}, \code{y}, \code{theta} in that order.
//'
//' \code{build_K} computes the gram matrix for a given \code{k}. For a faster, parallel approach,
//' \code{make_gram_par} uses a pre-determined Gaussian covariance function.
//'
// [[Rcpp::export(name="build_K")]]
arma::mat build_K(const arma::mat& x, const arma::mat& y, Rcpp::Function k, const arma::vec& theta)
{
  int nx = x.n_rows;
  int ny = y.n_rows;
  arma::mat K(nx, ny);
  for(int i = 0; i < nx; i++){
    for(int j = 0; j < ny; j++){
      K(i, j) = Rcpp::as<double>(k(x.row(i), y.row(j), theta));
    }
  }

  if(nx == ny)
  {
    for(int i = 0; i < nx; i++){
      K(i, i) += 1e-6;
    }
  }
  return K;
}


template <typename InputIterator1>
inline double kernel_gaussian_eval(InputIterator1 begin1, InputIterator1 end1, double magnitude, double lengthscale) {

  // value to store the sum of squared differences
  double red = 0;

  // set iterator to beginning of range
  InputIterator1 it1 = begin1;

  // for each input item
  while (it1 != end1) {

    // take the value and increment the iterator
    double d1 = *it1++;

    // sum the entries in the iterator
    red += d1;
  }

  return magnitude * std::exp(-0.5 / std::pow(lengthscale, 2) * red);
}



struct kernel_eval : public RcppParallel::Worker
{
  const RcppParallel::RMatrix<double> x_inp;
  const RcppParallel::RMatrix<double> y_inp;
  const double lengthscale;
  const double magnitude;
  RcppParallel::RMatrix<double> out;

  kernel_eval(const Rcpp::NumericMatrix x_in_,
              const Rcpp::NumericMatrix y_in_,
              double magnitude,
              double lengthscale,
              Rcpp::NumericMatrix out_)
    : x_inp(x_in_),
      y_inp(y_in_),
      magnitude(magnitude),
      lengthscale(lengthscale),
      out(out_) {}

  void operator()(std::size_t begin, std::size_t end) {
    for(std::size_t i = begin; i < end; i++){
      for(std::size_t j = 0; j < y_inp.nrow(); j++){

        RcppParallel::RMatrix<double>::Row row1 = x_inp.row(i);
        RcppParallel::RMatrix<double>::Row row2 = y_inp.row(j);

        std::vector<double> sq_diff(row1.size());
        std::transform(row1.begin(), row1.end(),
          row2.begin(),
          sq_diff.begin(),
          [](double x, double y) { return std::pow(x-y, 2); } );

        out(i, j) = kernel_gaussian_eval(sq_diff.begin(), sq_diff.end(), magnitude, lengthscale);
      }
    }
  }
};

//' Make Gram Matrix (Gaussian covariance)
//'
//' Compute the Gram (covariance) matrix in parallel with a Gaussian covariance function
//'
//' @param x input matrix 1
//' @param y input matrix 2
//' @param theta hyperparameter vector
//'
//' @details
//' This is a faster, parallel approach to build the Gram matrix using a pre-determined Gaussian covariance function.
//' \code{build_K} computes the gram matrix for a given \code{k}, but is slower and not in parallel.
// [[Rcpp::export]]
arma::mat make_gram_par(arma::mat& x, arma::mat& y, const arma::vec& theta) {

  int n_x = x.n_rows;
  int n_y = y.n_rows;
  Rcpp::NumericMatrix x_nm = mat_to_rcpp(x);
  Rcpp::NumericMatrix y_nm = mat_to_rcpp(y);
  Rcpp::NumericMatrix out_nm(n_x, n_y);

  double magnitude = theta(0);
  double lengthscale = theta(1);

  kernel_eval obj(x_nm, y_nm, magnitude, lengthscale, out_nm);

  RcppParallel::parallelFor(0, x_nm.nrow(), obj);

  arma::mat out_mat = rcpp_to_mat(out_nm);

  out_mat.diag() += 1e-3;

  return out_mat;
}
