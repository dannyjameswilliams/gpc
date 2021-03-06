// Generated by using Rcpp::compileAttributes() -> do not edit by hand
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include <RcppArmadillo.h>
#include <Rcpp.h>

using namespace Rcpp;

// algo_1
Rcpp::List algo_1(arma::vec& y, arma::mat& X, const int& nsteps, const int& nburn, const int& nimp, const arma::vec& init_theta, const arma::vec& init_marginal_lik, const arma::vec& init_f, const Rcpp::Function& prior_density, const Rcpp::Function& kernel, const Rcpp::Function& prop_dist_sample, const Rcpp::Function& prop_dist_density, const int print_every, const bool debug, const int chain_no, Rcpp::String kernel_pass);
RcppExport SEXP _gpc_algo_1(SEXP ySEXP, SEXP XSEXP, SEXP nstepsSEXP, SEXP nburnSEXP, SEXP nimpSEXP, SEXP init_thetaSEXP, SEXP init_marginal_likSEXP, SEXP init_fSEXP, SEXP prior_densitySEXP, SEXP kernelSEXP, SEXP prop_dist_sampleSEXP, SEXP prop_dist_densitySEXP, SEXP print_everySEXP, SEXP debugSEXP, SEXP chain_noSEXP, SEXP kernel_passSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::vec& >::type y(ySEXP);
    Rcpp::traits::input_parameter< arma::mat& >::type X(XSEXP);
    Rcpp::traits::input_parameter< const int& >::type nsteps(nstepsSEXP);
    Rcpp::traits::input_parameter< const int& >::type nburn(nburnSEXP);
    Rcpp::traits::input_parameter< const int& >::type nimp(nimpSEXP);
    Rcpp::traits::input_parameter< const arma::vec& >::type init_theta(init_thetaSEXP);
    Rcpp::traits::input_parameter< const arma::vec& >::type init_marginal_lik(init_marginal_likSEXP);
    Rcpp::traits::input_parameter< const arma::vec& >::type init_f(init_fSEXP);
    Rcpp::traits::input_parameter< const Rcpp::Function& >::type prior_density(prior_densitySEXP);
    Rcpp::traits::input_parameter< const Rcpp::Function& >::type kernel(kernelSEXP);
    Rcpp::traits::input_parameter< const Rcpp::Function& >::type prop_dist_sample(prop_dist_sampleSEXP);
    Rcpp::traits::input_parameter< const Rcpp::Function& >::type prop_dist_density(prop_dist_densitySEXP);
    Rcpp::traits::input_parameter< const int >::type print_every(print_everySEXP);
    Rcpp::traits::input_parameter< const bool >::type debug(debugSEXP);
    Rcpp::traits::input_parameter< const int >::type chain_no(chain_noSEXP);
    Rcpp::traits::input_parameter< Rcpp::String >::type kernel_pass(kernel_passSEXP);
    rcpp_result_gen = Rcpp::wrap(algo_1(y, X, nsteps, nburn, nimp, init_theta, init_marginal_lik, init_f, prior_density, kernel, prop_dist_sample, prop_dist_density, print_every, debug, chain_no, kernel_pass));
    return rcpp_result_gen;
END_RCPP
}
// get_approx_marginal
double get_approx_marginal(const arma::vec y, const arma::mat& K, const int nimp, const arma::vec theta, const Rcpp::List laplace_approx);
RcppExport SEXP _gpc_get_approx_marginal(SEXP ySEXP, SEXP KSEXP, SEXP nimpSEXP, SEXP thetaSEXP, SEXP laplace_approxSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::vec >::type y(ySEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type K(KSEXP);
    Rcpp::traits::input_parameter< const int >::type nimp(nimpSEXP);
    Rcpp::traits::input_parameter< const arma::vec >::type theta(thetaSEXP);
    Rcpp::traits::input_parameter< const Rcpp::List >::type laplace_approx(laplace_approxSEXP);
    rcpp_result_gen = Rcpp::wrap(get_approx_marginal(y, K, nimp, theta, laplace_approx));
    return rcpp_result_gen;
END_RCPP
}
// get_approx_marginal_par
double get_approx_marginal_par(arma::vec& y, const arma::mat& K, const int nimp, const arma::vec& theta, const Rcpp::List laplace_approx);
RcppExport SEXP _gpc_get_approx_marginal_par(SEXP ySEXP, SEXP KSEXP, SEXP nimpSEXP, SEXP thetaSEXP, SEXP laplace_approxSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::vec& >::type y(ySEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type K(KSEXP);
    Rcpp::traits::input_parameter< const int >::type nimp(nimpSEXP);
    Rcpp::traits::input_parameter< const arma::vec& >::type theta(thetaSEXP);
    Rcpp::traits::input_parameter< const Rcpp::List >::type laplace_approx(laplace_approxSEXP);
    rcpp_result_gen = Rcpp::wrap(get_approx_marginal_par(y, K, nimp, theta, laplace_approx));
    return rcpp_result_gen;
END_RCPP
}
// ell_ss_sample
arma::vec ell_ss_sample(const arma::vec& y, const arma::vec& f, arma::mat& K);
RcppExport SEXP _gpc_ell_ss_sample(SEXP ySEXP, SEXP fSEXP, SEXP KSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::vec& >::type y(ySEXP);
    Rcpp::traits::input_parameter< const arma::vec& >::type f(fSEXP);
    Rcpp::traits::input_parameter< arma::mat& >::type K(KSEXP);
    rcpp_result_gen = Rcpp::wrap(ell_ss_sample(y, f, K));
    return rcpp_result_gen;
END_RCPP
}
// build_K
arma::mat build_K(const arma::mat& x, const arma::mat& y, Rcpp::Function k, const arma::vec& theta);
RcppExport SEXP _gpc_build_K(SEXP xSEXP, SEXP ySEXP, SEXP kSEXP, SEXP thetaSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::mat& >::type x(xSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type y(ySEXP);
    Rcpp::traits::input_parameter< Rcpp::Function >::type k(kSEXP);
    Rcpp::traits::input_parameter< const arma::vec& >::type theta(thetaSEXP);
    rcpp_result_gen = Rcpp::wrap(build_K(x, y, k, theta));
    return rcpp_result_gen;
END_RCPP
}
// make_gram_par
arma::mat make_gram_par(arma::mat& x, arma::mat& y, const arma::vec& theta);
RcppExport SEXP _gpc_make_gram_par(SEXP xSEXP, SEXP ySEXP, SEXP thetaSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::mat& >::type x(xSEXP);
    Rcpp::traits::input_parameter< arma::mat& >::type y(ySEXP);
    Rcpp::traits::input_parameter< const arma::vec& >::type theta(thetaSEXP);
    rcpp_result_gen = Rcpp::wrap(make_gram_par(x, y, theta));
    return rcpp_result_gen;
END_RCPP
}
// laplace_approx
Rcpp::List laplace_approx(const arma::vec& y, const arma::mat& K);
RcppExport SEXP _gpc_laplace_approx(SEXP ySEXP, SEXP KSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::vec& >::type y(ySEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type K(KSEXP);
    rcpp_result_gen = Rcpp::wrap(laplace_approx(y, K));
    return rcpp_result_gen;
END_RCPP
}
// dmvnorm
arma::vec dmvnorm(arma::mat const& x, arma::vec const& mean, arma::mat sigma, bool const logd);
RcppExport SEXP _gpc_dmvnorm(SEXP xSEXP, SEXP meanSEXP, SEXP sigmaSEXP, SEXP logdSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::mat const& >::type x(xSEXP);
    Rcpp::traits::input_parameter< arma::vec const& >::type mean(meanSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type sigma(sigmaSEXP);
    Rcpp::traits::input_parameter< bool const >::type logd(logdSEXP);
    rcpp_result_gen = Rcpp::wrap(dmvnorm(x, mean, sigma, logd));
    return rcpp_result_gen;
END_RCPP
}
// predict_gp
arma::vec predict_gp(const arma::vec& y, arma::mat& X, arma::mat& newdata, const Rcpp::Function kernel, const Rcpp::List fit, const int nchains, const Rcpp::String kernel_pass, const int print_every);
RcppExport SEXP _gpc_predict_gp(SEXP ySEXP, SEXP XSEXP, SEXP newdataSEXP, SEXP kernelSEXP, SEXP fitSEXP, SEXP nchainsSEXP, SEXP kernel_passSEXP, SEXP print_everySEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::vec& >::type y(ySEXP);
    Rcpp::traits::input_parameter< arma::mat& >::type X(XSEXP);
    Rcpp::traits::input_parameter< arma::mat& >::type newdata(newdataSEXP);
    Rcpp::traits::input_parameter< const Rcpp::Function >::type kernel(kernelSEXP);
    Rcpp::traits::input_parameter< const Rcpp::List >::type fit(fitSEXP);
    Rcpp::traits::input_parameter< const int >::type nchains(nchainsSEXP);
    Rcpp::traits::input_parameter< const Rcpp::String >::type kernel_pass(kernel_passSEXP);
    Rcpp::traits::input_parameter< const int >::type print_every(print_everySEXP);
    rcpp_result_gen = Rcpp::wrap(predict_gp(y, X, newdata, kernel, fit, nchains, kernel_pass, print_every));
    return rcpp_result_gen;
END_RCPP
}
// chol_plus_diag
arma::mat chol_plus_diag(arma::mat& A, Rcpp::String type);
RcppExport SEXP _gpc_chol_plus_diag(SEXP ASEXP, SEXP typeSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::mat& >::type A(ASEXP);
    Rcpp::traits::input_parameter< Rcpp::String >::type type(typeSEXP);
    rcpp_result_gen = Rcpp::wrap(chol_plus_diag(A, type));
    return rcpp_result_gen;
END_RCPP
}

static const R_CallMethodDef CallEntries[] = {
    {"_gpc_algo_1", (DL_FUNC) &_gpc_algo_1, 16},
    {"_gpc_get_approx_marginal", (DL_FUNC) &_gpc_get_approx_marginal, 5},
    {"_gpc_get_approx_marginal_par", (DL_FUNC) &_gpc_get_approx_marginal_par, 5},
    {"_gpc_ell_ss_sample", (DL_FUNC) &_gpc_ell_ss_sample, 3},
    {"_gpc_build_K", (DL_FUNC) &_gpc_build_K, 4},
    {"_gpc_make_gram_par", (DL_FUNC) &_gpc_make_gram_par, 3},
    {"_gpc_laplace_approx", (DL_FUNC) &_gpc_laplace_approx, 2},
    {"_gpc_dmvnorm", (DL_FUNC) &_gpc_dmvnorm, 4},
    {"_gpc_predict_gp", (DL_FUNC) &_gpc_predict_gp, 8},
    {"_gpc_chol_plus_diag", (DL_FUNC) &_gpc_chol_plus_diag, 2},
    {NULL, NULL, 0}
};

RcppExport void R_init_gpc(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}
