#include <Python.h>
#include <boost/python.hpp>
#include <numpy/arrayobject.h>

#include <iostream>
using namespace std;
#include <boost/cstdint.hpp>
// #include <Eigen/Core>
// using namespace Eigen;

extern "C" {

void QUIC(char mode, uint32_t& p, const double* S, double* Lambda0,
	  uint32_t& pathLen, const double* path, double& tol,
	  int32_t& msg, uint32_t* iter, double* X,
	  double* W, double* opt, double* cputime);
}

class Quic {
public:
  void compute(char mode, uint32_t* p_, PyObject* S_, PyObject* Lambda0_,
		      uint32_t* pathLen, PyObject* path_, double* tol,
		      int32_t* msg, uint32_t* iter, PyObject* X_,
		      PyObject* W_, PyObject* opt_, PyObject* cputime_);
  void compute2(char mode, uint32_t p_, PyObject* S_, PyObject* Lambda0_,
		      uint32_t pathLen, PyObject* path_, double tol,
		      int32_t msg, PyObject* iter, PyObject* X_,
		      PyObject* W_, PyObject* opt_, PyObject* cputime_);
};

void Quic::compute(char mode, uint32_t* p_, PyObject* S_, PyObject* Lambda0_,
		      uint32_t* pathLen_, PyObject* path_, double* tol_,
		      int32_t* msg_, uint32_t* iter_, PyObject* X_,
		      PyObject* W_, PyObject* opt_, PyObject* cputime_) {
// uint32_t& p, const double* S, double* Lambda0,
// 	  uint32_t& pathLen, const double* path, double& tol,
// 	  int32_t& msg, uint32_t* iter, double* X,
// 	  double* W, double* opt, double* cputime  cout << "Calling quic" << endl;
  uint32_t &p = *p_;
  double * S = (double *) PyArray_DATA(S_);
  double * Lambda0 = (double *) PyArray_DATA(Lambda0_);
  uint32_t &pathLen = *pathLen_;
  double * path = (double *) PyArray_DATA(path_);
  double &tol = *tol_;
  int32_t& msg = *msg_;
  uint32_t* iter = iter_;
  double * X = (double *) PyArray_DATA(X_);
  double * W = (double *) PyArray_DATA(W_);
  double * opt = (double *) PyArray_DATA(opt_);
  double * cputime = (double *) PyArray_DATA(cputime_);
  QUIC(mode, p, S, Lambda0,
	  pathLen, path, tol,
	  msg, iter, X,
	  W, opt, cputime);
}


void Quic::compute2(char mode, uint32_t p_, PyObject* S_, PyObject* Lambda0_,
		      uint32_t pathLen, PyObject* path_, double tol,
		      int32_t msg, PyObject* iter_, PyObject* X_,
		      PyObject* W_, PyObject* opt_, PyObject* cputime_) {
// uint32_t& p, const double* S, double* Lambda0,
// 	  uint32_t& pathLen, const double* path, double& tol,
// 	  int32_t& msg, uint32_t* iter, double* X,
// 	  double* W, double* opt, double* cputime  cout << "Calling quic" << endl;
//   uint32_t &p = *p_;
  double * S = (double *) PyArray_DATA(S_);
  double * Lambda0 = (double *) PyArray_DATA(Lambda0_);
//   uint32_t &pathLen = *pathLen_;
  double * path = (double *) PyArray_DATA(path_);
//   double &tol = *tol_;
//   int32_t& msg = *msg_;
  uint32_t* iter = (uint32_t *) PyArray_DATA(iter_);;
  double * X = (double *) PyArray_DATA(X_);
  double * W = (double *) PyArray_DATA(W_);
  double * opt = (double *) PyArray_DATA(opt_);
  double * cputime = (double *) PyArray_DATA(cputime_);
  QUIC(mode, p_, S, Lambda0,
	  pathLen, path, tol,
	  msg, iter, X,
	  W, opt, cputime);
}

using namespace boost::python;
BOOST_PYTHON_MODULE(quic)
{
      class_<Quic>("Quic")
      .def("compute2", &Quic::compute2)  
      .def("compute", &Quic::compute);

}
