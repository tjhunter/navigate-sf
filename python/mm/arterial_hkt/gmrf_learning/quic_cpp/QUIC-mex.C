// $Id: QUIC-mex.C,v 1.15 2012-02-28 22:19:54 sustik Exp $

// This is the MEX wrapper for QUIC.  The algorithm is in QUIC.C.

// Invocation form within Matlab or Octave:
// [X W opt time iter] = QUIC(mode, ...)
// [X W opt time iter] = QUIC("default", S, L, tol, msg, maxIter, X0, W0)
// [X W opt time iter] = QUIC("path", S, L, path, tol, msg, maxIter, X0, W0)
// [X W opt time iter] = QUIC("trace", S, L, tol, msg, maxIter, X0, W0)
// See the README file and QUIC.m for more information.

#include <mex.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <stdint.h>

extern "C"
void QUIC(char mode, uint32_t& p, const double* S, double* Lambda,
	  uint32_t& pathLen, const double* path, double& tol,
	  int32_t& msg, uint32_t* iter, double* X,
	  double* W, double* opt, double* time);

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    if (nrhs < 2) {
	mexErrMsgIdAndTxt("QUIC:arguments",
			  "Missing arguments, please specify\n"
	    "             S - the empirical covariance matrix, and\n"
	    "             L - the regularization parameter.");
    }
    long argIdx = 0;
    char mode[8];
    mode[0] = 'd';
    if (mxIsChar(prhs[0])) {
	mxGetString(prhs[0], mode, 8);
	if (strcmp(mode, "path") &&
	    strcmp(mode, "trace") &&
	    strcmp(mode, "default"))
	    mexErrMsgIdAndTxt("QUIC:arguments",
			      "Invalid mode, use: 'default', 'path' or "
			      "'trace'.");
	argIdx++;
    }
    // The empirical covariance matrix:
    const double* S = mxGetPr(prhs[argIdx]);
    uint32_t p = mxGetN(prhs[argIdx]);
    if (p != mxGetM(prhs[argIdx])) {
	mexErrMsgIdAndTxt("QUIC:dimensions",
			  "Expected a square empirical covariance matrix.");
    }
    argIdx++;

    // Regularization parameter matrix:
    double* Lambda;
    unsigned long LambdaAlloc = 0;
    if (mxGetN(prhs[argIdx]) == 1 && mxGetM(prhs[argIdx]) == 1) {
	Lambda = (double*) malloc(p*p*sizeof(double));
	LambdaAlloc = 1;
	double lambda = mxGetPr(prhs[argIdx])[0];
	for (unsigned long i = 0; i < p*p; i++)
	    Lambda[i] = lambda;
    } else {
	if (mxGetN(prhs[argIdx]) != p && mxGetM(prhs[argIdx]) != p) {
	    mexErrMsgIdAndTxt("QUIC:dimensions",
			      "The regularization parameter is not a scalar\n"
		"              or a matching matrix.");
	}
	Lambda = mxGetPr(prhs[argIdx]);
    }
    argIdx++;

    uint32_t pathLen = 1;
    double* path = NULL;
    if (mode[0] == 'p') {
	pathLen = mxGetN(prhs[argIdx]);
	path = mxGetPr(prhs[argIdx]);
	if (pathLen <= 1) {
	    mexErrMsgIdAndTxt("QUICpath:dimensions",
			      "At least two path scaling values are "
			      "expected.");
	}
	argIdx++;
    }

    double tol = 1e-6;
    int32_t msg = 0;	
    uint32_t iter0 = 1000;
    // Tolerance value:
    if (nrhs > argIdx) {
	tol = mxGetPr(prhs[argIdx])[0];
	if (tol < 0) {
	    mexErrMsgIdAndTxt("QUIC:tolerance",
			      "Negative tolerance value.");
	}
    }
    argIdx++;
    
    if (nrhs > argIdx)
	msg = mxGetScalar(prhs[argIdx]);
    argIdx++;
    
    // Maximum number of Newton ierations (whole matrix update):
    if (nrhs > argIdx) {
	iter0 = mxGetScalar(prhs[argIdx]);
	if (iter0 > 1000)
	    iter0 = 1000;
    }
    argIdx++;

    double* X0 = NULL;
    double* W0 = NULL;
    if (nrhs > argIdx) {
	if (p != mxGetM(prhs[argIdx]) || p != mxGetN(prhs[argIdx]))
	    mexErrMsgIdAndTxt("QUIC:dimensions",
			      "Matrix dimensions should match.");
	X0 = mxGetPr(plhs[argIdx]);
	argIdx++;
	if (nrhs == argIdx)
	    mexErrMsgIdAndTxt("QUIC:initializations",
			      "Please specify both the initial estimate\n"
		"              and the inverse.");
	if (p != mxGetM(prhs[argIdx]) || p != mxGetN(prhs[argIdx])) {
	    mexErrMsgIdAndTxt("QUIC:dimensions",
			      "Matrix dimensions should match.");
	}
	W0 = mxGetPr(plhs[argIdx]);
	argIdx++;
    }

    double* X = NULL;
    double* W = NULL;
    if (mode[0] == 'p') {
	mwSize dims[] = {p, p, pathLen};
	mxArray* tmp = mxCreateNumericArray(3, dims, mxDOUBLE_CLASS, mxREAL);
	X = (double*) mxGetPr(tmp);
	if (nlhs > 0)
	    plhs[0] = tmp;
	tmp = mxCreateNumericArray(3, dims, mxDOUBLE_CLASS, mxREAL);
	W = (double*) mxGetPr(tmp);
	if (nlhs > 1)
	    plhs[1] = tmp;
    } else {
	mwSize dims[] = {p, p};
	mxArray* tmp = mxCreateNumericArray(2, dims, mxDOUBLE_CLASS, mxREAL);
	X = (double*) mxGetPr(tmp);
	if (nlhs > 0)
	    plhs[0] = tmp;
	tmp = mxCreateNumericArray(2, dims, mxDOUBLE_CLASS, mxREAL);
	W = (double*) mxGetPr(tmp);
	if (nlhs > 1)
	    plhs[1] = tmp;	
    }
    if (X0 != NULL) {
	memcpy(X, X0, sizeof(double)*p*p);
	memcpy(W, W0, sizeof(double)*p*p);
    } else {
	memset(X, 0, sizeof(double)*p*p);
	memset(W, 0, sizeof(double)*p*p);
	for (unsigned long i = 0; i < p*p; i += (p+1)) {
	    X[i] = 1.0;
	    W[i] = 1.0;
	}
    }
    double* opt = NULL;
    double* time = NULL;
    uint32_t* iter = NULL;
    unsigned long optsize = 1;
    if (mode[0] == 'p')
	optsize = pathLen;
    else if (mode[0] == 't')
	optsize = iter0;
    if (nlhs > 2) {
	plhs[2] = mxCreateDoubleMatrix(optsize, 1, mxREAL);
	opt = mxGetPr(plhs[2]);
    }
    if (nlhs > 3) {
	plhs[3] = mxCreateDoubleMatrix(optsize, 1, mxREAL);
	time = mxGetPr(plhs[3]);
    }
    if (mode[0] == 'p') {
	mwSize dims[] = {pathLen};
	mxArray* iter_mx = mxCreateNumericArray(1, dims, mxINT32_CLASS, mxREAL);
	if (nlhs > 4)
	    plhs[4] = iter_mx;
	iter = (uint32_t*) mxGetData(iter_mx);
	iter[0] = iter0;
    } else {
	mwSize dims[] = {1};
	mxArray* iter_mx = mxCreateNumericArray(1, dims, mxINT32_CLASS, mxREAL);
	if (nlhs > 4)
	    plhs[4] = iter_mx;
	iter = (uint32_t*) mxGetData(iter_mx);
	iter[0] = iter0;
    }

    QUIC(mode[0], p, S, Lambda, pathLen, path, tol, msg, iter, X, W,
	 opt, time);
    if (LambdaAlloc)
	free(Lambda);
}
