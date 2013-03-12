// $Id: QUIC.C,v 1.41 2012-02-28 22:19:54 sustik Exp $

// Solves the regularized inverse covariance matrix selection using a
// combination of Newton's method, quadratic approximation and
// coordinate descent.  The original algorithm was coded by Cho-Jui
// Hsieh.  Improvements were made by Matyas A. Sustik.
// This code is released under the GPL version 3.

// See the README file and QUIC.m for more information.
// Send questions, comments and license inquiries to: sustik@cs.utexas.edu

#define VERSION "2012.02"

#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#ifdef GDEBUG
#include "startgdb.h"
#endif

#ifndef MSG
  #define MSG printf
#endif

#define EPS (double(2.22E-16))
//#define EPS ((double)0x3cb0000000000000)

typedef struct {
    unsigned short i;
    unsigned short j;
} ushort_pair_t;

// It would be preferable to use an include such as lapack.h.  Except
// lapack.h is not available from the octave or liblapack-dev packages...
extern "C" {
    void dpotrf_(char* uplo, ptrdiff_t* n, double* A, ptrdiff_t* lda,
		 ptrdiff_t* info);
    void dpotri_(char* uplo, ptrdiff_t* n, double* A, ptrdiff_t* lda,
		 ptrdiff_t* info);
}

static inline unsigned long IsDiag(unsigned long p, const double* A)
{
    for (unsigned long k = 0, i = 0; i < p; i++, k += p)
	for (unsigned long j = 0; j < i; j++)
	    if (A[k + j] != 0.0)
		return 0;
    return 1;
}

static inline void CoordinateDescentUpdate(
    unsigned long p, const double* const S, const double* const Lambda,
    const double* X, const double* W, double* U, double* D,
    unsigned long i, unsigned long j, double& normD, double& diffD)
{
    unsigned long ip = i*p;
    unsigned long jp = j*p;
    unsigned long ij = ip + j;

    double a = W[ij]*W[ij];
    if (i != j)
        a += W[ip+i]*W[jp+j];
    double ainv = 1.0/a;  // multiplication is cheaper than division

    double b = S[ij] - W[ij];
    for (unsigned long k = 0; k < p ; k++)
        b += W[ip+k]*U[k*p+j];

    double l = Lambda[ij]*ainv;
    double c = X[ij] + D[ij];
    double f = b*ainv;
    double mu;
    normD -= fabs(D[ij]);
    if (c > f) {
        mu = -f - l;
        if (c + mu < 0.0) {
            mu = -c;
	    D[ij] = -X[ij];
	} else {
	    D[ij] += mu;
	}
    } else {
	mu = -f + l;
	if (c + mu > 0.0) {
	    mu = -c;
	    D[ij] = -X[ij];
	} else {
	    D[ij] += mu;
	}
    }
    diffD += fabs(mu);
    normD += fabs(D[ij]);
    if (mu != 0.0) {
        for (unsigned long k = 0; k < p; k++)
            U[ip+k] += mu*W[jp+k];
        if (i != j) {
            for (unsigned long k = 0; k < p; k++)
                U[jp+k] += mu*W[ip+k];
        }
    }
}

// Return the objective value.
static inline double DiagNewton(unsigned long p, const double* S,
				const double* Lambda, const double* X,
				const double* W, double* D)
{
    for (unsigned long ip = 0, i = 0; i < p; i++, ip += p) {
	for (unsigned long jp = 0, j = 0; j < i; j++, jp += p) {
	    unsigned long ij = ip + j;
	    double a = W[ip + i]*W[jp + j];
	    double ainv = 1.0/a;  // multiplication is cheaper than division
	    double b = S[ij];
	    double l = Lambda[ij]*ainv;
	    double f = b*ainv;
	    double mu;
	    double x = -b*ainv;
	    if (0 > f) {
		mu = -f - l;
		x -= l;
		if (mu < 0.0) {
		    mu = 0.0;
		    D[ij] = -X[ij];
		} else {
		    D[ij] += mu;
		}
	    } else {
		mu = -f + l;
		if (mu > 0.0) {
		    mu = 0.0;
		    D[ij] = -X[ij];
		} else {
		    D[ij] += mu;
		}
	    }
	}
    }
    double logdet = 0.0;
    double l1normX = 0.0;
    double trSX = 0.0;
    for (unsigned long i = 0, k = 0; i < p; i++, k += (p+1)) {
	logdet += log(X[k]);
	l1normX += fabs(X[k])*Lambda[k];
	trSX += X[k]*S[k];
	double a = W[k]*W[k];
	double ainv = 1.0/a;  // multiplication is cheaper than division
	double b = S[k] - W[k];
	double l = Lambda[k]*ainv;
	double c = X[k];
	double f = b*ainv;
	double mu;
	if (c > f) {
	    mu = -f - l;
	    if (c + mu < 0.0) {
		D[k] = -X[k];
		continue;
	    }
	} else {
	    mu = -f + l;
	    if (c + mu > 0.0) {
		D[k] = -X[k];
		continue;
	    }
	}
	D[k] += mu;
    }
    double fX = -logdet + trSX + l1normX;
    return fX;
}

#define QUIC_MSG_NO      0
#define QUIC_MSG_TIME    1
#define QUIC_MSG_NEWTON  2
#define QUIC_MSG_CD      3
#define QUIC_MSG_LINE    4

// mode = {'D', 'P', 'T'} for 'default', 'path' or 'trace'. 
extern "C"
void QUIC(char mode, uint32_t& p, const double* S, double* Lambda0,
	  uint32_t& pathLen, const double* path, double& tol,
	  int32_t& msg, uint32_t* iter, double* X,
	  double* W, double* opt, double* cputime)
{
#ifdef GDEBUG
    startgdb();
#endif
    if (mode >= 'a')
	mode -= ('a' - 'A');
    char m[] = "Running QUIC version";
    if (msg >= QUIC_MSG_TIME) {
	if (mode == 'P')
	    MSG("%s %s in 'path' mode.\n", m, VERSION);
	else if (mode == 'T')
	    MSG("%s %s in 'trace' mode.\n", m, VERSION);
	else
	    MSG("%s %s in 'default' mode.\n", m, VERSION);
    }
    double timeBegin = clock();
    srand(1);
    unsigned long maxNewtonIter = iter[0];
    double cdSweepTol = 0.05;
    unsigned long max_lineiter = 20;
    double fX = 1e+15;
    double fX1 = 1e+15;
    double fXprev = 1e+15;
    double sigma = 0.001;
    double* D = (double*) malloc(p*p*sizeof(double));
    double* X1 = (double*) malloc(p*p*sizeof(double));
    double* U = (double*) malloc(p*p*sizeof(double));
    double* Lambda;
    if (pathLen > 1) {
	Lambda = (double*) malloc(p*p*sizeof(double));
	for (unsigned long i = 0; i < p*p; i++)
	    Lambda[i] = Lambda0[i]*path[0];
    } else {
	Lambda = Lambda0;
    }
    ushort_pair_t* activeSet = (ushort_pair_t*) 
	malloc(p*(p+1)/2*sizeof(ushort_pair_t));
    double l1normX = 0.0;
    double trSX = 0.0;
    for (unsigned long i = 0, k = 0; i < p ; i++, k += p) {
	for (unsigned long j = 0; j < i; j++) {
	    l1normX += Lambda[k+j]*fabs(X[k+j]);
	    trSX += X[k+j]*S[k+j];
	}
    }
    l1normX *= 2.0;
    trSX *= 2.0;
    for (unsigned long i = 0, k = 0; i < p; i++, k += (p+1)) {
	l1normX += Lambda[k]*fabs(X[k]);
	trSX += X[k]*S[k];
    }
    unsigned long pathIdx = 0;
    unsigned long NewtonIter = 1;
    for (; NewtonIter <= maxNewtonIter; NewtonIter++) {
	double normD = 0.0;
	double diffD = 0.0;
	double subgrad = 1e+15;	
	if (NewtonIter == 1 && IsDiag(p, X)) {
	    if (msg >= QUIC_MSG_NEWTON) {
		MSG("Newton iteration 1.\n");
		MSG("  X is a diagonal matrix.\n");
	    }
	    memset(D, 0, p*p*sizeof(double));
	    fX = DiagNewton(p, S, Lambda, X, W, D);
	} else {
	    // Compute the active set and the minimum norm subgradient:
	    unsigned long numActive = 0;
	    memset(U, 0, p*p*sizeof(double));
	    memset(D, 0, p*p*sizeof(double));
	    subgrad = 0.0;
	    for (unsigned long k = 0, i = 0; i < p; i++, k += p) {
		for (unsigned long j = 0; j <= i; j++) {
		    double g = S[k+j] - W[k+j];
		    if (X[k+j] != 0.0 || (fabs(g) > Lambda[k+j])) {
			activeSet[numActive].i = i;
			activeSet[numActive].j = j;
			numActive++;
			if (X[k+j] > 0) 
			    g += Lambda[k+j];
			else if (X[k+j] < 0) 
			    g -= Lambda[k+j];
			else 
			    g = fabs(g) - Lambda[k+j];
			subgrad += fabs(g);
		    }
		}
	    }
	    if (msg >= QUIC_MSG_NEWTON) {
		MSG("Newton iteration %ld.\n", NewtonIter);
		MSG("  Active set size = %ld.\n", numActive);
		MSG("  sub-gradient = %le, l1-norm of X = %le.\n",
		       subgrad, l1normX);
	    }
	    for (unsigned long cdSweep = 1; cdSweep <= 1 + NewtonIter/3;
		 cdSweep++) {
		diffD = 0.0;
		for (unsigned long i = 0; i < numActive; i++ ) {
		    unsigned long j = i + rand()%(numActive - i);
		    unsigned long k1 = activeSet[i].i, k2 = activeSet[i].j;
		    activeSet[i].i = activeSet[j].i;
		    activeSet[i].j = activeSet[j].j;
		    activeSet[j].i = k1;
		    activeSet[j].j = k2;
		}
		for (unsigned long l = 0; l < numActive; l++) {
		    int i = activeSet[l].i;
		    int j = activeSet[l].j;
		    CoordinateDescentUpdate(p, S, Lambda, X, W,
					    U, D, i, j, normD,
					    diffD);
		}
		if (msg >= QUIC_MSG_CD) {
		    MSG("  Coordinate descent sweep %ld. norm of D = %le, "
			   "change in D = %le.\n", cdSweep, normD, diffD);
		}
		if (diffD < normD*cdSweepTol)
		    break;
	    }
	}
	if (fX == 1e+15) {
	    // Note that the upper triangular part is the lower
	    // triangular part for the C arrays.
	    ptrdiff_t info = 0;
	    ptrdiff_t p0 = p;
	    memcpy(X1, X, sizeof(double)*p*p);
	    dpotrf_((char*) "U", &p0, X1, &p0, &info);
	    if (info != 0) {
		MSG("Error! Lack of positive definiteness!");
		iter[0] = -1;
		free(activeSet);
		free(U);
		free(X1);
		free(D);
		if (pathLen > 1)
		    free(Lambda);
		return;
	    }
	    double detX = 0.0;
	    for (unsigned long i = 0, k = 0; i < p; i++, k += (p+1))
		detX += log(X1[k]);
	    detX *= 2.0;
	    fX = -detX + trSX + l1normX;
	}
	double trgradgD = 0.0;
	for (unsigned long i = 0, k = 0; i < p ; i++, k += p)
	    for (unsigned long j = 0; j < i; j++)
		trgradgD += (S[k+j]-W[k+j])*D[k+j];
	trgradgD *= 2.0;
	for (unsigned long i = 0, k = 0; i < p; i++, k += (p+1))
	    trgradgD += (S[k]-W[k])*D[k];

	double alpha = 1.0;
	double l1normXD = 0.0;
	double fX1prev = 1e+15;
	for (unsigned long lineiter = 0; lineiter < max_lineiter;
	     lineiter++) {
	    double l1normX1 = 0.0;
	    double trSX1 = 0.0;
	    for (unsigned long i = 0, k = 0; i < p ; i++, k += p) {
		for (unsigned long j = 0; j < i; j++) {
		    unsigned long ij = k + j;
		    X1[ij] = X[ij] + D[ij]*alpha;
		    l1normX1 += fabs(X1[ij])*Lambda[ij];
		    trSX1 += X1[ij]*S[ij];
		}
	    }
	    l1normX1 *= 2.0;
	    trSX1 *= 2.0;
	    for (unsigned long i = 0, k = 0; i < p; i++, k += (p+1)) {
		X1[k] = D[k]*alpha + X[k];
		l1normX1 += fabs(X1[k])*Lambda[k];
		trSX1 += X1[k]*S[k];
	    }
	    // Note that the upper triangular part is the lower
	    // triangular part for the C arrays.
	    ptrdiff_t info = 0;
	    ptrdiff_t p0 = p;
	    dpotrf_((char*) "U", &p0, X1, &p0, &info);
	    if (info != 0) {
		if (msg >= QUIC_MSG_LINE)
		    MSG("    Line search step size %e.  Lack of positive "
			   "definiteness.\n", alpha);
		alpha *= 0.5;
		continue;
	    }
	    double detX1 = 0.0;
	    for (unsigned long i = 0, k = 0; i < p; i++, k += (p+1))
		detX1 += log(X1[k]);
	    detX1 *= 2.0;
	    fX1 = -detX1 + trSX1 + l1normX1;
	    if (alpha == 1.0)
		l1normXD = l1normX1;
	    if (fX1 < fX + alpha*sigma*(trgradgD + l1normXD - l1normX)) {
		if (msg >= QUIC_MSG_LINE)
		    MSG("    Line search step size chosen: %e.\n", alpha);
		fXprev = fX;
		fX = fX1;
		l1normX = l1normX1;
		break;
	    }
	    if (msg >= QUIC_MSG_LINE)
		MSG("    Line search step size %e.  Objective value "
		       "would increase by %e.\n", alpha, fX1 - fX);
	    if (fX1prev < fX1) {
		fXprev = fX;
		l1normX = l1normX1;
		break;
	    }
	    fX1prev = fX1;
	    alpha *= 0.5;
	}
	if (msg >= QUIC_MSG_NEWTON)
	    MSG("  Objective value decreased by %e.\n", fXprev - fX);
	// compute W = inv(X):
	ptrdiff_t info;
	ptrdiff_t p0 = p;
	dpotri_((char*) "U", &p0, X1, &p0, &info);
	
	for (unsigned long i = 0; i < p; i++) {
	    for (unsigned long j = 0; j <= i; j++) {
		double tmp = X1[i*p+j];
		W[j*p+i] = tmp;
		W[i*p+j] = tmp;
	    }
	}
	for (unsigned long i = 0, k = 0; i < p; i++, k += p)
	    for (unsigned long j = 0; j <= i; j++)
		X[k+j] += alpha*D[k+j];
	if (true) {
// 	if (mode == 'T') {
	    if (opt != NULL)
		opt[NewtonIter - 1] = fX;
	    if (cputime != NULL)
		cputime[NewtonIter -1] = (clock()-timeBegin)/CLOCKS_PER_SEC;
	}
	if (subgrad*alpha < l1normX*tol
	    || (fabs((fX - fXprev)/fX) < EPS)) {
	    if (mode =='P') {
		if (opt != NULL)
		    opt[pathIdx] = fX;
		iter[pathIdx] = NewtonIter;
		for (unsigned long i = 0, k = 0; i < p; i++, k += p)
		    for (unsigned long j = i+1; j < p; j++)
			X[k+j] = X[j*p+i];
		double elapsedTime = (clock() - timeBegin)/CLOCKS_PER_SEC;
		if (cputime != NULL)
		    cputime[pathIdx] = elapsedTime;
		// Next lambda.
		pathIdx++;
		if (pathIdx == pathLen)
		    break;
		MSG("  New scaling value: %e\n", path[pathIdx]);
		unsigned long p2 = p*p;
		memcpy(X + p2, X, p2*sizeof(double)); 
		memcpy(W + p2, W, p2*sizeof(double)); 
		X += p2;
		W += p2;
		for (unsigned long i = 0; i < p*p; i++)
		    Lambda[i] = Lambda0[i]*path[pathIdx];
		continue;
	    }
	    break;
	}
    }
    if (mode == 'D' || mode == 'T') {
	iter[0] = NewtonIter;
	if (opt)
	    opt[0] = fX;
    }
    free(activeSet);
    free(U);
    free(X1);
    free(D);
    if (pathLen > 1)
	free(Lambda);
    for (unsigned long i = 0, k = 0; i < p; i++, k += p)
	for (unsigned long j = i+1; j < p; j++)
	    X[k+j] = X[j*p+i];
    double elapsedTime = (clock() - timeBegin)/CLOCKS_PER_SEC;
    if (mode == 'D')
	cputime[0] = elapsedTime;
    if (msg >= QUIC_MSG_TIME)
	MSG("QUIC CPU time: %.3f seconds\n", elapsedTime);
}

extern "C"
void QUICR(char** modeptr, uint32_t& p, const double* S, double* Lambda0,
	   uint32_t& pathLen, const double* path, double& tol,
	   int32_t& msg, uint32_t* iter, double* X,
	   double* W, double* opt, double* cputime)
{
    char mode = **modeptr;
    QUIC(mode, p, S, Lambda0, pathLen, path, tol, msg, iter, X, W,
	 opt, cputime);
}
