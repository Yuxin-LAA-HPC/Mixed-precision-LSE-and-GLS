#ifndef _MPLS_UTIL_H
#define _MPLS_UTIL_H

#include "mpls_types.h"

#ifdef __cplusplus
extern "C" {
#endif

int check_accuracy(int m, int n, int nrhs, double *A, int lda, double *B,
        int ldb, double *X, int ldx);
int check_accuracy_gls(int n, int m, int p, double *A, int lda, double *B,
        int ldb, double *D, double *X, double *Y);
void sprintmat(char *name, int nrows, int ncols, float A[], int ldA);
void dprintmat(char *name, int nrows, int ncols, double A[], int ldA);
void cprintmat(char *name, int nrows, int ncols, cplx A[], int ldA);
void zprintmat(char *name, int nrows, int ncols, zplx A[], int ldA);
void iprintmat(char *name, int nrows, int ncols, int A[], int ldA);

#ifdef __cplusplus
}
#endif

#endif
