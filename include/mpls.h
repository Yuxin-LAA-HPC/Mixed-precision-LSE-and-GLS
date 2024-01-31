#ifndef _MPLS_H
#define _MPLS_H

#include "mpls_types.h"
//#include "mpls_util.h"
#include "lapack.h"
#include <math.h>

int mplsy(int m, int n, int nrhs, double *A, int lda, double *B, int ldb,
        double rcond, double *work, float *works, int lwork);

int mplse(int m, int n, int p, double *A, int lda, double *B, int ldb,
        double *c, double *d, double *x, double *work, float *works,
        int lwork);
int mplse_2block(int m, int n, int p, double *A, int lda, double *B,
        int ldb, double *c, double *d, double *x, double *work,
        float *works, int lwork);
int mplse_2block_totalh(int m, int n, int p, double *A, int lda, double *B,
        int ldb, double *c, double *d, double *x, double *work,
        float *works, int lwork);

int mpgls(int n, int m, int p, double *A, int lda, double *B, int ldb,
        double *d, double *x, double *y, float *works, int lworks,
        double *work, int lwork);

#endif
