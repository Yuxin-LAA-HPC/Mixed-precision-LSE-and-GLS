#ifndef _MPLS_UTIL_H
#define _MPLS_UTIL_H

#include "mpls_types.h"

#ifdef __cplusplus
extern "C" {
#endif

void funcAx(int length, int m, int n, int p, double *A, int lda, double *B,
        int ldb, double *x, double *y);
void funcAxres(int length, int m, int n, int p, double *A, int lda,
        double *B, int ldb, double *x, double *y);
void funcAx_scal(int length, int m, int n, int p, double *A, int lda,
        double *B, int ldb, double alpha, double *x, double *y);
void funcAxres_scal(int length, int m, int n, int p, double *A, int lda,
        double *B, int ldb, double alpha, double *x, double *y);
void funcpretwoside_left(int m, int n, int p, double *As, int lda,
        double *Bs, int ldb, double *T2s, double *workssub, double *r,
        double *work);
void funcpretwoside_right(int m, int n, int p, double *As, int lda,
        double *Bs, int ldb, double *T2s, double *workssub, double *r,
        double *work);
void funcpretwoside_left_lse_scal(int m, int n, int p, double *As,
        int lda, double *Bs, int ldb, double *T2s, double alpha,
        double *workssub, double *r, double *work);
void funcpretwoside_right_lse_scal(int m, int n, int p, double *As,
        int lda, double *Bs, int ldb, double *T2s, double alpha,
        double *workssub, double *r, double *work);
int funcpreleft_lse_scal(int m, int n, int p, double *As, int lda,
        double *Bs, int ldb, double *T2s, double alpha, double *workssub,
        double *r, double *work);
void funcpreright(int n, double *x, double *y);
void rot_givens(double a, double b, double *c, double *s);

void funcAx_gls(int length, int m, int n, int p, double *A, int lda,
        double *B, int ldb, double alpha, double *x, double *y);
void funcAxres_gls(int length, int m, int n, int p, double *A, int lda,
        double *B, int ldb, double alpha, double *x, double *y);
void funcpretwoside_left_gls_p(int m, int n, int p, double *As, int lda,
        double *Bs, int ldb, double *Ts, double alpha, double *workssub,
        double *r, double *work);
void funcpretwoside_right_gls_p(int m, int n, int p, double *As, int lda,
        double *Bs, int ldb, double *T2s, double alpha, double *workssub,
        double *r, double *work);
void funcpretwoside_left_gls_n(int m, int n, int p, double *As, int lda,
        double *Bs, int ldb, double *Ts, double alpha, double *workssub,
        double *r, double *work);
void funcpretwoside_right_gls_n(int m, int n, int p, double *As, int lda,
        double *Bs, int ldb, double *T2s, double alpha, double *workssub,
        double *r, double *work);

void dgerqf_ql_(int *n, int *p, double *A, int *lda, double *tau,
        double *work, int *lwork, int *info);
void sgerqf_ql_(int *n, int *p, float *A, int *lda, float *tau,
        float *work, int *lwork, int *info);

void gen_mat(int m, int n, double cond, double *A, double *work, int lwork);

int check_accuracy(int m, int n, int nrhs, double *A, int lda, double *B,
        int ldb, double *X, int ldx);
int check_accuracy_gls(int n, int m, int p, double *A, int lda, double *B,
        int ldb, double *D, double *X, double *Y, double *work);
void sprintmat(char *name, int nrows, int ncols, float A[], int ldA);
void dprintmat(char *name, int nrows, int ncols, double A[], int ldA);
void cprintmat(char *name, int nrows, int ncols, cplx A[], int ldA);
void zprintmat(char *name, int nrows, int ncols, zplx A[], int ldA);
void iprintmat(char *name, int nrows, int ncols, int A[], int ldA);

#ifdef __cplusplus
}
#endif

#endif
