#ifndef _MPLS_H
#define _MPLS_H

#include "mpls_types.h"
//#include "mpls_util.h"
#include "lapack.h"
#include <math.h>

int mplse(int m, int n, int p, double *A, int lda, double *B, int ldb,
        double *c, double *d, double *x, double *work, float *works,
        int lwork);

int mpgls(int n, int m, int p, double *A, int lda, double *B, int ldb,
        double *d, double *x, double *y, float *works, int lworks,
        double *work, int lwork);

int gmres_lse_left_scal(int n, int mo, int no, int po, double *Ao, int lda,
        double *Bo, int ldb, double *As, int ldas, double *Bs, int ldbs,
        double *T2s, double alpha, double *workssub,
        double *b, double *x, int restrt, int maxiter, int *num_iter,
        double tol, double *work, int lwork);
int mplse_gmres_scal(int m, int n, int p, double *A, int lda, double *B,
        int ldb, double *c, double *d, double *x, double *work,
        float *works, int lwork);


int gmres_lse_twoside_scal(int n, int mo, int no, int po, double *Ao, int lda,
        double *Bo, int ldb, double *As, int ldas, double *Bs, int ldbs,
        double *T2s, double alpha, double *workssub,
        double *b, double *x, int restrt, int maxiter, int *num_iter,
        double tol, double *work, int lwork);
int mplse_gmres_twoside_scal(int m, int n, int p, double *A, int lda,
        double *B, int ldb, double *c, double *d, double *x, double *work,
        float *works, int lwork);

int gmres_gls_twoside_plarge(int n, int mo, int no, int po, double *Ao,
	int lda, double *Bo, int ldb, double *As, int ldas, double *Bs,
	int ldbs, double *T2s, double alpha, double *workssub, double *b,
	double *x, int restrt, int maxiter, int *num_iter, double tol,
	double *work, int lwork);
int gmres_gls_twoside_nlarge(int n, int mo, int no, int po, double *Ao,
	int lda, double *Bo, int ldb, double *As, int ldas, double *Bs,
	int ldbs, double *T2s, double alpha, double *workssub, double *b,
	double *x, int restrt, int maxiter, int *num_iter, double tol,
	double *work, int lwork);
int mpgls_gmres_twoside(int n, int m, int p, double *A, int lda, double *B,
        int ldb, double *d, double *x, double *y, float *works, int lworks,
        double *work, int lwork);
#endif
