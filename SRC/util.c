#include <stdio.h>
#include <stdlib.h>
#include "../include/mpls_util.h"
#include "../include/lapack.h"

int check_accuracy(int m, int n, int nrhs, double *A, int lda, double *B,
        int ldb, double *X, int ldx)
{
    double one = 1.0, rone = -1.0;
    int length = m*nrhs, incx = 1;
    double *Bcopy = (double *)calloc(m*nrhs, sizeof(double));

    dlacpy_("A", &m, &nrhs, B, &ldb, Bcopy, &m, 1);
    dgemm_("N", "N", &m, &nrhs, &n, &one, A, &lda, X, &ldx, &rone, Bcopy, &m,
            1, 1);
    printf("%%The norm of AX-B: %.16f\n", dnrm2_(&length, Bcopy, &incx));

    free(Bcopy);
    return 0;
}

int check_accuracy_gls(int n, int m, int p, double *A, int lda, double *B,
        int ldb, double *D, double *X, double *Y)
{
    double one = 1.0, rone = -1.0;
    int incx = 1;
    double *Dcopy = (double *)calloc(n, sizeof(double));

    dcopy_(&n, D, &incx, Dcopy, &incx);
    dgemv_("N", &n, &m, &rone, A, &lda, X, &incx, &one, Dcopy, &incx, 1);
    dgemv_("N", &n, &p, &rone, B, &ldb, Y, &incx, &one, Dcopy, &incx, 1);
    printf("%%The norm of AX+BY-D: %.16f\n", dnrm2_(&n, Dcopy, &incx));
    printf("%%The norm of Y: %.16f\n", dnrm2_(&p, Y, &incx));

    free(Dcopy);
    return 0;
}

void sprintmat(char *name, int nrows, int ncols, float A[], int ldA)
{
    int i, j;

    printf("%s = zeros(%d, %d);\n", name, nrows, ncols);
    for (j = 0; j < ncols; j++)
    for (i = 0; i < nrows; i++)
        printf("%s(%4d, %4d) = %.16e;\n", name, i+1, j+1, A[i + j*ldA]);
    fflush(stdout);
}

void dprintmat(char *name, int nrows, int ncols, double A[], int ldA)
{
    int i, j;

    printf("%s = zeros(%d, %d);\n", name, nrows, ncols);
    for (j = 0; j < ncols; j++)
    for (i = 0; i < nrows; i++)
        printf("%s(%4d, %4d) = %.16e;\n", name, i+1, j+1, A[i + j*ldA]);
    fflush(stdout);
}

void cprintmat(char *name, int nrows, int ncols, cplx A[], int ldA)
{
    int i, j;

    printf("%s = zeros(%d, %d);\n", name, nrows, ncols);
    for (j = 0; j < ncols; j++)
    for (i = 0; i < nrows; i++)
        printf("%s(%4d, %4d) = %.16e + %.16ei;\n", name, i+1, j+1,
                creal(A[i + j*ldA]), cimag(A[i + j*ldA]));
    fflush(stdout);
}

void zprintmat(char *name, int nrows, int ncols, zplx A[], int ldA)
{
    int i, j;

    printf("%s = zeros(%d, %d);\n", name, nrows, ncols);
    for (j = 0; j < ncols; j++)
    for (i = 0; i < nrows; i++)
        printf("%s(%4d, %4d) = %.16e + %.16ei;\n", name, i+1, j+1,
                creal(A[i + j*ldA]), cimag(A[i + j*ldA]));
    fflush(stdout);
}

void iprintmat(char *name, int nrows, int ncols, int A[], int ldA)
{
    int i, j;

    printf("%s = zeros(%d, %d);\n", name, nrows, ncols);
    for (j = 0; j < ncols; j++)
    for (i = 0; i < nrows; i++)
        printf("%s(%4d, %4d) = %4d;\n", name, i+1, j+1, A[i + j*ldA]);
    fflush(stdout);
}

void compute_orthbasis(int m, int n, int p, float *As, float *Bs,
        float *workssub, int lworkssub)
{
    float *Q = (float*)calloc(n*n, sizeof(float));
    float *Z = (float*)calloc(m*m, sizeof(float));
    float *T = (float*)calloc(m*n, sizeof(float));
    float *R = (float*)calloc(p*p, sizeof(float));
    float zerof = 0.0f, onef = 1.0f;
    int info;

    slaset_("A", &n, &n, &zerof, &onef, Q, &n, 1);
    slaset_("A", &m, &m, &zerof, &onef, Z, &m, 1);
    sormrq_("L", "N", &n, &n, &p, Bs, &p, workssub, Q, &n,
            &workssub[p+n], &lworkssub, &info, 1, 1);
    sormqr_("L", "N", &m, &m, &n, As, &m, &workssub[p], Z, &m,
            &workssub[p+n], &lworkssub, &info, 1, 1);
    slacpy_("U", &n, &n, As, &m, T, &m, 1);
    slacpy_("U", &p, &p, &Bs[(n-p)*p], &p, R, &p, 1);

    sprintmat("Q", n, n, Q, n);
    sprintmat("Z", m, m, Z, m);
    sprintmat("T", m, n, T, m);
    sprintmat("R", p, p, R, p);

    free(Q);
    free(Z);
    free(T);
    free(R);
}
