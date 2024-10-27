#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "../include/mpls_util.h" 
#include "../include/lapack.h"
#include "../include/my_wtime.h"

void funcAx_gls(int length, int m, int n, int p, double *A, int lda,
        double *B, int ldb, double alpha, double *x, double *y)
{
    double one = 1.0, zero = 0.0;
    int incx = 1;

    // y_1 = alpha*x_1 + B^T*x_2.
    dcopy_(&p, x, &incx, y, &incx);
    dgemv_("T", &n, &p, &one, B, &ldb, &x[p], &incx, &alpha, y, &incx, 1);
    // y_2 = B*x_1+Ax_3.
    dgemv_("N", &n, &p, &one, B, &ldb, x, &incx, &zero, &y[p], &incx, 1);
    dgemv_("N", &n, &m, &one, A, &lda, &x[p+n], &incx, &one, &y[p], &incx, 1);
    // y_3 = A^T*x_2.
    dgemv_("T", &n, &m, &one, A, &lda, &x[p], &incx, &zero, &y[p+n], &incx,
        1);
}

void funcAxres_gls(int length, int m, int n, int p, double *A, int lda,
        double *B, int ldb, double alpha, double *x, double *y)
{
    double one = 1.0, rone = -1.0, ralpha = -alpha;
    int incx = 1;

    // y_1 = y_1 - alpha*x_1 - B^T*x_2.
    daxpy_(&p, &ralpha, x, &incx, y, &incx);
    dgemv_("T", &n, &p, &rone, B, &ldb, &x[p], &incx, &one, y, &incx, 1);
    // y_2 = y_2-B*x_1-Ax_3.
    dgemv_("N", &n, &p, &rone, B, &ldb, x, &incx, &one, &y[p], &incx, 1);
    dgemv_("N", &n, &m, &rone, A, &lda, &x[p+n], &incx, &one, &y[p], &incx, 1);
    // y_3 = y_3-A^T*x_2.
    dgemv_("T", &n, &m, &rone, A, &lda, &x[p], &incx, &one, &y[p+n], &incx,
        1);
}

void funcAx(int length, int m, int n, int p, double *A, int lda, double *B,
        int ldb, double *x, double *y)
{
    double one = 1.0, zero = 0.0;
    int incx = 1;

    // y_1 = x_1 + A*x_3.
    dcopy_(&m, x, &incx, y, &incx);
    dgemv_("N", &m, &n, &one, A, &lda, &x[m+p], &incx, &one, y, &incx, 1);
    // y_3 = A^T*x_1.
    dgemv_("T", &m, &n, &one, A, &lda, x, &incx, &zero, &y[m+p], &incx, 1);
    // y_2 = B*x_3.
    dgemv_("N", &p, &n, &one, B, &ldb, &x[m+p], &incx, &zero, &y[m], &incx,
        1);
    // y_3 = y_3 + B^T*x_2;
    dgemv_("T", &p, &n, &one, B, &ldb, &x[m], &incx, &one, &y[m+p], &incx,
        1);
}

void funcAxres(int length, int m, int n, int p, double *A, int lda,
        double *B, int ldb, double *x, double *y)
{
    double one = 1.0, rone = -1.0;
    int incx = 1;

    // y_1 = y_1 - x_1 - A*x_3.
    daxpy_(&m, &rone, x, &incx, y, &incx);
    dgemv_("N", &m, &n, &rone, A, &lda, &x[m+p], &incx, &one, y, &incx, 1);
    // y_3 = y_3 - A^T*x_1.
    dgemv_("T", &m, &n, &rone, A, &lda, x, &incx, &one, &y[m+p], &incx, 1);
    // y_2 = y_2 - B*x_3.
    dgemv_("N", &p, &n, &rone, B, &ldb, &x[m+p], &incx, &one, &y[m], &incx,
        1);
    // y_3 = y_3 - B^T*x_2;
    dgemv_("T", &p, &n, &rone, B, &ldb, &x[m], &incx, &one, &y[m+p], &incx,
        1);
}

void funcAx_scal(int length, int m, int n, int p, double *A, int lda,
        double *B, int ldb, double alpha, double *x, double *y)
{
    double one = 1.0, zero = 0.0;
    int incx = 1;

    // y_1 = alpha*x_1 + A*x_3.
    dcopy_(&m, x, &incx, y, &incx);
    dgemv_("N", &m, &n, &one, A, &lda, &x[m+p], &incx, &alpha, y, &incx, 1);
    // y_3 = A^T*x_1.
    dgemv_("T", &m, &n, &one, A, &lda, x, &incx, &zero, &y[m+p], &incx, 1);
    // y_2 = B*x_3.
    dgemv_("N", &p, &n, &one, B, &ldb, &x[m+p], &incx, &zero, &y[m],
            &incx, 1);
    // y_3 = y_3 + B^T*x_2;
    dgemv_("T", &p, &n, &one, B, &ldb, &x[m], &incx, &one, &y[m+p],
            &incx, 1);
}

void funcAxres_scal(int length, int m, int n, int p, double *A, int lda,
        double *B, int ldb, double alpha, double *x, double *y)
{
    double one = 1.0, rone = -1.0, ralpha = -alpha;
    int incx = 1;

    // y_1 = y_1 - alpha*x_1 - A*x_3.
    daxpy_(&m, &ralpha, x, &incx, y, &incx);
    dgemv_("N", &m, &n, &rone, A, &lda, &x[m+p], &incx, &one, y, &incx, 1);
    // y_3 = y_3 - A^T*x_1.
    dgemv_("T", &m, &n, &rone, A, &lda, x, &incx, &one, &y[m+p], &incx, 1);
    // y_2 = y_2 - B*x_3.
    dgemv_("N", &p, &n, &rone, B, &ldb, &x[m+p], &incx, &one, &y[m], &incx,
        1);
    // y_3 = y_3 - B^T*x_2;
    dgemv_("T", &p, &n, &rone, B, &ldb, &x[m], &incx, &one, &y[m+p], &incx,
        1);
}

void funcpretwoside_left_gls_n(int m, int n, int p, double *As, int lda,
        double *Bs, int ldb, double *Ts, double alpha, double *workssub,
        double *r, double *work)
{
    // Require p < n.
    int incx = 1, info, np = MIN(p, n), ntemp = n - p, lworkssub = m*n*10;
    int mnp = m - n + p;
    double *r1, *r2, *r3, *r3temp;
    double one = 1.0, zero = 0.0, rone = -1.0;
    double sqrtalpha = sqrt(alpha), dtemp = 1.0/sqrtalpha;

    r1 = r;
    r2 = r1 + p;
    r3 = r2 + n;
    r3temp = work;

    dscal_(&p, &dtemp, r1, &incx);
    dormqr_("L", "T", &n, &incx, &m, As, &lda, workssub, r2, &n,
            &workssub[m+np], &lworkssub, &info, 1, 1);
    dtrtrs_("U", "N", "N", &p, &incx, &Ts[n-p], &n, &r2[n-p], &p, &info,
            1, 1, 1);
    dgemv_("N", &ntemp, &p, &rone, Ts, &n, &r2[n-p], &incx, &one, r2,
            &incx, 1);
    dscal_(&n, &sqrtalpha, r2, &incx);

    dtrtrs_("U", "T", "N", &m, &incx, As, &lda, r3, &m, &info, 1, 1, 1);
    dcopy_(&m, r3, &incx, r3temp, &incx);
    dgemv_("T", &ntemp, &mnp, &one, Ts, &n, r3temp, &incx, &zero, &r3[n-p],
            &incx, 1);
    dgemv_("T", &mnp, &mnp, &one, &Ts[n-p], &n, &r3temp[n-p], &incx, &one,
            &r3[n-p], &incx, 1);
    dscal_(&m, &dtemp, r3, &incx);

}

void funcpretwoside_right_gls_n(int m, int n, int p, double *As, int lda,
        double *Bs, int ldb, double *Ts, double alpha, double *workssub,
        double *r, double *work)
{
    // Require p < n.
    int incx = 1, info, np = MIN(p, n), ntemp = n - p, lworkssub = m*n*10;
    int mnp = m - n + p;
    double *r1, *r2, *r3, *r3temp;
    double one = 1.0, zero = 0.0, rone = -1.0;
    double sqrtalpha = sqrt(alpha), dtemp = 1.0/sqrtalpha;

    r1 = r;
    r2 = r1 + p;
    r3 = r2 + n;
    r3temp = work;

    dscal_(&p, &dtemp, r1, &incx);
    dcopy_(&n, r2, &incx, r3temp, &incx);
    dgemv_("T", &ntemp, &p, &rone, Ts, &n, r2, &incx, &zero, &r2[n-p],
            &incx, 1);
    dtrtrs_("U", "T", "N", &p, &incx, &Ts[n-p], &n, &r2[n-p], &p, &info,
            1, 1, 1);
    dtrtrs_("U", "T", "N", &p, &incx, &Ts[n-p], &n, &r3temp[n-p], &p, &info,
            1, 1, 1);
    daxpy_(&p, &one, &r3temp[n-p], &incx, &r2[n-p], &incx);
    dormqr_("L", "N", &n, &incx, &m, As, &lda, workssub, r2, &n,
            &workssub[m+np], &lworkssub, &info, 1, 1);
    dscal_(&n, &sqrtalpha, r2, &incx);

    dcopy_(&m, r3, &incx, r3temp, &incx);
    dgemv_("N", &ntemp, &mnp, &one, Ts, &n, &r3temp[n-p], &incx, &one,
        r3, &incx, 1);
    dgemv_("N", &mnp, &mnp, &one, &Ts[n-p], &n, &r3temp[n-p], &incx, &zero,
        &r3[n-p], &incx, 1);
    dtrtrs_("U", "N", "N", &m, &incx, As, &lda, r3, &m, &info, 1, 1, 1);
    dscal_(&m, &dtemp, r3, &incx);

}

void funcpretwoside_left_gls_p(int m, int n, int p, double *As, int lda,
        double *Bs, int ldb, double *Ts, double alpha, double *workssub,
        double *r, double *work)
{
    // Require p >= n.
    int incx = 1, info, np = MIN(p, n), lworkssub = m*n*10;
    double *r1, *r2, *r3, *r3temp;
    double one = 1.0, zero = 0.0;
    double sqrtalpha = sqrt(alpha), dtemp = 1.0/sqrtalpha;

    r1 = r;
    r2 = r1 + p;
    r3 = r2 + n;
    r3temp = work;

    dscal_(&p, &dtemp, r1, &incx);
    dormqr_("L", "T", &n, &incx, &m, As, &lda, workssub, r2, &n,
            &workssub[m+np], &lworkssub, &info, 1, 1);
    dtrtrs_("U", "N", "N", &n, &incx, &Ts[(p-n)*n], &n, r2, &n, &info,
            1, 1, 1);
    dscal_(&n, &sqrtalpha, r2, &incx);

    dcopy_(&m, r3, &incx, r3temp, &incx);
    dtrtrs_("U", "T", "N", &m, &incx, As, &lda, r3temp, &m, &info, 1, 1, 1);
    dgemv_("T", &m, &m, &one, &Ts[(p-n)*n], &n, r3temp, &incx, &zero, r3,
            &incx, 1);
    dscal_(&m, &dtemp, r3, &incx);

}

void funcpretwoside_right_gls_p(int m, int n, int p, double *As, int lda,
        double *Bs, int ldb, double *Ts, double alpha, double *workssub,
        double *r, double *work)
{
    // Require p >= n.
    int incx = 1, info, np = MIN(p, n), lworkssub = m*n*10;
    double *r1, *r2, *r3, *r3temp;
    double one = 1.0, zero = 0.0;
    double sqrtalpha = sqrt(alpha), dtemp = 1.0/sqrtalpha;

    r1 = r;
    r2 = r1 + p;
    r3 = r2 + n;
    r3temp = work;

    dscal_(&p, &dtemp, r1, &incx);
    dtrtrs_("U", "T", "N", &n, &incx, &Ts[(p-n)*n], &n, r2, &n, &info,
            1, 1, 1);
    dormqr_("L", "N", &n, &incx, &m, As, &lda, workssub, r2, &n,
            &workssub[m+np], &lworkssub, &info, 1, 1);
    dscal_(&n, &sqrtalpha, r2, &incx);

    dcopy_(&m, r3, &incx, r3temp, &incx);
    dgemv_("N", &m, &m, &one, &Ts[(p-n)*n], &n, r3temp, &incx, &zero, r3,
            &incx, 1);
    dtrtrs_("U", "N", "N", &m, &incx, As, &lda, r3, &m, &info, 1, 1, 1);
    dscal_(&m, &dtemp, r3, &incx);

}


void funcpretwoside_left_lse_scal(int m, int n, int p, double *As, int lda,
    double *Bs, int ldb, double *T2s, double alpha, double *workssub,
    double *r, double *work)
{
    // Require m >= n.
    int incx = 1, info, mn = MIN(m, n), ntemp = n - p, lworkssub = m*n*10;
    double *r1, *r2, *r3, *r2temp;
    double one = 1.0, zero = 0.0, sqrtalpha = sqrt(alpha), dtemp = 1.0/sqrtalpha;

    r1 = r;
    r2 = r1 + m;
    r3 = r2 + p;
    r2temp = work;

    dscal_(&m, &dtemp, r1, &incx);
    dcopy_(&p, r2, &incx, r2temp, &incx);
    dtrtrs_("U", "N", "N", &p, &incx, &Bs[(n-p)*p], &p, r2temp, &p, &info,
            1, 1, 1);
    dgemv_("N", &p, &p, &one, &T2s[ntemp], &n, r2temp, &incx, &zero,
            r2, &incx, 1);
    dscal_(&p, &dtemp, r2, &incx);

    dormrq_("L", "N", &n, &incx, &p, Bs, &p, workssub, r3, &n,
            &workssub[p+mn], &lworkssub, &info, 1, 1);
    dtrtrs_("U", "T", "N", &n, &incx, As, &m, r3, &n, &info, 1, 1, 1);
    dscal_(&n, &sqrtalpha, r3, &incx);
}

void funcpretwoside_right_lse_scal(int m, int n, int p, double *As, int lda,
    double *Bs, int ldb, double *T2s, double alpha, double *workssub,
    double *r, double *work)
{
    // Require m >= n.
    int incx = 1, info, mn = MIN(m, n), ntemp = n - p, lworkssub = m*n*10;
    double *r1, *r2, *r3, *r2temp;
    double one = 1.0, zero = 0.0, sqrtalpha = sqrt(alpha), dtemp = 1.0/sqrtalpha;

    r1 = r;
    r2 = r1 + m;
    r3 = r2 + p;
    r2temp = work;

    dscal_(&m, &dtemp, r1, &incx);
    dcopy_(&p, r2, &incx, r2temp, &incx);
    dgemv_("T", &p, &p, &one, &T2s[ntemp], &n, r2temp, &incx, &zero,
            r2, &incx, 1);
    dtrtrs_("U", "T", "N", &p, &incx, &Bs[(n-p)*p], &p, r2, &p, &info,
            1, 1, 1);
    dscal_(&p, &dtemp, r2, &incx);

    dtrtrs_("U", "N", "N", &n, &incx, As, &m, r3, &n, &info, 1, 1, 1);
    dormrq_("L", "T", &n, &incx, &p, Bs, &p, workssub, r3, &n,
            &workssub[p+mn], &lworkssub, &info, 1, 1);
    dscal_(&n, &sqrtalpha, r3, &incx);
}



int funcpreleft_lse_scal(int m, int n, int p, double *As, int lda,
    double *Bs, int ldb, double *T2s, double alpha, double *workssub,
    double *r, double *work)
{
    int incx = 1, info, mn = MIN(m, n), ntemp = n - p, lworkssub = m*n*10;
    double *r1, *r2, *r3, *r1temp, *r2temp, *r3temp;
    double one = 1.0, rone = -1.0, invalpha = 1.0/alpha;

    r1 = r;
    r2 = r1 + m;
    r3 = r2 + p;
    r1temp = work;
    r2temp = r1temp + m;
    r3temp = r2temp + p;

    dtrtrs_("U", "N", "N", &p, &incx, &Bs[(n-p)*p], &p, r2, &p, &info,
            1, 1, 1);
    dcopy_(&p, r2, &incx, &r3temp[n-p], &incx);//y_2 stores in r2temp.

    dormrq_("L", "N", &n, &incx, &p, Bs, &p, workssub, r3, &n,
            &workssub[p+mn], &lworkssub, &info, 1, 1);
    dtrtrs_("U", "T", "N", &ntemp, &incx, As, &m, r3, &ntemp, &info,
            1, 1, 1);// q_1 stores in r3.
    dormqr_("L", "T", &m, &incx, &mn, As, &m, &workssub[p], r1, &m,
            &workssub[p+mn], &lworkssub, &info, 1, 1);// q_3 stores in &r1[n].
    dgemv_("N", &p, &p, &rone, &T2s[ntemp], &n, &r3temp[n-p],
            &incx, &one, &r1[n-p], &incx, 1);// q_2 stores in &r1[n-p].
    daxpy_(&ntemp, &rone, r3, &incx, r1, &incx);
    dgemv_("N", &ntemp, &p, &rone, &As[ntemp*m], &m, &r3temp[n-p], &incx,
            &one, r1, &incx, 1); // r1 is -q_1-T_12*y_2+(Z^Tf)_1.
    dtrtrs_("U", "N", "N", &ntemp, &incx, As, &m, r1, &ntemp, &info,
            1, 1, 1);
    dcopy_(&ntemp, r1, &incx, r3temp, &incx);//y stores in r1temp.
    dormrq_("L", "T", &n, &incx, &p, Bs, &p, workssub, r3temp, &n,
            &workssub[p+mn], &lworkssub, &info, 1, 1);// Delta x stores in r1temp.
    dcopy_(&ntemp, r3, &incx, r1, &incx);//q stores in r1.
    dgemv_("T", &n, &p, &one, T2s, &n, r1, &incx, &rone, &r3[n-p],
            &incx, 1);
    dtrtrs_("U", "T", "N", &p, &incx, &Bs[(n-p)*p], &p, &r3[n-p], &p,
            &info, 1, 1, 1);// Delta lambda stores in &r3s[n-p].
    dormqr_("L", "N", &m, &incx, &mn, As, &m, &workssub[p], r1, &m,
            &workssub[p+mn], &lworkssub, &info, 1, 1);// Delta r stores in r1.
    dscal_(&m, &invalpha, r1, &incx);
    dcopy_(&p, &r3[n-p], &incx, r2, &incx);
    dscal_(&p, &rone, r2, &incx);
    dcopy_(&n, r3temp, &incx, r3, &incx);

    return 0;

}

void funcpreright(int n, double *x, double *y)
{
    int incx = 1;
    dcopy_(&n, x, &incx, y, &incx);
}

void rot_givens(double a, double b, double *c, double *s)
{
    double temp;

    if (b == 0.0)
    {
        *c = 1.0;
        *s = 0.0;
    }
    else if (fabs(b) > fabs(a))
    {
        temp = a/b;
        *s = 1.0/sqrt(1.0+temp*temp);
        *c = temp*(*s);
    }
    else
    {
        temp = b/a;
        *c = 1.0/sqrt(1.0+temp*temp);
        *s = temp*(*c);
    }
}

int check_accuracy(int m, int n, int nrhs, double *A, int lda, double *B,
        int ldb, double *X, int ldx)
{
    double one = 1.0, rone = -1.0;
    int length = m*nrhs, lengthA = m*n, lengthX = n*nrhs, incx = 1;
    double *Bcopy = (double *)calloc(m*nrhs, sizeof(double));

    dlacpy_("A", &m, &nrhs, B, &ldb, Bcopy, &m, 1);
    dgemm_("N", "N", &m, &nrhs, &n, &one, A, &lda, X, &ldx, &rone, Bcopy, &m,
            1, 1);
    printf("%%The norm of AX-B: %.20f\n", dnrm2_(&length, Bcopy, &incx));
    printf("%%||AX-B||/(||A||*||X|| + ||B||): %.20f\n", dnrm2_(&length, Bcopy, &incx)/(dnrm2_(&lengthA, A, &incx)*dnrm2_(&lengthX, X, &incx) - dnrm2_(&length, B, &incx)));

    free(Bcopy);
    return 0;
}

int check_accuracy_gls(int n, int m, int p, double *A, int lda, double *B,
        int ldb, double *D, double *X, double *Y, double *work)
{
    double one = 1.0, rone = -1.0;
    int incx = 1, lengthA = n*m, lengthB = n*p;
    double *Dcopy;
    Dcopy = work;

    dcopy_(&n, D, &incx, Dcopy, &incx);
    dgemv_("N", &n, &m, &rone, A, &lda, X, &incx, &one, Dcopy, &incx, 1);
    dgemv_("N", &n, &p, &rone, B, &ldb, Y, &incx, &one, Dcopy, &incx, 1);
    printf("%%The norm of AX+BY-D: %.20f\n", dnrm2_(&n, Dcopy, &incx));
    printf("%%||AX+BY-D||/(||A||*||X|| + ||B||*||Y|| + ||D||): %.20f\n", dnrm2_(&n, Dcopy, &incx)/(dnrm2_(&lengthA, A, &incx)*dnrm2_(&m, X, &incx) + dnrm2_(&lengthB, B, &incx)*dnrm2_(&p, Y, &incx) + dnrm2_(&n, D, &incx)));
    printf("%%The norm of Y: %.20f\n", dnrm2_(&p, Y, &incx));

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

void dgerqf_ql_(int *n, int *p, double *A, int *lda, double *tau,
    double *work, int *lwork, int *info)
{
    int i, incx = 1, ntemp = *n, ptemp = *p;
    double *Acopy, *worksub;

    Acopy = work;
    worksub = Acopy + ntemp*ptemp;

    for (i = 0; i < ntemp; i++)
    dcopy_(p, &A[i], lda, &Acopy[i*ptemp], &incx);
    dgeqlf_(p, n, Acopy, p, tau, worksub, lwork, info);
    for (i = 0; i < *n; i++)
    dcopy_(p, &Acopy[i*ptemp], &incx, &A[i], lda);

}

void sgerqf_ql_(int *n, int *p, float *A, int *lda, float *tau,
    float *work, int *lwork, int *info)
{
    int i, incx = 1, ntemp = *n, ptemp = *p;
    float *Acopy, *worksub;

    Acopy = work;
    worksub = Acopy + ntemp*ptemp;

    for (i = 0; i < ntemp; i++)
        scopy_(p, &A[i], lda, &Acopy[i*ptemp], &incx);
    sgeqlf_(p, n, Acopy, p, tau, worksub, lwork, info);
    for (i = 0; i < *n; i++)
        scopy_(p, &Acopy[i*ptemp], &incx, &A[i], lda);

}

void gen_mat(int m, int n, double cond, double *A, double *work, int lwork)
{
    int incx = 1, info, idist, irsigm, mode;
    double *sva, *tau, *U, *V, *Utemp, *Vtemp, *worksub;
    int *iseed = (int *)calloc(4, sizeof(int));
    double one = 1.0, zero = 0.0;

    sva = work;
    tau = sva + m;
    Utemp = tau + m;
    U = Utemp + m*n;
    Vtemp = U + m*n;
    V = Vtemp + n*n;
    worksub = V + n*n;

    // Generate testing matrix.
    srand(0);
    irsigm = 0;
    idist = 3;
    mode = 3;
    iseed[0] = rand()%1000;
    iseed[1] = rand()%1000;
    iseed[2] = rand()%1000;
    iseed[3] = rand()%1000;
    dlatm1_(&mode, &cond, &irsigm, &idist, iseed, sva, &n, &info);

    for (int i = 0; i < m*n; i++)
        Utemp[i] = rand()%100;
    for (int i = 0; i < n*n; i++)
        Vtemp[i] = rand()%100;
    for (int i = 0; i < n; i++)
    {
        U[i*m + i] = 1.0;
        V[i*n + i] = 1.0;
    }
    dgeqrf_(&m, &n, Utemp, &m, tau, worksub, &lwork, &info);
    dormqr_("L", "N", &m, &n, &n, Utemp, &m, tau, U, &m, worksub, &lwork,
            &info, 1, 1);
    dgeqrf_(&n, &n, Vtemp, &n, tau, worksub, &lwork, &info);
    dormqr_("L", "N", &n, &n, &n, Vtemp, &n, tau, V, &n, worksub, &lwork,
            &info, 1, 1);
    for (int i = 0; i < n; i++)
        dscal_(&m, &sva[i], &U[i*m], &incx);
    dgemm_("N", "N", &m, &n, &n, &one, U, &m, V, &n, &zero, A, &m, 1, 1);

    free(iseed);
}
