#include <stdio.h>
#include <stdlib.h>
#include "../include/lapack.h"
#include "../include/mpls_util.h"
#include "../include/mpls.h"
#include "../include/my_wtime.h"

int test_gls(int n, int m, int p, double cond, double *timesum)
{
    printf("%% Test matrix with n = %d, m = %d, p = %d\n", n, m, p);
    int incx = 1, restart = MIN(300, m+n+p), info, mtemp;
    int lwork = n*m + 2*n*p + 4*n + 4*m + 3*p + MAX(n, p)*256 + 9*(m+n+p) + (m+n+p)*restart + 6*restart + restart*restart + 5*n*p;
    int lworks = n*m + 2*n*p + 4*n + 3*m + 2*p + MAX(n, p)*256 + 5*n*p;
    double *AB = (double *)calloc((m+p)*n, sizeof(double));
    double *x = (double *)calloc(m, sizeof(double));
    double *y = (double *)calloc(p, sizeof(double));
    double *d = (double *)calloc(n, sizeof(double));
    double *Acopy = (double *)calloc(m*n, sizeof(double));
    double *Bcopy = (double *)calloc(p*n, sizeof(double));
    double *dcopy = (double *)calloc(n, sizeof(double));
    double *work = (double *)calloc(lwork, sizeof(double));
    double *sva = (double *)calloc(n, sizeof(double));
    float *works = (float *)calloc(lworks, sizeof(float));
    int *iseed = (int *)calloc(4, sizeof(int));
    double start, over;

    // Generate testing matrix.
    srand(0);
    mtemp = m+p;
    gen_mat(mtemp, n, cond, work, work+mtemp*n, lwork);
    for (int i = 0; i < mtemp; i++)
        for (int j = 0; j < n; j++)
            AB[i*n + j] = work[j*mtemp + i];
    for (int i = 0; i < n; i++)
        d[i] = 1.0;

    // Test the mixed precision generalized LS algorithm by using traditional
    // iterative refinement.
    dlacpy_("A", &n, &m, AB, &n, Acopy, &n, 1);
    dlacpy_("A", &n, &p, &AB[m*n], &n, Bcopy, &n, 1);
    dcopy_(&n, d, &incx, dcopy, &incx);
    start = tic();
    mpgls(n, m, p, Acopy, n, Bcopy, n, dcopy, x, y, works,
            lworks, work, lwork);
    over = tic();
    double timemp = over-start;
    printf("%%Time of mpgls: %.6f\n", over-start);
    for (int i = 0; i < 5; i++)
        timesum[i] = work[i];
    // Check accuracy.
    check_accuracy_gls(n, m, p, AB, n, &AB[m*n], n, d, x, y, work);

    // Test the mixed precision generalized LS algorithm by using GMRES-based
    // iterative refinement with two-side preconditioning.
    dlacpy_("A", &n, &m, AB, &n, Acopy, &n, 1);
    dlacpy_("A", &n, &p, &AB[m*n], &n, Bcopy, &n, 1);
    dcopy_(&n, d, &incx, dcopy, &incx);
    start = tic();
    mpgls_gmres_twoside(n, m, p, Acopy, n, Bcopy, n, dcopy, x, y, works,
        lworks, work, lwork);
    over = tic();
    double timemp_gmres = over-start;
    printf("%%Time of mpgls_gmres_twoside: %.6f\n", over-start);
    for (int i = 0; i < 4; i++)
        timesum[5+i] = work[i];
    // Check accuracy.
    check_accuracy_gls(n, m, p, AB, n, &AB[m*n], n, d, x, y, work);

    // Test the fixed precision (double precision) LAPACK subroutine.
    dlacpy_("A", &n, &m, AB, &n, Acopy, &n, 1);
    dlacpy_("A", &n, &p, &AB[m*n], &n, Bcopy, &n, 1);
    dcopy_(&n, d, &incx, dcopy, &incx);
    start = tic();
    dggglm_(&n, &m, &p, Acopy, &n, Bcopy, &n, dcopy, x, y, work,
           &lwork, &info);
    over = tic();
    double timed = over-start;
    timesum[9] = timed;
    printf("%% Time of dggglm: %.6f\n", over-start);
    // Check accuracy.
    check_accuracy_gls(n, m, p, AB, n, &AB[m*n], n, d, x, y, work);
    printf("%% ratio: %.4f\n", (timed-timemp)/timed);
    printf("%% ratio_gmres: %.4f\n", (timed-timemp_gmres)/timed);
    dprintmat("timesubsum", 10, 1, timesum, 10);

    free(AB);
    free(y);
    free(d);
    free(x);
    free(dcopy);
    free(Acopy);
    free(Bcopy);
    free(work);
    free(sva);
    free(works);
    free(iseed);
    return 0;
}

int test(int times_mn, int times_pm, int m0, double cond, double *time)
{
    int num = 3;
    int m = m0, n = 1024, p;

    for (int i = 0; i < num; i++)
    {
        m = m + 1024; n = m*times_mn; p = m*times_pm;
        //p = p*2; n = p*times_np; m = n*times_mn;
        //n = n + 1024; p = n*times_mn; m = n/times_pm;
        test_gls(n, m, p, cond, time+10*i);
    }
    dprintmat("time", 10, 3, time, 10);
    return 0;
}

int test_nlarge(int times_mn, int times_pm, int m0, double cond, double *time)
{
    int num = 3;
    int m = m0, n, p;

    for (int i = 0; i < num; i++)
    {
        m = m + 1024; n = m*times_mn; p = n - m + 128*(i+1);
        test_gls(n, m, p, cond, time+10*i);
    }
    dprintmat("time", 10, 3, time, 10);
    return 0;
}

int test_all(double cond)
{
    double *time = (double *)calloc(10*3*6, sizeof(double));

    test(8, 32,  0, cond, time);
    test(8, 64,  0, cond, time+10*3);
    test(10, 32,  0, cond, time+10*3*2);
    test(10, 64,  0, cond, time+10*3*3);
    test(12, 32, 0, cond, time+10*3*4);
    test(12, 64, 0, cond, time+10*3*5);
    //test_nlarge(4, 3, 0, cond, time+10*3*3);
    //test_nlarge(5, 4, 0, cond, time+10*3*4);
    //test_nlarge(6, 5, 0, cond, time+10*3*5);

    dprintmat("timeall", 10, 18, time, 10);

    free(time);
    return 0;
}

int main (int argc, char **argv)
{
    // m <= n <= m+p
    double *time = (double *)calloc(6*4*6, sizeof(double));
    int m = 1024;
    // Warm up.
    test_gls(m*2, m, m*4, 1e5, time);
    free(time);

    // Start tests.
    test_all(1e3);

    return 0;
}
