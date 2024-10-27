#include <stdio.h>
#include <stdlib.h>
#include "../include/lapack.h"
#include "../include/mpls_util.h"
#include "../include/mpls.h"
#include "../include/my_wtime.h"

int test_lse(int m, int n, int p, double cond_AB, double *timeout)
{
    int incx = 1, info, mtemp;
    int lworks = 2*m*n+2*p*n+4*m+3*n+2*p+2*m*n;
    int lwork = m*n+m+p+n*n + m*n+n*p*2+2*m+2*n+2*p+m*n*14;
    double *AB = (double *)calloc((m+p)*n, sizeof(double));
    double *c = (double *)calloc(m, sizeof(double));
    double *d = (double *)calloc(p, sizeof(double));
    double *x = (double *)calloc(n, sizeof(double));
    double *Acopy = (double *)calloc(m*n, sizeof(double));
    double *Bcopy = (double *)calloc(p*n, sizeof(double));
    double *ccopy = (double *)calloc(m, sizeof(double));
    double *dcopy = (double *)calloc(p, sizeof(double));
    double *work = (double *)calloc(lwork, sizeof(double));
    float *works = (float *)calloc(lworks, sizeof(float));
    double start, over, time1, time2, time3, time4;

    // Generate testing matrix.
    srand(0);
    mtemp = m+p;
    gen_mat(mtemp, n, cond_AB, AB, work, lwork);
    for (int i = 0; i < m; i++)
        c[i] = 1.0;
    for (int i = 0; i < p; i++)
        d[i] = 1.0;

    // Test the mixed precision LSE algorithm by using GMRES-based iterative
    // refinement with two-side preconditioning.
    dlacpy_("A", &m, &n, AB, &mtemp, Acopy, &m, 1);
    dlacpy_("A", &p, &n, &AB[m], &mtemp, Bcopy, &p, 1);
    dcopy_(&m, c, &incx, ccopy, &incx);
    dcopy_(&p, d, &incx, dcopy, &incx);
    for (int i = 0; i < 11; i++)
        timeout[i] = 0.0;
    start = tic();
    mplse_gmres_twoside_scal(m, n, p, Acopy, m, Bcopy, p, ccopy, dcopy, x,
            work, works, lworks);
    over = tic();
    time1 = over-start;
    printf("%%Time of mplse_gmres_twoside: %.6f\n", over-start);
    dlacpy_("A", &m, &n, AB, &mtemp, Acopy, &m, 1);
    dlacpy_("A", &p, &n, &AB[m], &mtemp, Bcopy, &p, 1);
    check_accuracy(m, n, 1, Acopy, m, c, m, x, n);
    check_accuracy(p, n, 1, Bcopy, p, d, p, x, n);
    for (int i = 0; i < 4; i++)
        timeout[i] = work[i];
    dprintmat("time_mplse_gmres_twoside", 5, 1, timeout, 5);

    // Test the mixed precision LSE algorithm by using GMRES-based iterative
    // refinement with left preconditioning.
    dlacpy_("A", &m, &n, AB, &mtemp, Acopy, &m, 1);
    dlacpy_("A", &p, &n, &AB[m], &mtemp, Bcopy, &p, 1);
    dcopy_(&m, c, &incx, ccopy, &incx);
    dcopy_(&p, d, &incx, dcopy, &incx);
    start = tic();
    mplse_gmres_scal(m, n, p, Acopy, m, Bcopy, p, ccopy, dcopy, x, work,
            works, lworks);
    over = tic();
    time4 = over-start;
    printf("%%Time of mplse_gmres: %.6f\n", over-start);
    dlacpy_("A", &m, &n, AB, &mtemp, Acopy, &m, 1);
    dlacpy_("A", &p, &n, &AB[m], &mtemp, Bcopy, &p, 1);
    check_accuracy(m, n, 1, Acopy, m, c, m, x, n);
    check_accuracy(p, n, 1, Bcopy, p, d, p, x, n);
    for (int i = 0; i < 4; i++)
        timeout[i+4] = work[i];

    // Test the mixed precision LSE algorithm by using traditional iterative
    // refinement.
    dlacpy_("A", &m, &n, AB, &mtemp, Acopy, &m, 1);
    dlacpy_("A", &p, &n, &AB[m], &mtemp, Bcopy, &p, 1);
    dcopy_(&m, c, &incx, ccopy, &incx);
    dcopy_(&p, d, &incx, dcopy, &incx);
    start = tic();
    mplse(m, n, p, Acopy, m, Bcopy, p, ccopy, dcopy, x, work, works, lworks);
    over = tic();
    time3 = over-start;
    printf("%%Time of mplse: %.6f\n", over-start);
    dlacpy_("A", &m, &n, AB, &mtemp, Acopy, &m, 1);
    dlacpy_("A", &p, &n, &AB[m], &mtemp, Bcopy, &p, 1);
    check_accuracy(m, n, 1, Acopy, m, c, m, x, n);
    check_accuracy(p, n, 1, Bcopy, p, d, p, x, n);
    for (int i = 0; i < 5; i++)
        timeout[i+8] = work[i];

    // Test the fixed precision (double precision) LAPACK subroutine.
    dlacpy_("A", &m, &n, AB, &mtemp, Acopy, &m, 1);
    dlacpy_("A", &p, &n, &AB[m], &mtemp, Bcopy, &p, 1);
    dcopy_(&m, c, &incx, ccopy, &incx);
    dcopy_(&p, d, &incx, dcopy, &incx);
    start = tic();
    dgglse_(&m, &n, &p, Acopy, &m, Bcopy, &p, ccopy, dcopy, x, work,
           &lwork, &info);
    over = tic();
    time2 = over-start;
    timeout[13] = over-start;
    printf("%%Time of dgglse: %.6f\n", over-start);
    dlacpy_("A", &m, &n, AB, &mtemp, Acopy, &m, 1);
    dlacpy_("A", &p, &n, &AB[m], &mtemp, Bcopy, &p, 1);
    check_accuracy(m, n, 1, Acopy, m, c, m, x, n);
    check_accuracy(p, n, 1, Bcopy, p, d, p, x, n);
    printf("%%radio-mplse_gmres_twoside: %.6f\n", (time2-time1)/time2);
    printf("%%radio-mplse_gmres: %.6f\n", (time2-time4)/time2);
    printf("%%radio-mplse: %.6f\n", (time2-time3)/time2);

    dprintmat("timesub", 14, 1, timeout, 14);

    free(AB);
    free(c);
    free(d);
    free(x);
    free(ccopy);
    free(dcopy);
    free(Acopy);
    free(Bcopy);
    free(work);
    free(works);
    return 0;
}

int test(int times_mn, int times_pn, int n0, double cond, double *time)
{
    int num = 3;
    int m, n = n0, p;

    for (int i = 0; i < num; i++)
    {
        n = n + 1024; m = n*times_mn; p = n/times_pn;
        //p = p*2; n = p*times_np; m = n*times_mn;
        test_lse(m, n, p, cond, time+14*i);
    }
    dprintmat("time", 14, 3, time, 14);
    return 0;
}

int test_all(double cond)
{
    double *time = (double *)calloc(14*3*6, sizeof(double));

    test(8, 64,  0, cond, time);
    test(8, 32,  0, cond, time+14*3);
    test(10, 64, 0, cond, time+14*3*2);
    test(10, 32, 0, cond, time+14*3*3);
    test(12, 64, 0, cond, time+14*3*4);
    test(12, 32, 0, cond, time+14*3*5);

    dprintmat("timeall", 14, 18, time, 14);

    free(time);
    return 0;
}

int main (int argc, char **argv)
{
    double *time = (double *)calloc(4*11*8, sizeof(double));
    int n = 2048;
    n = 128;
    // Warm up.
    test_lse(n*256, n, n/2, 1e5, time);
    // Start tests.
    test_all(1e3);
    return 0;
}
