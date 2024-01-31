#include <stdio.h>
#include <stdlib.h>
#include "../include/lapack.h"
#include "../include/mpls_util.h"
#include "../include/mpls.h"
#include "../include/my_wtime.h"

int test_gls(int n, int m, int p, double *timesum)
{
    int i, j, incx = 1;
    int lwork = 2*m*n+2*n+m+p, lworks = 2*n*m+2*n*p+n*3+m*3+p*2+MAX(n,p)*65;
    double *AB = (double *)calloc((m+p)*n, sizeof(double));
    double *A = AB, *B = A + m*n;
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
    double rcond, norma, normb, cond;
    int rank, oneint = 1, twoint = 2, info;
    int ranko = 4, irsigm, idist, mode, mtemp;
    double start, over;
    //double *timesum = (double *)calloc(6, sizeof(double));

    // Generate testing matrix.
    srand(0);
    irsigm = 0;
    idist = 3;
    mode = 5;
    cond = 1e6;
    iseed[0] = rand()%1000;
    iseed[1] = rand()%1000;
    iseed[2] = rand()%1000;
    iseed[3] = rand()%1000;
    dlatm1_(&mode, &cond, &irsigm, &idist, iseed, sva, &n, &info);
    mtemp = m+p;
    dqrt13_(&oneint, &n, &mtemp, AB, &n, &norma, iseed);
    printf("%%norm of AB: %.6f\n", norma);
    for (int i = 0; i < n; i++)
        d[i] = 1.0;

    dlacpy_("A", &n, &m, A, &n, Acopy, &n, 1);
    dlacpy_("A", &n, &p, B, &n, Bcopy, &n, 1);
    dcopy_(&n, d, &incx, dcopy, &incx);
    //dprintmat("A", n, m, A, n);
    //dprintmat("B", n, p, B, n);
    //dprintmat("d", n, 1, d, n);
    start = tic();
    mpgls(n, m, p, Acopy, n, Bcopy, n, dcopy, x, y, works, lworks, work,
            lwork);
    over = tic();
    double timemp = over-start;
    printf("%%Time of mpgls: %.6f\n", over-start);
    check_accuracy_gls(n, m, p, A, n, B, n, d, x, y);
    for (int i = 0; i < 5; i++)
        timesum[i] = work[i];
    //dprintmat("x", m, 1, x, m);
    //dprintmat("y", p, 1, y, p);

    dlacpy_("A", &n, &m, A, &n, Acopy, &n, 1);
    dlacpy_("A", &n, &p, B, &n, Bcopy, &n, 1);
    dcopy_(&n, d, &incx, dcopy, &incx);
    start = tic();
    dggglm_(&n, &m, &p, Acopy, &n, Bcopy, &n, dcopy, x, y, work,
           &lwork, &info);
    over = tic();
    double timed = over-start;
    timesum[5] = timed;
    printf("%%Time of dggglm: %.6f\n", over-start);
    check_accuracy_gls(n, m, p, A, n, B, n, d, x, y);
    printf("%% ratio: %.4f\n", (timed-timemp)/timed);
    dprintmat("timesubsum", 6, 1, timesum, 6);
    //dprintmat("x", m, 1, x, m);
    //dprintmat("y", p, 1, y, p);

    //free(timesum);
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

int test()
{
    double *time = (double *)calloc(6*4*6, sizeof(double));
    double r_m, r_p;
    int n, m, p, n0 = 1024, m0 = 512;

    r_m = 2, r_p = 1;
    n = n0;
    for (int i = 0; i < 4; i++)
    {
        n = n*2, m = n/r_m, p = (n-m)*r_p;
        test_gls(n, m, p, time+i*6);
    }

    r_m = 4, r_p = 1;
    n = n0, m = m0;
    for (int i = 0; i < 4; i++)
    {
        n = n*2, m = n/r_m, p = (n-m)*r_p;
        test_gls(n, m, p, time+24+i*6);
    }

    r_m = 2, r_p = 2;
    n = n0, m = m0;
    for (int i = 0; i < 4; i++)
    {
        n = n*2, m = n/r_m, p = (n-m)*r_p;
        test_gls(n, m, p, time+24*2+i*6);
    }

    r_m = 4, r_p = 2;
    n = n0, m = m0;
    for (int i = 0; i < 4; i++)
    {
        n = n*2, m = n/r_m, p = (n-m)*r_p;
        test_gls(n, m, p, time+24*3+i*6);
    }

    r_m = 2, r_p = 3;
    n = n0, m = m0;
    for (int i = 0; i < 4; i++)
    {
        n = n*2, m = n/r_m, p = (n-m)*r_p;
        test_gls(n, m, p, time+24*4+i*6);
    }

    r_m = 4, r_p = 3;
    n = n0, m = m0;
    for (int i = 0; i < 4; i++)
    {
        n = n*2, m = n/r_m, p = (n-m)*r_p;
        test_gls(n, m, p, time+24*5+i*6);
    }
    dprintmat("timesum", 24, 6, time, 24);

    free(time);
    return 0;
}

int main (int argc, char **argv)
{
    // m <= n <= m+p
    test();
    //double *time = (double *)calloc(6*4*6, sizeof(double));
    //int n = 4096, m = 1024*2, p = n-m+128;
    //n = 10;
    //m = 6;
    //p = 5;
    //test_gls(n, m, p);
    //test_gls(n, m, p);
    //n = 4096*2, m = 1024*4, p = n-m+128*2;
    //test_gls(n, m, p);
    //n = 4096*4, m = 1024*8, p = n-m+128*4;
    //test_gls(n, m, p);
    //n = 4096, m = n, p = 128;
    //test_gls(n, m, p);
    return 0;
}
