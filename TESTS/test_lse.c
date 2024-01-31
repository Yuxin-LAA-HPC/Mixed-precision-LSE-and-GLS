#include <stdio.h>
#include <stdlib.h>
#include "../include/lapack.h"
#include "../include/mpls_util.h"
#include "../include/mpls.h"
#include "../include/my_wtime.h"

int test_lse(int m, int n, int p, double *timeout)
{
    int i, j, incx = 1;
    int lworks = 2*m*n+2*p*n+4*m+3*n+2*p+2*m*n;
    int lwork = m*n+m+p+n*n;//3*m+5*n+2*p;//3*m*n;
    double *AB = (double *)calloc((m+p)*n, sizeof(double));
    double *A = AB, *B = A + m*n;
    double *c = (double *)calloc(m, sizeof(double));
    double *d = (double *)calloc(p, sizeof(double));
    double *x = (double *)calloc(n, sizeof(double));
    double *Acopy = (double *)calloc(m*n, sizeof(double));
    double *Bcopy = (double *)calloc(p*n, sizeof(double));
    double *ccopy = (double *)calloc(m, sizeof(double));
    double *dcopy = (double *)calloc(p, sizeof(double));
    double *work = (double *)calloc(lwork, sizeof(double));
    double *sva = (double *)calloc(n, sizeof(double));
    float *works = (float *)calloc(lworks, sizeof(float));
    int *iseed = (int *)calloc(4, sizeof(int));
    double rcond, norma, normb, cond;
    int rank, oneint = 1, twoint = 2, info;
    int ranko = 4, irsigm, idist, mode, mtemp;
    double start, over, time;

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
    //dlatm1_(&mode, &cond, &irsigm, &idist, iseed, sva, &n, &info);
    mtemp = m+p;
    dqrt13_(&oneint, &mtemp, &n, AB, &mtemp, &norma, iseed);
    printf("%%norm of AB: %.6f\n", norma);
    for (int i = 0; i < m; i++)
        c[i] = 1.0;
    for (int i = 0; i < p; i++)
        d[i] = 1.0;

    dlacpy_("A", &m, &n, A, &m, Acopy, &m, 1);
    dlacpy_("A", &p, &n, B, &p, Bcopy, &p, 1);
    dcopy_(&m, c, &incx, ccopy, &incx);
    dcopy_(&p, d, &incx, dcopy, &incx);
    //dprintmat("A", m, n, A, m);
    //dprintmat("B", p, n, B, p);
    //dprintmat("c", m, 1, c, m);
    //dprintmat("d", p, 1, d, p);
    for (int i = 0; i < 11; i++)
        timeout[i] = 0.0;
    start = tic();
    mplse_2block(m, n, p, Acopy, m, Bcopy, p, ccopy, dcopy, x, work, works, lworks);
    over = tic();
    double time1 = over-start;
    printf("%%Time of mplse_2block: %.6f\n", over-start);
    check_accuracy(m, n, 1, A, m, c, m, x, n);
    check_accuracy(p, n, 1, B, p, d, p, x, n);
    for (int i = 0; i < 5; i++)
        timeout[i] = work[i];
    //dprintmat("x", n, 1, x, n);

    dlacpy_("A", &m, &n, A, &m, Acopy, &m, 1);
    dlacpy_("A", &p, &n, B, &p, Bcopy, &p, 1);
    dcopy_(&m, c, &incx, ccopy, &incx);
    dcopy_(&p, d, &incx, dcopy, &incx);
    start = tic();
    mplse_2block_totalh(m, n, p, Acopy, m, Bcopy, p, ccopy, dcopy, x, work, works, lworks);
    over = tic();
    double time4 = over-start;
    printf("%%Time of mplse_2block_totalh: %.6f\n", over-start);
    check_accuracy(m, n, 1, A, m, c, m, x, n);
    check_accuracy(p, n, 1, B, p, d, p, x, n);
    for (int i = 0; i < 5; i++)
        timeout[i] = work[i];

    dlacpy_("A", &m, &n, A, &m, Acopy, &m, 1);
    dlacpy_("A", &p, &n, B, &p, Bcopy, &p, 1);
    dcopy_(&m, c, &incx, ccopy, &incx);
    dcopy_(&p, d, &incx, dcopy, &incx);
    start = tic();
    mplse(m, n, p, Acopy, m, Bcopy, p, ccopy, dcopy, x, work, works, lworks);
    over = tic();
    double time3 = over-start;
    printf("%%Time of mplse_3block: %.6f\n", over-start);
    check_accuracy(m, n, 1, A, m, c, m, x, n);
    check_accuracy(p, n, 1, B, p, d, p, x, n);
    for (int i = 0; i < 5; i++)
        timeout[i+5] = work[i];
    //dprintmat("xtrue", n, 1, x, n);

    dlacpy_("A", &m, &n, A, &m, Acopy, &m, 1);
    dlacpy_("A", &p, &n, B, &p, Bcopy, &p, 1);
    dcopy_(&m, c, &incx, ccopy, &incx);
    dcopy_(&p, d, &incx, dcopy, &incx);
    start = tic();
    dgglse_(&m, &n, &p, Acopy, &m, Bcopy, &p, ccopy, dcopy, x, work,
           &lwork, &info);
    over = tic();
    double time2 = over-start;
    timeout[10] = over-start;
    printf("%%Time of dgglse: %.6f\n", over-start);
    check_accuracy(m, n, 1, A, m, c, m, x, n);
    check_accuracy(p, n, 1, B, p, d, p, x, n);
    printf("%%radio-2block: %.6f\n", (time2-time1)/time2);
    printf("%%radio-2block-totalh: %.6f\n", (time2-time4)/time2);
    printf("%%radio-3block: %.6f\n", (time2-time3)/time2);
    //dprintmat("xtrue", n, 1, x, n);

    //dprintmat("timesub", 11, 1, timeout, 11);
    printf("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n");

    free(AB);
    free(c);
    free(d);
    free(x);
    free(ccopy);
    free(dcopy);
    free(Acopy);
    free(Bcopy);
    free(work);
    free(sva);
    free(works);
    free(iseed);
    return 0;
}

int test(int times_np, int times_mn, int p0, double *time)
{
    int num = 4;
    //double *time = (double *)calloc(num*6, sizeof(double));
    int m, n, p = p0;

    for (int i = 0; i < num; i++)
    {
        p = p*2; n = p*times_np; m = n*times_mn;
        test_lse(m, n, p, time+11*i);
    }
    dprintmat("time", 11, 4, time, 11);
    //free(time);
    return 0;
}

int test_largecase()
{
    double *time = (double *)calloc(6*6, sizeof(double));
    int m, n, p = 512;

    n = p*4; m = n*6;
    test_lse(m, n, p, time);
    n = p*4; m = n*8;
    test_lse(m, n, p, time+6);
    n = p*4; m = n*10;
    test_lse(m, n, p, time+12);
    n = p*8; m = n*2;
    test_lse(m, n, p, time+18);
    n = p*8; m = n*4;
    test_lse(m, n, p, time+24);
    n = p*8; m = n*6;
    test_lse(m, n, p, time+30);
    dprintmat("timeall", 6, 36, time, 6);
    free(time);
    return 0;
}

int main (int argc, char **argv)
{
    double *time = (double *)calloc(4*11*8, sizeof(double));
    test(8, 1, 64, time+264);
    test(16, 1, 64, time+308);
    test(4, 6, 64, time);
    //test_largecase();
    test(4, 8, 64, time+44);
    test(4, 10, 64, time+88);
    test(8, 2, 64, time+132);
    test(8, 4, 64, time+176);
    test(8, 6, 64, time+220);
    dprintmat("timeall", 11, 32, time, 11);
    free(time);
    //int m = 4096*8, n = 1024*2, p = 128*8;
    ////m = 10;
    ////n = 6;
    ////p = 4;
    //test_lse(m, n, p);
    //test_lse(m, n, p);
    //test_lse(m, n, p);
    //m = 4096; n = 1024; p = 128;
    //test_lse(m, n, p);
    //m = 4096; n = 1024; p = 128;
    //test_lse(m, n, p);
    //m = 4096*2; n = 1024*2; p = 128*2;
    //test_lse(m, n, p);
    //m = 4096*4; n = 1024*4; p = 128*4;
    //test_lse(m, n, p);
    //m = 4096*4; n = 1024*4; p = 128*2;
    //test_lse(m, n, p);
    ////test_lse(4096*4, 2048, 256);
    return 0;
}
