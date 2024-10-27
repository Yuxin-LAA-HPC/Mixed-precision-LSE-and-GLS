#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../include/mpls_util.h"
#include "../include/mpls.h"
#include "../include/lapack.h"
#include "../include/my_wtime.h"

#define MEASURETIME 0

int mpgls_gmres_twoside(int n, int m, int p, double *A, int lda, double *B,
        int ldb, double *d, double *x, double *y, float *works, int lworks,
        double *work, int lwork)
{
#ifdef MEASURETIME
    printf("%% Measure time\n");
    double start, over;
    double *time = (double*)calloc(5, sizeof(double));
    start = tic();
#endif
    float *As, *Bs, *Ts, *ds, *xs, *ys, *zs, *r1s, *r2s, *r3s, *workssub;
    float *Bstemp;
    int lworkssub, lworksub, info = 0, incx = 1, i;
    int np = MIN(n, p), ntemp = n-m;
    int pnm = p-n+m, length;
    float zerof = 0.0f;
    double *r1, *r2, *r3, *z, *worksub;
    double *As_d, *Bs_d, *Ts_d, *workssub_d, *x_aug, *b_aug;
    double normA, normB, normd;
    double tol = 3.0*1e-16;
    int maxiter = 10, num_iter;
    double alpha = 1.0, invalpha, rinvalpha;

    // Require float space n*m + 2*n*p + 4*n + 3*m + 2*p + MAX(n, p)*64
    As = works;
    Bs = As + n*m;
    Ts = Bs + n*p;
    Bstemp = Ts + n*p;
    ds = Bstemp + n*p;
    xs = ds + n;
    ys = xs + m;
    zs = ys + p;
    r1s = zs + n;
    r2s = r1s + p;
    r3s = r2s + n;
    workssub = r3s + m;
    lworkssub = lworks - n*m - 3*n*p - n*3 - m*2 - p*2;

    // Require double space n*m + 2*n*p + 4*n + 4*m + 3*p + MAX(n, p)*64
    // + 9*(m+n+p) + (m+n+p)*restart + 6*restart + restart*restart
    z = work;
    r1 = z + n;
    r2 = r1 + p;
    r3 = r2 + n;
    As_d = r3 + m;
    Bs_d = As_d + n*m;
    Ts_d = Bs_d + n*p;
    x_aug = Ts_d + n*p;
    b_aug = x_aug + m + n + p;
    workssub_d = b_aug + m + n + p;
    worksub = workssub_d + m + n + MAX(n, p)*256;
    lworksub = lwork - 4*n - 4*m - 3*p - n*m - 2*n*p - MAX(n, p)*256;

    length = n*m;
    dlag2s_(&n, &m, A, &lda, As, &n, &info);
    normA = (double) (snrm2_(&length, As, &incx));
    length = n*p;
    dlag2s_(&n, &p, B, &ldb, Bs, &n, &info);
    normB = (double) (snrm2_(&length, Bs, &incx));
    dlag2s_(&n, &incx, d, &n, ds, &n, &info);
    normd = dnrm2_(&n, d, &incx);
#ifdef MEASURETIME
    over = tic();
    time[0] = over-start;
    start = tic();
#endif
    sggglm_qz_(&n, &m, &p, As, &n, Bs, &n, ds, xs, ys, workssub, &lworkssub,
            &info);
#ifdef MEASURETIME
    over = tic();
    time[1] = over-start;
    start = tic();
#endif

    for (i = 0; i < n; i++)
        scopy_(&p, &Bs[i], &n, &Bstemp[i*p], &incx);
    // Compute the initial z.
    memset(zs, 0.0f, m*sizeof(float));
    scopy_(&p, ys, &incx, r1s, &incx);
    sormql_("L", "T", &p, &incx, &np, &Bstemp[MAX(n-p, 0)*p], &p, &workssub[m],
            r1s, &p, &workssub[m+np], &lworkssub, &info, 1, 1);
    strtrs_("U", "T", "N", &ntemp, &incx, &Bs[(m+p-n)*n+m], &n, &r1s[pnm],
            &ntemp, &info, 1, 1, 1);
    scopy_(&ntemp, &r1s[pnm], &incx, &zs[m], &incx);
    sormqr_("L", "N", &n, &incx, &m, As, &n, workssub, zs, &n,
            workssub+m+np, &lworkssub, &info, 1, 1);

    slaset_("A", &n, &p, &zerof, &zerof, Ts, &n, 1);
    if (n > p)
    {
        length = n-p;
        slacpy_("A", &length, &p, Bs, &n, Ts, &n, 1);
        slacpy_("U", &p, &p, &Bs[n-p], &n, &Ts[n-p], &n, 1);
    }
    else
        slacpy_("U", &n, &n, &Bs[(p-n)*n], &n, &Ts[(p-n)*n], &n, 1);

    lworkssub = lworkssub - m - np;
#ifdef MEASURETIME
    over = tic();
    time[2] = over-start;
    start = tic();
#endif
    length = m+p+n;
    slag2d_(&n, &m, As, &n, As_d, &n, &info);
    // Find alpha.
    //alpha = 1.0;
    slag2d_(&n, &p, Bs, &n, Bs_d, &n, &info);
    slag2d_(&n, &p, Ts, &n, Ts_d, &n, &info);
    slag2d_(&length, &incx, workssub, &length, workssub_d, &length, &info);
    for (int i = 0; i < length; i++)
        b_aug[i] = 0.0;
    dcopy_(&n, d, &incx, &b_aug[p], &incx);
    slag2d_(&p, &incx, ys, &p, x_aug, &p, &info);
    slag2d_(&n, &incx, zs, &n, &x_aug[p], &n, &info);
    invalpha = dnrm2_(&p, x_aug, &incx);
    alpha = 1.0/invalpha;
    rinvalpha = -alpha;
    printf("alpha = %.16f\n", alpha);
    dscal_(&n, &rinvalpha, &x_aug[p], &incx);
    slag2d_(&m, &incx, xs, &m, &x_aug[p+n], &m, &info);
    if (n <= p)
        gmres_gls_twoside_plarge(length, m, n, p, A, lda, B, ldb,
        As_d, n, Bs_d, n, Ts_d, alpha, workssub_d, b_aug, x_aug,
        MIN(300, m+n+p), 200, &num_iter, tol, worksub, lwork);
    else
        gmres_gls_twoside_nlarge(length, m, n, p, A, lda, B, ldb,
        As_d, n, Bs_d, n, Ts_d, alpha, workssub_d, b_aug, x_aug,
        MIN(300, m+n+p), 200, &num_iter, tol, worksub, lwork);

    dcopy_(&p, x_aug, &incx, y, &incx);
    dcopy_(&m, &x_aug[p+n], &incx, x, &incx);
#ifdef MEASURETIME
    over = tic();
    time[3] = over-start;
    for (int i = 0; i < 4; i++)
        work[i] = time[i];
    free(time);
#endif

    printf("iter = %d\n", num_iter);
    return 0;
}

