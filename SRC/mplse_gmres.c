#include <stdio.h>
#include <stdlib.h>
#include "../include/lapack.h"
#include "../include/mpls.h"
#include "../include/mpls_util.h"
#include "../include/my_wtime.h"

#define MEASURETIME 0


int mplse_gmres_scal(int m, int n, int p, double *A, int lda, double *B,
    int ldb, double *c, double *d, double *xout, double *work,
    float *works, int lwork)
{
#ifdef MEASURETIME
    printf("%% Measure time\n");
    double start, over;
    double *time = (double*)calloc(5, sizeof(double));
    start = tic();
#endif
    float *As, *As_copy, *Bs, *cs, *cs_copy, *ds, *xs, *rs, *workssub;
    float *Atrs, *r1s, *r2s, *r3s, *T2s;
    double *r, *orth, *Atr, *Atr_iter, *lambda;
    double *r1, *r2, *r3, *x;
    double one = 1.0, zero  = 0.0;
    float onef = 1.0f, ronef = -1.0f, zerof = 0.0f;
    int lworkssub, num_iter;
    int info, incx = 1, mn = MIN(m, n), ntemp, length;;
    double norma, normb, tol = 1e-12;
    double *b_aug, *x_aug, *As_d, *Bs_d, *workssub_d, *T2s_d, *worksub;
    double alpha = 1.0, invalpha, rinvalpha;

    As = works;
    As_copy = As + m*n;
    Bs = As_copy + m*n;
    cs = Bs + p*n;
    cs_copy = cs + m;
    ds = cs_copy + m;
    xs = ds + p;
    rs = xs + n;
    Atrs = rs + m;
    r1s = Atrs + n;
    r2s = r1s + m;
    r3s = r2s + p;
    T2s = r3s + n;
    workssub = T2s + n*p;
    lworkssub = lwork - 2*m*n - 2*p*n - 4*m - 2*p - 3*n;

    r = work;
    lambda = r + m;
    x = lambda + p;
    r1 = x + n;
    r2 = r1 + m;
    r3 = r2 + p;
    Atr = r3 + n;
    Atr_iter = Atr + n;
    orth = Atr_iter + n;
    As_d = orth + n;
    Bs_d = As_d+ m*n;
    T2s_d = Bs_d + n*p;
    b_aug = T2s_d + n*p;
    x_aug = b_aug + m + n + p;
    workssub_d = x_aug + m + n + p;
    worksub = workssub_d + m*n*4;

    dlag2s_(&m, &n, A, &lda, As, &m, &info);
    length = m*n;
    norma = (double)snrm2_(&length, As, &incx);
    slacpy_("A", &m, &n, As, &m, As_copy, &m, 1);
    dlag2s_(&p, &n, B, &ldb, Bs, &p, &info);
    length = n*p;
    normb = (double)snrm2_(&length, Bs, &incx);
    dlag2s_(&m, &incx, c, &m, cs, &m, &info);
    scopy_(&m, cs, &incx, cs_copy, &incx);
    dlag2s_(&p, &incx, d, &p, ds, &p, &info);
#ifdef MEASURETIME
    over = tic();
    time[0] = over-start;
#endif
    // Use LAPACK subroutine sgglse to compute the LSE solution x in lower
    // precision.
    // At the same time, we can obtain the GRQ factorization of B and A.
#ifdef MEASURETIME
    start = tic();
#endif
    sgglse_qz_(&m, &n, &p, As, &m, Bs, &p, cs, ds, xs, workssub, &lworkssub,
            &info);
#ifdef MEASURETIME
    over = tic();
    time[1] = over-start;
#endif
    // Compute the solution r and lambda of the 3-block saddle-point system.
    // First compute r in lower precision.
#ifdef MEASURETIME
    start = tic();
#endif
    scopy_(&m, cs_copy, &incx, rs, &incx);
    sgemv_("N", &m, &n, &ronef, As_copy, &m, xs, &incx, &onef, rs, &incx, 1);
    // Then compute A^Tr in workint precision.
    slag2d_(&m, &incx, rs, &m, r, &m, &info);
    dgemv_("T", &m, &n, &one, A, &lda, r, &incx, &zero, Atr, &incx, 1);
    // Then compute lamda by solving B^T lambda=Atr in lower precision.
    // lambda stores in Atrs[n-p:n-1]
    dlag2s_(&n, &incx, Atr, &n, Atrs, &n, &info);
    sormrq_("L", "N", &n, &incx, &p, Bs, &p, workssub, Atrs, &n,
            &workssub[p+mn], &lworkssub, &info, 1, 1);
    strtrs_("U", "T", "N", &p, &incx, &Bs[(n-p)*p], &p, &Atrs[n-p], &p,
            &info, 1, 1, 1);
    slag2d_(&p, &incx, &Atrs[n-p], &p, lambda, &p, &info);
    slag2d_(&n, &incx, xs, &n, x, &n, &info);


    ntemp = n-p;
    slaset_("L", &n, &p, &zerof, &zerof, T2s, &n, 1);
    slacpy_("A", &ntemp, &p, &As[ntemp*m], &m, T2s, &n, 1);
    slacpy_("U", &p, &p, &As[ntemp*m+ntemp], &m, &T2s[ntemp], &n, 1);
    length = m+p+n;
#ifdef MEASURETIME
    over = tic();
    time[2] = over-start;
#endif

#ifdef MEASURETIME
    start = tic();
#endif
    slag2d_(&m, &n, As, &m, As_d, &m, &info);
    // Find alpha.
    invalpha = dnrm2_(&m, r, &incx);
    alpha = 1.0/invalpha;
    //alpha = 1.0;
    rinvalpha = -invalpha;
    slag2d_(&p, &n, Bs, &p, Bs_d, &p, &info);
    slag2d_(&n, &p, T2s, &n, T2s_d, &n, &info);
    slag2d_(&length, &incx, workssub, &length, workssub_d, &length,
        &info);
    dcopy_(&m, r, &incx, x_aug, &incx);
    dscal_(&m, &invalpha, x_aug, &incx);
    dcopy_(&p, lambda, &incx, &x_aug[m], &incx);
    dscal_(&p, &rinvalpha, &x_aug[m], &incx);
    dcopy_(&n, x, &incx, &x_aug[m+p], &incx);
    dcopy_(&m, c, &incx, b_aug, &incx);
    dcopy_(&p, d, &incx, &b_aug[m], &incx);
    for (int i = 0; i < n; i++)
        b_aug[m+p+i] = 0.0;
    gmres_lse_left_scal(length, m, n, p, A, lda, B, ldb, As_d, m, Bs_d, p,
            T2s_d, alpha, workssub_d, b_aug, x_aug, 300, 200, &num_iter,
            tol, worksub, lwork);
#ifdef MEASURETIME
    over = tic();
    time[3] = over-start;
    for (int i = 0; i < 4; i++)
        work[i] = time[i];
#endif
    printf("iter = %d\n", num_iter);

    dcopy_(&n, &x_aug[m+p], &incx, xout, &incx);

    return 0;
}
