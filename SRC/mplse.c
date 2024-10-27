#include <stdio.h>
#include <stdlib.h>
#include "../include/lapack.h"
#include "../include/mpls.h"
#include "../include/mpls_util.h"
#include "../include/my_wtime.h"

#define MEASURETIME 0

int mplse(int m, int n, int p, double *A, int lda, double *B, int ldb,
        double *c, double *d, double *xout, double *work, float *works,
        int lwork)
{
#ifdef MEASURETIME
    printf("%% Measure time\n");
    double start, over;
    double *time = (double*)calloc(5, sizeof(double));
    start = tic();
#endif
    float *As, *As_copy, *Bs, *cs, *cs_copy, *ds, *xs, *rs, *workssub;
    float *Atrs, *r1s, *r2s, *r3s, *T2s, *Bstemp;
    double *r, *orth, *Atr, *Atr_iter, *lambda;
    double *r1, *r2, *r3, *x;
    double one = 1.0, rone = -1.0, zero  = 0.0;
    float onef = 1.0f, ronef = -1.0f, zerof = 0.0f;
    int lworkssub, maxiter = 50;
    int i, info, incx = 1, mn = MIN(m, n), ntemp, length;;
    double norma, normb, normx, normr, normd, tol = 1e-13;
    double normr2, normr3, normlambda;

    As = works;
    As_copy = As + m*n;
    Bs = As_copy + m*n;
    Bstemp = Bs + p*n;
    cs = Bstemp + p*n;
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
    lworkssub = lwork - 2*m*n - 3*p*n - 4*m - 2*p - 3*n;

    r = work;
    lambda = r + m;
    x = lambda + p;
    r1 = x + n;
    r2 = r1 + m;
    r3 = r2 + p;
    Atr = r3 + n;
    Atr_iter = Atr + n;
    orth = Atr_iter + n;

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
    normd = (double)snrm2_(&p, ds, &incx);
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
    for (i = 0; i < n; i++)
        scopy_(&p, &Bs[i*p], &incx, &Bstemp[i], &n);
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
    for (int iter = 0; iter < maxiter; iter++)
    {
        // Compute the residual of the 3-block augmemt system.
#ifdef MEASURETIME
        start = tic();
#endif
        dcopy_(&m, c, &incx, r1, &incx);
        normr = dnrm2_(&n, r, &incx);
        daxpy_(&m, &rone, r, &incx, r1, &incx);
        normx = dnrm2_(&n, x, &incx);
        dgemv_("N", &m, &n, &rone, A, &lda, x, &incx, &one, r1, &incx, 1);
        dcopy_(&p, d, &incx, r2, &incx);
        dgemv_("N", &p, &n, &rone, B, &ldb, x, &incx, &one, r2, &incx, 1);
        normr2 = dnrm2_(&p, r2, &incx);
        dgemv_("T", &m, &n, &one, A, &lda, r, &incx, &zero, r3, &incx, 1);
        dgemv_("T", &p, &n, &one, B, &ldb, lambda, &incx, &rone, r3, &incx,
                1);
        normlambda = dnrm2_(&p, lambda, &incx);
        normr3 = dnrm2_(&n, r3, &incx);
        // Check convergence.
        if ((normr2 < (normd + normb*normx)*tol) && (normr3 < (normb*normlambda + norma*normr)*tol))
        {
            dcopy_(&n, x, &incx, xout, &incx);
            printf("iter = %d;\n", iter);
#ifdef MEASURETIME
            for (int ii = 0; ii < 5; ii++)
                work[ii] = time[ii];
            free(time);
#endif
            return 0;
        }
#ifdef MEASURETIME
        over = tic();
        time[3] = time[3] + over-start;
#endif
        // Solve the correction of [r;-lambda;x] by solving linear system.
        // Solve the correction linear system by the RQ factorization of B
        // and A.
#ifdef MEASURETIME
        start = tic();
#endif
        dlag2s_(&length, &incx, r1, &length, r1s, &length, &info);
        strtrs_("U", "N", "N", &p, &incx, &Bs[(n-p)*p], &p, r2s, &p, &info,
                1, 1, 1);
        scopy_(&p, r2s, &incx, &xs[n-p], &incx);//y_2 stores in &xs[n-p].

        sormrq_("L", "N", &n, &incx, &p, Bs, &p, workssub, r3s, &n,
                &workssub[p+mn], &lworkssub, &info, 1, 1);
        strtrs_("U", "T", "N", &ntemp, &incx, As, &m, r3s, &ntemp, &info,
                1, 1, 1);// q_1 stores in r3s.
        sormqr_("L", "T", &m, &incx, &mn, As, &m, &workssub[p], r1s, &m,
                &workssub[p+mn], &lworkssub, &info, 1, 1);// q_3 stores in &r1s[n].
        sgemv_("N", &p, &p, &ronef, &T2s[ntemp], &n, &xs[n-p],
                &incx, &onef, &r1s[n-p], &incx, 1);// q_2 stores in &r1s[n-p].
        saxpy_(&ntemp, &ronef, r3s, &incx, r1s, &incx);
        sgemv_("N", &ntemp, &p, &ronef, &As[ntemp*m], &m, &xs[n-p], &incx,
                &onef, r1s, &incx, 1); // r1s is -q_1-T_12*y_2+(Z^Tf)_1.
        strtrs_("U", "N", "N", &ntemp, &incx, As, &m, r1s, &ntemp, &info,
                1, 1, 1);
        scopy_(&ntemp, r1s, &incx, xs, &incx);//y stores in xs.
        sormrq_("L", "T", &n, &incx, &p, Bs, &p, workssub, xs, &n,
                &workssub[p+mn], &lworkssub, &info, 1, 1);// Delta x stores in xs.
        scopy_(&ntemp, r3s, &incx, r1s, &incx);//q stores in r1s.
        sgemv_("T", &n, &p, &onef, T2s, &n, r1s, &incx, &ronef, &r3s[n-p],
                &incx, 1);
        strtrs_("U", "T", "N", &p, &incx, &Bs[(n-p)*p], &p, &r3s[n-p], &p,
                &info, 1, 1, 1);// Delta lambda stores in &r3s[n-p].
        sormqr_("L", "N", &m, &incx, &mn, As, &m, &workssub[p], r1s, &m,
                &workssub[p+mn], &lworkssub, &info, 1, 1);// Delta r stores in r1s.
        // Update r, lambda, x.
        for (i = 0; i < m; i++)
            r[i] = r[i] + (double)r1s[i];
        for (i = 0; i < p; i++)
            lambda[i] = lambda[i] + (double)r3s[n-p+i];
        for (i = 0; i < n; i++)
            x[i] = x[i] + (double)xs[i];
#ifdef MEASURETIME
        over = tic();
        time[4] = time[4] + over-start;
#endif

    }

    printf("%% MAXITER is too small!\n");
    dcopy_(&n, x, &incx, xout, &incx);
    return 1;
}

