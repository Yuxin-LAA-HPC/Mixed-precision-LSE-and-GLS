#include <stdio.h>
#include <string.h>
#include "../include/mpls_util.h"
#include "../include/mpls.h"
#include "../include/lapack.h"
#include "../include/my_wtime.h"

#define MEASURETIME 0

int mpgls(int n, int m, int p, double *A, int lda, double *B, int ldb,
        double *d, double *x, double *y, float *works, int lworks,
        double *work, int lwork)
{
#ifdef MEASURETIME
    printf("%% Measure time\n");
    double start, over;
    double *time = (double*)calloc(5, sizeof(double));
    start = tic();
#endif
    float *As, *Bs, *Ts, *ds, *xs, *ys, *zs, *r1s, *r2s, *r3s, *workssub;
    int lworkssub, lworksub, info = 0, incx = 1, i;
    int np = MIN(n, p), ntemp = n-m;
    int nsum = m+n+p, pnm = p-n+m, length;
    double one = 1.0, rone = -1.0, zero = 0.0;
    float onef = 1.0f, ronef = -1.0f;
    double *r1, *r2, *r3, *z, *worksub;
    double normA, normB, normd, normr1, normr2, normr3, normx, normy, normz;
    double tol = 1e-14;
    int maxiter = 10;

    As = works;
    Bs = As + n*m;
    Ts = Bs + n*p;
    ds = Ts + n*p;
    xs = ds + n;
    ys = xs + m;
    zs = ys + p;
    r1s = zs + n;
    r2s = r1s + p;
    r3s = r2s + n;
    workssub = r3s + m;
    lworkssub = lworks - n*m - 2*n*p - n*3 - m*2 - p*2;

    z = work;
    r1 = z + n;
    r2 = r1 + p;
    r3 = r2 + n;
    worksub = r3 + m;
    lworksub = lwork - 2*n - m - p;

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

    // Compute the initial z.
    memset(zs, 0.0f, m*sizeof(float));
    scopy_(&p, ys, &incx, r1s, &incx);
    sormrq_("L", "N", &p, &incx, &np, &Bs[MAX(n-p, 0)], &n, &workssub[m],
            r1s, &p, &workssub[m+np], &lworkssub, &info, 1, 1);
    strtrs_("U", "T", "N", &ntemp, &incx, &Bs[(m+p-n)*n+m], &n, &r1s[pnm],
            &ntemp, &info, 1, 1, 1);
    scopy_(&ntemp, &r1s[pnm], &incx, &zs[m], &incx);
    sormqr_("L", "N", &n, &incx, &m, As, &n, workssub, zs, &n,
            workssub+m+np, &lworkssub, &info, 1, 1);

    memset(Ts, 0.0f, n*p*sizeof(float));
    if (n > p)
    {
        length = n-p;
        slacpy_("A", &length, &p, Bs, &n, Ts, &n, 1);
        slacpy_("U", &p, &p, &Bs[n-p], &n, &Ts[n-p], &n, 1);
    }
    else
        slacpy_("U", &n, &n, &Bs[(p-n)*n], &n, &Ts[(p-n)*n], &n, 1);

    slag2d_(&m, &incx, xs, &m, x, &m, &info);
    slag2d_(&p, &incx, ys, &p, y, &p, &info);
    slag2d_(&n, &incx, zs, &n, z, &n, &info);
    lworkssub = lworkssub - m - np;
#ifdef MEASURETIME
    over = tic();
    time[2] = over-start;
#endif
    for (int iter = 0; iter < maxiter; iter++)
    {
#ifdef MEASURETIME
        start = tic();
#endif
        dcopy_(&p, y, &incx, r1, &incx);
        dgemv_("T", &n, &p, &one, B, &ldb, z, &incx, &rone, r1, &incx, 1);
        dcopy_(&n, d, &incx, r2, &incx);
        dgemv_("N", &n, &m, &rone, A, &lda, x, &incx, &one, r2, &incx, 1);
        dgemv_("N", &n, &p, &rone, B, &ldb, y, &incx, &one, r2, &incx, 1);
        dgemv_("T", &n, &m, &one, A, &lda, z, &incx, &zero, r3, &incx, 1);
        dlag2s_(&nsum, &incx, r1, &nsum, r1s, &nsum, &info);

        normr1 = dnrm2_(&p, r1, &incx);
        normr2 = dnrm2_(&n, r2, &incx);
        normr3 = dnrm2_(&m, r3, &incx);
        normx = dnrm2_(&m, x, &incx);
        normy = dnrm2_(&p, y, &incx);
        normz = dnrm2_(&n, z, &incx);
        if ((normr1 <= tol*(normy+normB*normz)) && (normr2 <= tol*(normd+normB*normy+normA*normx)) && (normr3 <= tol*(normA*normz)))
        {
            printf("%% iter = %d;\n", iter+1);
#ifdef MEASURETIME
            over = tic();
            time[3] = over-start+time[3];
            for (int i = 0; i < 5; i++)
                work[i] = time[i];
#endif
            return 0;
        }
#ifdef MEASURETIME
        over = tic();
        time[3] = over-start+time[3];
        start = tic();
#endif

        // Compute h1 = R^{-T}*r3 storing in r3s.
        strtrs_("U", "T", "N", &m, &incx, As, &n, r3s, &m, &info, 1, 1, 1);
        // Compute Q^T*r2s storing in r2s.
        sormqr_("L", "T", &n, &incx, &m, As, &n, workssub, r2s, &n,
                &workssub[m+np], &lworkssub, &info, 1, 1);
        // Compute Z*r1s storing in r1s.
        sormrq_("L", "N", &p, &incx, &np, &Bs[MAX(n-p, 0)], &n,
                &workssub[m], r1s, &p, &workssub[m+np], &lworkssub, &info,
                1, 1);
        // Compute g1 = (Z*r1s)_1-T_{11}^{T}*h1 storing in (r1s)_1.
        sgemv_("T", &m, &pnm, &ronef, Ts, &n, r3s, &incx, &onef, r1s,
                &incx, 1);
        // Compute g2 = T_{22}^{-T}*(Q^T*r2s)_2 storing i (r1s)_2.
        strtrs_("U", "N", "N", &ntemp, &incx, &Ts[(m+p-n)*n+m], &n,
                &r2s[m], &ntemp, &info, 1, 1, 1);
        // Store (Z*r1s)_2 in (r2s)_2.
        sswap_(&ntemp, &r1s[pnm], &incx, &r2s[m], &incx);
        // Compute h2 = T_{22}^{-T}*((Z*r1s)_2-g2-T_{12}^Th1)) storing in
        // r2s[m+1:end].
        saxpy_(&ntemp, &ronef, &r1s[pnm], &incx, &r2s[m], &incx);
        sgemv_("T", &m, &ntemp, &ronef, &Ts[pnm*n], &n, r3s, &incx, &onef,
                &r2s[m], &incx, 1);
        strtrs_("U", "T", "N", &ntemp, &incx, &Ts[(m+p-n)*n+m], &n,
                &r2s[m], &ntemp, &info, 1, 1, 1);
        // Compute Dx = R^{-1}*((Q^{T}*r2s)_1-T11*g1-T12*g2) storing in r2s.
        sgemv_("N", &m, &p, &ronef, Ts, &n, r1s, &incx, &onef, r2s, &incx,
                1);
        strtrs_("U", "N", "N", &m, &incx, As, &n, r2s, &m, &info, 1, 1, 1);
        // Exchange h1 and Dx.
        sswap_(&m, r2s, &incx, r3s, &incx);
        // Compute Dz = Q*h and Dy = Z^T*g.
        sormqr_("L", "N", &n, &incx, &m, As, &n, workssub, r2s, &n,
                &workssub[m+np], &lworkssub, &info, 1, 1);
        sormrq_("L", "T", &p, &incx, &np, &Bs[MAX(n-p, 0)], &n,
                &workssub[m], r1s, &p, &workssub[m+np], &lworkssub, &info,
                1, 1);
        // Update x, y, z.
        for (i = 0; i < m; i++)
            x[i] = x[i] + (double)r3s[i];
        for (i = 0; i < p; i++)
            y[i] = y[i] + (double)r1s[i];
        for (i = 0; i < n; i++)
            z[i] = z[i] - (double)r2s[i];
#ifdef MEASURETIME
        over = tic();
        time[4] = over-start+time[4];
        start = tic();
#endif
    }

    printf("%% Maxiter is too samll!\n");
    return 0;
}

int mpgls_old(int n, int m, int p, double *A, int lda, double *B, int ldb,
        double *D, double *X, double *Y, float *works, double *work,
        int lwork)
{
    float *As, *Bs, *Ds, *Xs, *Ys, *Rs, *workssub;
    int lworkssub, lworksub, info, incx = 1, i, np = MIN(n, p), ntemp;
    double one = 1.0, rone = -1.0;
    float onef = 1.0f, ronef = -1.0f;
    double *R, *worksub;

    As = works;
    Bs = As + n*m;
    Ds = Bs + n*p;
    Xs = Ds + n;
    Ys = Xs + m;
    Rs = Ys + p;
    workssub = Rs + n;
    lworkssub = lwork - n*m - n*p - n*2 - m - p;

    R = work;
    worksub = R + n;
    lworksub = lwork - n;

    dlag2s_(&n, &m, A, &n, As, &n, &info);
    dlag2s_(&n, &p, B, &n, Bs, &n, &info);
    dlag2s_(&n, &incx, D, &n, Ds, &n, &info);
    dlag2s_(&m, &incx, X, &m, Xs, &m, &info);
    dlag2s_(&p, &incx, Y, &p, Ys, &p, &info);
    sggglm_qz_(&n, &m, &p, As, &n, Bs, &n, Ds, Xs, Ys, workssub, &lworkssub,
            &info);

    slag2d_(&m, &incx, Xs, &m, X, &m, &info);
    slag2d_(&p, &incx, Ys, &p, Y, &p, &info);
    lworkssub = lworkssub - m - np;
    ntemp = n-m;
    for (int iter = 0; iter < 3000; iter++)
    {
        dcopy_(&n, D, &incx, R, &incx);
        dgemv_("N", &n, &m, &rone, A, &lda, X, &incx, &one, R, &incx, 1);
        dgemv_("N", &n, &p, &rone, B, &ldb, Y, &incx, &one, R, &incx, 1);
        dlag2s_(&n, &incx, R, &n, Rs, &n, &info);

        //sprintmat("rs", n, 1, Rs, n);
        sormqr_("L", "T", &n, &incx, &m, As, &n, workssub, Rs, &n,
                &workssub[m+np], &lworkssub, &info, 1, 1);
        strtrs_("U", "N", "N", &ntemp, &incx, &Bs[(m+p-n)*n+m], &n, &Rs[m],
                &ntemp, &info, 1, 1, 1);
        scopy_(&ntemp, &Rs[m], &incx, &Ys[m+p-n], &incx);
        for (i = 0; i < m+p-n; i++)
            Ys[i] = 0.0f;
        sgemv_("N", &m, &ntemp, &ronef, &Bs[(m+p-n)*n], &n, &Ys[m+p-n],
                &incx, &onef, Rs, &incx, 1);
        strtrs_("U", "N", "N", &m, &incx, As, &n, Rs, &m, &info, 1, 1, 1);
        scopy_(&m, Rs, &incx, Xs, &incx);
        sormqr_("L", "T", &p, &incx, &np, &Bs[MAX(n-p, 0)], &n,
                &workssub[m], Ys, &p, &workssub[m+np], &lworkssub, &info,
                1, 1);

        //sprintmat("xs", m, 1, Xs, m);
        //sprintmat("ys", p, 1, Ys, p);
        //dprintmat("xo", m, 1, X, m);
        //dprintmat("yo", p, 1, Y, p);
        for (i = 0; i < m; i++)
            X[i] = X[i] + (double)Xs[i];
        for (i = 0; i < p; i++)
            Y[i] = Y[i] + (double)Ys[i];
        //dprintmat("x", m, 1, X, m);
        //dprintmat("y", p, 1, Y, p);
    }

    return 0;
}
