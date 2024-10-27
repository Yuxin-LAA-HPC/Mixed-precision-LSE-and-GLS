#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "../include/mpls_util.h"
#include "../include/mpls.h"
#include "../include/lapack.h"
#include "../include/my_wtime.h"

#define MEASURETIME 0
#undef MEASURETIME 0


int gmres_lse_twoside_scal(int n, int mo, int no, int po, double *Ao, int lda,
    double *Bo, int ldb, double *As, int ldas, double *Bs, int ldbs,
    double *T2s, double alpha, double *workssub,
    double *b, double *x, int restrt, int maxiter, int *num_iter,
    double tol, double *work, int lwork)
{
    // As and Bs store the GRQ factorization result of Ao and Bo.
    // no, mo, po is the dimension of Ao and Bo.
    // n = no+mo+po is the dimension of [I 0 Ao; 0 0 Bo; Ao^T Bo^T 0].
    double start, over, time[5];
    double *rtmp, *r, *V, *H, *cs, *sn, *e1, *s, *vcur, *vec_tmp, *addvec;
    double *w, *y, *worksub;
    double bnrm2, temp, para_scal, error, normA;
    double one = 1.0, zero = 0.0;
    int incx = 1, ldh, m, i, tempint;

    for (int j = 0; j < 5; j++)
        time[j] = 0.0;
    *num_iter = 0;

    m = restrt;
    rtmp = work;
    r = rtmp + n;
    V = r + n;
    H = V + n*(m+1);
    cs = H + (m+1)*m;
    sn = cs + m;
    e1 = sn + m;
    s = e1 + n;
    y = s + m+1;
    vcur = y + m;
    vec_tmp = vcur + n;
    addvec = vec_tmp + n;
    w = addvec + n;
    worksub = w + n;

    ldh = m+1;

#ifdef MEASURETIME
    start = tic();
#endif
    tempint = mo*no;
    normA = pow(dnrm2_(&tempint, Ao, &incx), 2);
    tempint = po*no;
    normA = normA + pow(dnrm2_(&tempint, Bo, &incx), 2);
    normA = sqrt(2.0*normA + (double)mo);
    dcopy_(&n, b, &incx, r, &incx);
    funcAxres_scal(n, mo, no, po, Ao, lda, Bo, ldb, alpha, x, r);//rtmp-A*x
#ifdef MEASURETIME
    over = tic();
    time[1] = over-start+time[1];
    start = tic();
#endif
    funcpretwoside_left_lse_scal(mo, no, po, As, ldas, Bs, ldbs, T2s, alpha,
        workssub, r, worksub);
#ifdef MEASURETIME
    over = tic();
    time[2] = over-start+time[2];
#endif

    bnrm2 = dnrm2_(&n, r, &incx);
    if (bnrm2 == 0.0) bnrm2 = 1.0;

    error = dnrm2_(&n, r, &incx)/bnrm2;
    if (error < tol)
    {
#ifdef MEASURETIME
        printf("%%GMRES time:");
        dprintmat("time_gmres", 5, 1, time, 5);
#endif
        return 0;
    }

    bnrm2 = normA*dnrm2_(&n, x, &incx) + dnrm2_(&n, b, &incx);
    e1[0] = 1.0;

    for (int iter = 0; iter < maxiter; iter++)
    {
#ifdef MEASURETIME
        start = tic();
#endif
        dcopy_(&n, b, &incx, r, &incx);
        funcAxres_scal(n, mo, no, po, Ao, lda, Bo, ldb, alpha, x, r);//rtmp-A*x
#ifdef MEASURETIME
        over = tic();
        time[1] = over-start+time[1];
        printf("Axres time:%.16f\n", over-start);
        start = tic();
#endif
        funcpretwoside_left_lse_scal(mo, no, po, As, ldas, Bs, ldbs, T2s, 
                alpha, workssub, r, worksub);
        temp = dnrm2_(&n, r, &incx);
        para_scal = 1.0/temp;
        dcopy_(&n, r, &incx, V, &incx);
        dscal_(&n, &para_scal, V, &incx);
        dcopy_(&n, e1, &incx, s, &incx);
        dscal_(&n, &temp, s, &incx);
#ifdef MEASURETIME
    over = tic();
    time[2] = over-start+time[2];
#endif
        for (i = 0; i < m; i++)
        {
#ifdef MEASURETIME
            start = tic();
#endif
            *num_iter = *num_iter + 1;
            dcopy_(&n, &V[i*n], &incx, vcur, &incx);
            funcpretwoside_right_lse_scal(mo, no, po, As, ldas, Bs, ldbs, 
                    T2s, alpha, workssub, vcur, worksub);
            funcAx_scal(n, mo, no, po, Ao, lda, Bo, ldb, alpha, vcur, w);//A*x
#ifdef MEASURETIME
            over = tic();
            time[1] = over-start+time[1];
            start = tic();
#endif
            funcpretwoside_left_lse_scal(mo, no, po, As, ldas, Bs, ldbs,
                T2s, alpha, workssub, w, worksub);
#ifdef MEASURETIME
            over = tic();
            time[2] = over-start+time[2];
            start = tic();
#endif

            for (int k = 0; k <= i; k++)
            {
                H[i*ldh+k] = ddot_(&n, w, &incx, &V[k*n], &incx);
                para_scal = -H[i*ldh+k];
                daxpy_(&n, &para_scal, &V[k*n], &incx, w, &incx);
            }
            H[i*ldh+i+1] = dnrm2_(&n, w, &incx);
            para_scal = 1.0/H[i*ldh+i+1];
            dcopy_(&n, w, &incx, &V[(i+1)*n], &incx);
            dscal_(&n, &para_scal, &V[(i+1)*n], &incx);

            for (int k = 0; k <= i-1; k++)
            {
                temp = cs[k]*H[i*ldh+k] + sn[k]*H[i*ldh+k+1];
                H[i*ldh+k+1] = -sn[k]*H[i*ldh+k] + cs[k]*H[i*ldh+k+1];
                H[i*ldh+k] = temp;
            }
            rot_givens(H[i*ldh+i], H[i*ldh+i+1], &cs[i], &sn[i]);
            temp = cs[i]*s[i];
            s[i+1] = -sn[i]*s[i];
            s[i] = temp;
            H[i*ldh+i] = cs[i]*H[i*ldh+i] + sn[i]*H[i*ldh+i+1];
            H[i*ldh+i+1] = 0.0;
#ifdef MEASURETIME
            over = tic();
            time[3] = over-start+time[3];
#endif
            error = fabs(s[i+1])/bnrm2;
            if (error <= tol)
            {
                i = i+1;
                dcopy_(&i, s, &incx, y, &incx);
                dtrsv_("U", "N", "N", &i, H, &ldh, y, &incx, 1, 1, 1);
                dgemv_("N", &n, &i, &one, V, &n, y, &incx, &zero, vec_tmp,
                    &incx, 1);
                funcpretwoside_right_lse_scal(mo, no, po, As, ldas, Bs,
                    ldbs, T2s, alpha, workssub, vec_tmp, worksub);
                daxpy_(&n, &one, vec_tmp, &incx, x, &incx);
                break;
            }

        }

    if (error <= tol) break;
        dcopy_(&m, s, &incx, y, &incx);
        dtrsv_("U", "N", "N", &m, H, &ldh, y, &incx, 1, 1, 1);
        dgemv_("N", &n, &m, &one, V, &n, y, &incx, &zero, vec_tmp,
                &incx, 1);
        funcpretwoside_right_lse_scal(mo, no, po, As, ldas, Bs, ldbs, T2s,
        alpha, workssub, vec_tmp, worksub);
        daxpy_(&n, &one, vec_tmp, &incx, x, &incx);

        dcopy_(&n, b, &incx, r, &incx);
        funcAxres_scal(n, mo, no, po, Ao, lda, Bo, ldb, alpha, x, r);//rtmp-A*x
        funcpretwoside_left_lse_scal(mo, no, po, As, ldas, Bs, ldbs, T2s, alpha,
        workssub, r, worksub);
    s[i+1] = dnrm2_(&n, r, &incx);
    error = s[i+1]/bnrm2;
    if (error <= tol) break;
    }

    if (error > tol)
    {
#ifdef MEASURETIME
        printf("%%GMRES time:");
        dprintmat("time_gmres", 5, 1, time, 5);
#endif
        printf("%% Not converge! Maxiter is too small!");
        return 1;
    }
#ifdef MEASURETIME
    printf("%%GMRES time:");
    dprintmat("time_gmres", 5, 1, time, 5);
#endif
    return 0;
}

