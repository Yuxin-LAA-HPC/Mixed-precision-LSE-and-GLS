#include <stdlib.h>
#include <stdio.h>
#include "../include/lapack.h"
#include "../include/mpls_util.h"
#include "../include/mpls.h"

int mplsy(int m, int n, int nrhs, double *A, int lda, double *B, int ldb,
        double rcond, double *work, float *works, int lwork)
{
    int *jpvt = (int *)calloc(n, sizeof(int));
    int info, lworksub, lworkssub;
    float *As, *Bs, *Ds, *Zts, *Q1s, *workssub;
    float rconds = (double)rcond;
    double *R, *X, *worksub;
    double one = 1.0, rone = -1.0, zero = 0.0;
    float onef = 1.0f, ronef = -1.0f, zerof = 0.0f;
    int incx = 1, i, j, rank;

    As = works;
    Bs = As + m*n;
    Ds = Bs + m*nrhs;
    Q1s = Ds + n*nrhs;
    Zts = Q1s + m*n;
    workssub = Zts + n*n;
    lworkssub = lwork - 2*m*n - m*nrhs - n*n - n*nrhs;

    R = work;
    X = R + m*nrhs;
    worksub = X + n*nrhs;
    lworksub = lwork - m*nrhs - n*nrhs;

    dlag2s_(&m, &n, A, &lda, As, &m, &info);
    dlag2s_(&m, &nrhs, B, &ldb, Bs, &m, &info);
    // Compute ls in single precision and store Q1, Zt, T.
    sgelsy_qz_(&m, &n, &nrhs, As, &m, Bs, &m, jpvt, &rconds, &rank, Q1s,
            Zts, workssub, &lworksub, &info);
    //sgelsy_(&m, &n, &nrhs, As, &m, Bs, &m, jpvt, &rconds, &rank,
    //        workssub, &lworksub, &info);

    // Copy Bs to X;
    //slag2d_(&m, &n, Q1s, &m, Q1, &m, &info);
    //slag2d_(&n, &n, Zts, &n, Zt, &n, &info);
    slag2d_(&n, &nrhs, Bs, &m, X, &n, &info);
    //slag2d_(&m, &n, As, &m, T, &m, &info);
    //dprintmat("Q1", m, rank, Q1, m);
    //dprintmat("Zt", n, rank, Zt, n);
    //sprintmat("Zts", n, rank, Zts, n);
    for (int iter = 0; iter < 2; iter++)
    {
        // R = B;
        dlacpy_("A", &m, &nrhs, B, &ldb, R, &m, 1);
        // R = AX - B;
        dgemm_("N", "N", &m, &nrhs, &n, &one, A, &lda, X, &n, &rone, R, &m,
                1, 1);
        dlag2s_(&m, &nrhs, R, &m, Bs, &m, &info);
        // Solve AD = R;
        sgemm_("T", "N", &rank, &nrhs, &m, &onef, Q1s, &m, Bs, &m, &zerof, Ds,
                &rank, 1, 1);
        strsm_("L", "U", "N", "N", &rank, &nrhs, &onef, As, &m, Ds, &rank,
                1, 1, 1, 1);
        sgemm_("N", "N", &n, &nrhs, &rank, &onef, Zts, &n, Ds, &rank,
                &zerof, Bs, &m, 1, 1);
        for (j = 0; j < nrhs; j++)
        {
            for (i = 0; i < n; i++)
                worksub[jpvt[i]-1] = (double)Bs[j*m+i];
            daxpy_(&n, &rone, worksub, &incx, &X[j*n], &incx);
        }

    }
    dlacpy_("A", &n, &nrhs, X, &n, B, &m, 1);


    free(jpvt);
    return 0;
}
