#ifndef LAPACK_H
#define LAPACK_H


#include "mpls_types.h"


#define MIN(x, y) ((x) <= (y) ? (x) : (y))
#define MAX(x, y) ((x) >= (y) ? (x) : (y))


#ifdef __cplusplus
extern "C" {
#endif
extern float slamch_(char *cmach);
extern double dlamch_(char *CMACH);

extern double scabs1_(cplx *x);
extern double dcabs1_(zplx *x);

extern void dlatm1_(int *MODE, double *COND, int *IRSIGN, int *IDIST,
        int ISEED[], double D[], int *N, int *INFO);
extern void zlatm1_(int *MODE, double *COND, int *IRSIGN, int *IDIST,
        int ISEED[], zplx D[], int *N, int *INFO);

extern void dlaord_(char *JOB, int *N, double *X, int *INCX, int);

extern void dqrt13_(int *SCALE, int *M, int *N, double *A, int *LDA,
        double *NORMA, int *ISEED);

extern void dqrt15_(int *SCALE, int *RKSEL, int *M, int *N, int *NRHS,
        double *A, int *LDA, double *B, int *LDB, double *S, int *RANK,
        double *NORMA, double *NORMB, int *ISEED, double *WORK, int *LWORK);

extern void saxpy_(int *N, float *ALPHA, float X[], int *INCX, float Y[],
        int *INCY);
extern void daxpy_(int *N, double *ALPHA, double X[], int *INCX, double Y[],
        int *INCY);
extern void caxpy_(int *N, cplx *ALPHA, cplx X[], int *INCX, cplx Y[],
        int *INCY);
extern void zaxpy_(int *N, zplx *ALPHA, zplx X[], int *INCX, zplx Y[],
        int *INCY);

extern void drot_(int *N, double DX[], int *INCX, double DY[], int *INCY,
        double *C, double *S);
extern void srot_(int *N, float DX[], int *INCX, float DY[], int *INCY,
        float *C, float *S);

extern void drotm_(int *N, double DX[], int *INCX, double DY[], int *INCY,
        double DPARAM[]);
extern void srotm_(int *N, float DX[], int *INCX, float DY[], int *INCY,
        float DPARAM[]);
extern void zrot_(int *N, zplx CX[], int *INCX, zplx CY[], int *INCY,
        double *C, zplx *S);

extern double ddot_(int *N, double X[], int *INCZ, double Y[], int *INCY);
extern double zdotc_(int *N, zplx X[], int *INCZ, zplx Y[], int *INCY);
extern float sdot_(int *N, float X[], int *INCZ, float Y[], int *INCY);

extern void scopy_(int *N, float X[], int *INCX, float Y[], int *INCY);
extern void dcopy_(int *N, double X[], int *INCX, double Y[], int *INCY);
extern void ccopy_(int *N, cplx X[], int *INCX, cplx Y[], int *INCY);
extern void zcopy_(int *N, zplx X[], int *INCX, zplx Y[], int *INCY);

extern float snrm2_(int *N, float X[], int *INCX);
extern double dnrm2_(int *N, double X[], int *INCX);
extern float scnrm2_(int *N, cplx X[], int *INCX);
extern double dznrm2_(int *N, zplx X[], int *INCX);

extern void sgemm_(char *TRANSA, char *TRANSB, int *M, int *N, int *K,
        float *ALPHA, float A[], int *LDA, float B[], int *LDB, float *BETA,
        float C[], int *LDC, int, int);
extern void dgemm_(char *TRANSA, char *TRANSB, int *M, int *N, int *K,
        double *ALPHA, double A[], int *LDA, double B[], int *LDB,
        double *BETA, double C[], int *LDC, int, int);
extern void cgemm_(char *TRANSA, char *TRANSB, int *M, int *N, int *K,
        cplx *ALPHA, cplx A[], int *LDA, cplx B[], int *LDB, cplx *BETA,
        cplx C[], int *LDC, int, int);
extern void zgemm_(char *TRANSA, char *TRANSB, int *M, int *N, int *K,
        zplx *ALPHA, zplx A[], int *LDA, zplx B[], int *LDB, zplx *BETA,
        zplx C[], int *LDC, int, int);

extern void sgemv_(char *TRANS, int *M, int *N, float *ALPHA, float *A,
        int *LDA, float *X, int *INCX, float *BETA, float *Y, int *INCY,
        int);
extern void dgemv_(char *TRANS, int *M, int *N, double *ALPHA, double *A,
        int *LDA, double *X, int *INCX, double *BETA, double *Y, int *INCY,
        int);

extern void strmv_(char *UPLO, char *TRANS, char *DIAG, int *N, float *A,
        int *LDA, float *X, int *INCX, int, int, int);
extern void dtrmv_(char *UPLO, char *TRANS, char *DIAG, int *N, double *A,
        int *LDA, double *X, int *INCX, int, int, int);
extern void dtrmm_(char *SIDE, char *UPLO, char *TRANSA, char *DIAG, int *M,
        int *N, double *ALPHA, double A[], int *LDA, double B[], int *LDB,
        int, int, int, int);

extern void sgeqrf_(int *M, int *N, float A[], int *LDA, float TAU[],
        float WORK[], int *LWORK, int *INFO);
extern void dgeqrf_(int *M, int *N, double A[], int *LDA, double TAU[],
        double WORK[], int *LWORK, int *INFO);
extern void cgeqrf_(int *M, int *N, cplx A[], int *LDA, cplx TAU[],
        cplx WORK[], int *LWORK, int *INFO);
extern void zgeqrf_(int *M, int *N, zplx A[], int *LDA, zplx TAU[],
        zplx WORK[], int *LWORK, int *INFO);

extern void sgelqf_(int *M, int *N, float A[], int *LDA, float TAU[],
        float WORK[], int *LWORK, int *INFO);
extern void dgelqf_(int *M, int *N, double A[], int *LDA, double TAU[],
        double WORK[], int *LWORK, int *INFO);
extern void cgelqf_(int *M, int *N, cplx A[], int *LDA, cplx TAU[],
        cplx WORK[], int *LWORK, int *INFO);
extern void zgelqf_(int *M, int *N, zplx A[], int *LDA, zplx TAU[],
        zplx WORK[], int *LWORK, int *INFO);

extern void slacpy_(char *UPLO, int *M, int *N, float A[], int *LDA,
        float B[], int *LDB, int);
extern void dlacpy_(char *UPLO, int *M, int *N, double A[], int *LDA,
        double B[], int *LDB, int);
extern void clacpy_(char *UPLO, int *M, int *N, cplx A[], int *LDA,
        cplx B[], int *LDB, int);
extern void zlacpy_(char *UPLO, int *M, int *N, zplx A[], int *LDA,
        zplx B[], int *LDB, int);

extern void slaset_(char *UPLO, int *M, int *N, float *ALPHA, float *BETA,
        float A[], int *LDA, int);
extern void dlaset_(char *UPLO, int *M, int *N, double *ALPHA, double *BETA,
        double A[], int *LDA, int);
extern void claset_(char *UPLO, int *M, int *N, cplx *ALPHA, cplx *BETA,
        cplx A[], int *LDA, int);
extern void zlaset_(char *UPLO, int *M, int *N, zplx *ALPHA, zplx *BETA,
        zplx A[], int *LDA, int);

extern void sormqr_(char *SIDE, char *TRANS, int *M, int *N, int *K,
        float A[], int *LDA, float TAU[], float C[], int *LDC, float WORK[],
        int *LWORK, int *INFO, int, int);
extern void dormqr_(char *SIDE, char *TRANS, int *M, int *N, int *K,
        double A[], int *LDA, double TAU[], double C[], int *LDC,
        double WORK[], int *LWORK, int *INFO, int, int);
extern void cunmqr_(char *SIDE, char *TRANS, int *M, int *N, int *K,
        cplx A[], int *LDA, cplx TAU[], cplx C[], int *LDC, cplx WORK[],
        int *LWORK, int *INFO, int, int);
extern void zunmqr_(char *SIDE, char *TRANS, int *M, int *N, int *K,
        zplx A[], int *LDA, zplx TAU[], zplx C[], int *LDC, zplx WORK[],
        int *LWORK, int *INFO, int, int);

extern void sormlq_(char *SIDE, char *TRANS, int *M, int *N, int *K,
        float A[], int *LDA, float TAU[], float C[], int *LDC, float WORK[],
        int *LWORK, int *INFO, int, int);
extern void dormlq_(char *SIDE, char *TRANS, int *M, int *N, int *K,
        double A[], int *LDA, double TAU[], double C[], int *LDC,
        double WORK[], int *LWORK, int *INFO, int, int);
extern void cunmlq_(char *SIDE, char *TRANS, int *M, int *N, int *K,
        cplx A[], int *LDA, cplx TAU[], cplx C[], int *LDC, cplx WORK[],
        int *LWORK, int *INFO, int, int);
extern void zunmlq_(char *SIDE, char *TRANS, int *M, int *N, int *K,
        zplx A[], int *LDA, zplx TAU[], zplx C[], int *LDC, zplx WORK[],
        int *LWORK, int *INFO, int, int);

extern void sormrq_(char *SIDE, char *TRANS, int *M, int *N, int *K,
        float A[], int *LDA, float TAU[], float C[], int *LDC, float WORK[],
        int *LWORK, int *INFO, int, int);

extern void ssyev_(char *JOBZ, char *UPLO, int *N, float A[], int *LDA,
        float W[], float WORK[], int *LWORK, int *INFO, int, int);
extern void dsyev_(char *JOBZ, char *UPLO, int *N, double A[], int *LDA,
        double W[], double WORK[], int *LWORK, int *INFO, int, int);
extern void cheev_(char *JOBZ, char *UPLO, int *N, cplx A[], int *LDA,
        float W[], cplx WORK[], int *LWORK, float RWORK[], int *INFO, int,
        int);
extern void zheev_(char *JOBZ, char *UPLO, int *N, zplx A[], int *LDA,
        double W[], zplx WORK[], int *LWORK, double RWORK[], int *INFO, int,
        int);

extern void sscal_(int *N, float *DA, float DX[], int *INCX);
extern void dscal_(int *N, double *DA, double DX[], int *INCX);
extern void cscal_(int *N, cplx *DA, cplx DX[], int *INCX);
extern void zscal_(int *N, zplx *DA, zplx DX[], int *INCX);

extern void sgesvj_(char *JOBA, char *JOBU, char *JOBV, int *M, int *N,
        float A[], int *LDA, float SVA[], int *MV, float V[], int *LDV,
        float WORK[], int *LWORK, int *INFO, int, int, int);
extern void dgesvj_(char *JOBA, char *JOBU, char *JOBV, int *M, int *N,
        double A[], int *LDA, double SVA[], int *MV, double V[], int *LDV,
        double WORK[], int *LWORK, int *INFO, int, int, int);
extern void cgesvj_(char *JOBA, char *JOBU, char *JOBV, int *M, int *N,
        cplx A[], int *LDA, float SVA[], int *MV, cplx V[], int *LDV,
        cplx CWORK[], int *LWORK, float RWORK[], int *LRWORK, int *INFO,
        int, int, int);
extern void zgesvj_(char *JOBA, char *JOBU, char *JOBV, int *M, int *N,
        zplx A[], int *LDA, double SVA[], int *MV, zplx V[], int *LDV,
        zplx CWORK[], int *LWORK, double RWORK[], int *LRWORK, int *INFO,
        int, int, int);

extern void sgejsv_(char *JOBA, char *JOBU, char *JOBV, char *JOBR,
        char *JOBT, char *JOBP, int *M, int *N, float A[], int *LDA,
        float SVA[], float U[], int *LDU, float V[], int *LDV,
        float WORK[], int *LWORK, int IWORK[], int *info,
        int, int, int, int, int);
extern void dgejsv_(char *JOBA, char *JOBU, char *JOBV, char *JOBR,
        char *JOBT, char *JOBP, int *M, int *N, double A[], int *LDA,
        double SVA[], double U[], int *LDU, double V[], int *LDV,
        double WORK[], int *LWORK, int IWORK[], int *info, int, int, int,
        int, int);
extern void cgejsv_(char *JOBA, char *JOBU, char *JOBV, char *JOBR,
        char *JOBT, char *JOBP, int *M, int *N, cplx A[], int *LDA,
        float SVA[], cplx U[], int *LDU, cplx V[], int *LDV,
        cplx CWORK[], int *LWORK, float RWORK[], int *LRWORK, int IWORK[],
        int *info, int, int, int, int, int);
extern void zgejsv_(char *JOBA, char *JOBU, char *JOBV, char *JOBR,
        char *JOBT, char *JOBP, int *M, int *N, zplx A[], int *LDA,
        double SVA[], zplx U[], int *LDU, zplx V[], int *LDV,
        zplx CWORK[], int *LWORK, double RWORK[], int *LRWORK, int IWORK[],
        int *info, int, int, int, int, int);

extern void sgesdd_(char *JOBZ, int *M, int *N, float A[], int *LDA,
        float S[], float U[], int *LDU, float VT[], int *LDVT,
        float WORK[], int *LWORK, int *INFO, int);
extern void sgesvd_(char *JOBU, char *JOBV, int *M, int *N, float A[],
        int *LDA,
        float S[], float U[], int *LDU, float VT[], int *LDVT,
        float WORK[], int *LWORK, int *INFO, int, int);

extern void strsm_(char *SIDE, char *UPLO, char *TRANSA, char *DIAG, int *M,
        int *N, float *ALPHA, float A[], int *LDA, float B[], int *LDB,
        int, int, int, int);
extern void dtrsm_(char *SIDE, char *UPLO, char *TRANSA, char *DIAG, int *M,
        int *N, double *ALPHA, double A[], int *LDA, double B[], int *LDB,
        int, int, int, int);

extern void strtrs_(char *UPLO, char *TRANS, char *DIAG, int *N, int *NRHS,
        float *A, int *LDA, float *B, int *LDB, int *INFO, int, int, int);
extern void dtrtrs_(char *UPLO, char *TRANS, char *DIAG, int *N, int *NRHS,
        double *A, int *LDA, double *B, int *LDB, int *INFO, int, int, int);

extern void dpotrf_(char *UPLC, int *N, double A[], int *LDA, int *INFO, int);

extern void dsyrk_(char *UPLO, char *TRANS, int *N, int *K, double *ALPHA,
        double A[], int *LDA, double *BETA, double C[], int *LDC, int, int);
extern void cherk_(char *UPLO, char *TRANS, int *N, int *K, cplx *ALPHA,
        cplx A[], int *LDA, cplx *BETA, cplx C[], int *LDC, int, int);
extern void zsyrk_(char *UPLO, char *TRANS, int *N, int *K, zplx *ALPHA,
        zplx A[], int *LDA, zplx *BETA, zplx C[], int *LDC, int, int);

extern void sgeqp3_(int *M, int *N, float A[], int *LDA, int JPVT[],
        float TAU[], float WORK[], int *lwork, int *info);
extern void dgeqp3_(int *M, int *N, double A[], int *LDA, int JPVT[],
        double TAU[], double WORK[], int *lwork, int *info);
extern void cgeqp3_(int *M, int *N, cplx A[], int *LDA, int JPVT[],
        cplx TAU[], cplx WORK[], int *lwork, float *RWORKF, int *info);

extern void sswap_(int *N, float SX[], int *INCX, float SY[], int *INCY);
extern void dswap_(int *N, double SX[], int *INCX, double SY[], int *INCY);

extern int isamax_(int *N, float SX[], int *INXC);
extern int idamax_(int *N, double SX[], int *INXC);
extern int icamax_(int *N, cplx SX[], int *INXC);
extern int izamax_(int *N, zplx SX[], int *INXC);

extern int isamin_(int *N, float SX[], int *INXC);
extern int idamin_(int *N, double SX[], int *INXC);

extern void ssyrk_(char *UPLO, char *TRANS, int *N, int *K, float *ALPHA,
        float A[], int *LDA, float *BETA, float C[], int *LDC, int, int);

extern void dorgqr_(int *M, int *N, int *K, double A[], int *LDA,
        double TAU[], double WORK[], int *LWORK, int *INFO);
extern void zungqr_(int *M, int *N, int *K, zplx A[], int *LDA,
        zplx TAU[], zplx WORK[], int *LWORK, int *INFO);

extern void slag2d_(int *M, int *N, float *As, int *LDAS, double *A,
        int *LDA, int *INFO);
extern void dlag2s_(int *M, int *N, double *Ad, int *LDAD, float *A,
        int *LDA, int *INFO);

extern void sgelsy_(int *M, int *N, int *NRHS, float *A, int *LDA, float *B,
        int *LDB, int *JPVT, float *RCOND, int *RANK, float *WORK, int *LWORK,
        int *INFO);
extern void dgelsy_(int *M, int *N, int *NRHS, double *A, int *LDA, double *B,
        int *LDB, int *JPVT, double *RCOND, int *RANK, double *WORK, int *LWORK,
        int *INFO);

extern void sgelsy_dqz_(int *M, int *N, int *NRHS, float *A, int *LDA,
        float *B, int *LDB, int *JPVT, float *RCOND, int *RANK, double *Q1,
        double *Zt, double *AD, double *WORKD, float *WORK, int *LWORK,
        int *INFO);
extern void sgelsy_qz_(int *M, int *N, int *NRHS, float *A, int *LDA,
        float *B, int *LDB, int *JPVT, float *RCOND, int *RANK, float *Q1,
        float *Zt, float *WORK, int *LWORK, int *INFO);

extern void sgglse_(int *M, int *N, int *P, float *A, int *LDA, float *B,
        int *LDB, float *C, float *D, float *X, float *WORK, int *LWORK,
        int *INFO);
extern void dgglse_(int *M, int *N, int *P, double *A, int *LDA, double *B,
        int *LDB, double *C, double *D, double *X, double *WORK, int *LWORK,
        int *INFO);

extern void sggglm_(int *N, int *M, int *P, float *A, int *LDA, float *B,
        int *LDB, float *D, float *X, float *Y, float *WORK, int *LWORK,
        int *INFO);
extern void dggglm_(int *N, int *M, int *P, double *A, int *LDA, double *B,
        int *LDB, double *D, double *X, double *Y, double *WORK, int *LWORK,
        int *INFO);

extern void sgglse_qz_(int *M, int *N, int *P, float *A, int *LDA, float *B,
        int *LDB, float *C, float *D, float *X, float *WORK, int *LWORK,
        int *INFO);

extern void sggglm_qz_(int *N, int *M, int *P, float *A, int *LDA, float *B,
        int *LDB, float *D, float *X, float *Y, float *WORK, int *LWORK,
        int *INFO);
#ifdef __cplusplus
}
#endif


#if 0
extern double dnrm2_(int *N, double X[], int *INCX);
extern void drot_(int *N, double X[], int *INCX, double Y[], int *INCY,
        double *C, double *S);
extern void dscal_(int *N, double *ALPHA, double X[], int *INCX);
extern int idamax_(int *N, double X[], int *INCX);
/* This one is not portable.
extern zplx zdotc_(int *N, zplx ZX[], int *INCX, zplx ZY[],
        int *INCY); */
extern void myzdotc_(zplx *RESULT, int *N, zplx ZX[], int *INCX,
        zplx ZY[], int *INCY);
extern void zdscal_(int *N, double *DA, zplx ZX[], int *INCX);


extern void dgeev_(char *JOBVL, char *JOBVR, int *N, double A[], int *LDA,
        double WR[], double WI[], double VL[], int *LDVL, double VR[],
        int *LDVR, double WORK[], int *LWORK, int *INFO, int, int);
extern void dgehrd_(int *N, int *ILO, int *IHI, double A[], int *LDA,
        double TAU[], double WORK[], int *LWORK, int *INFO);
extern void dgesv_(int *N, int *NRHS, double A[], int *LDA, int IPIV[],
        double B[], int *LDB, int *INFO);
extern void dgesvx_(char *FACT, char *TRANS, int *N, int *NRHS, double A[],
        int *LDA, double AF[], int *LDAF, int IPIV[], char *EQUED, double R[],
        double C[], double B[], int *LDB, double X[], int *LDX, double *RCOND,
        double FERR[], double BERR[], double WORK[], int IWORK[], int *INFO,
        int, int, int);
extern void dhseqr_(char *JOB, char *COMPZ, int *N, int *ILO, int *IHI,
        double H[], int *LDH, double WR[], double WI[], double Z[], int *LDZ,
        double WORK[], int *LWORK, int *INFO, int, int);
extern void dlacpy_(char *UPLO, int *M, int *N, double A[], int *LDA,
        double B[], int *LDB, int);
extern double dlapy2_(double *X, double *Y);
extern void dlartg_(double *F, double *G, double *CS, double *SN, double *R);
extern void dlaset_(char *UPLO, int *M, int *N, double *ALPHA, double *BETA,
        double A[], int *LDA, int);
extern void dorghr_(int *N, int *ILO, int *IHI, double A[], int *LDA,
        double TAU[], double WORK[], int *LWORK, int *INFO);
extern void dtrevc_(char *SIDE, char *HOWMNY, long SELECT[], int *N,
        double T[], int *LDT, double VL[], int *LDVL, double VR[], int *LDVR,
        int *MM, int *M, double WORK[], int *INFO, int, int);
extern void dtrevc3_(char *SIDE, char *HOWMNY, long SELECT[], int *N,
        double T[], int *LDT, double VL[], int *LDVL, double VR[], int *LDVR,
        int *MM, int *M, double WORK[], int *LWORK, int *INFO, int, int);
#endif


#endif
