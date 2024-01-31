#ifndef _MPLS_TYPES_H
#define _MPLS_TYPES_H

#include <complex.h>

#if 1
typedef float complex cplx;
typedef double complex zplx;
#else
typedef struct
{
    float re;
    float im;
} cplx;

typedef struct
{
    double re;
    double im;
} zplx;
#endif

typedef enum {ORTH_NORMAL = 0, ORTH_REORTH = 1} ORTH_T;

#endif
