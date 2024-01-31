#ifndef MY_WTIME_H
#define MY_WTIME_H

#define my_wtime my_wtime_
#define tic tic_
#define toc toc_

#ifdef __cplusplus
extern "C" {
#endif

double my_wtime();
double tic();
double toc();

#ifdef __cplusplus
}
#endif

#endif
