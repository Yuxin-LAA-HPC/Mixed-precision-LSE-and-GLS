#ifdef unix
#include <sys/time.h>
#endif

#include <time.h>

#include "../include/my_wtime.h"

static double now;

double my_wtime()
{
#ifdef unix
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec + (double)tv.tv_usec * 1.0e-6;
#else
    return (double)clock() / (double)CLOCKS_PER_SEC;
#endif
}

double tic()
{
    return now = my_wtime();
}

double toc()
{
    return my_wtime()-now;
}
