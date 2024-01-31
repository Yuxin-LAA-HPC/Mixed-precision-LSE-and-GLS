BLASLIB      = -lopenblas#/home/yxma2/LAPACK/lib/libopenblas.a
LAPACKLIB    = -llapack#/home/yxma2/LAPACK/liblapackm.a
#SCALAPACKLIB = $(HOME)/lib/libscalapack.a
INCLUDE      = -I../include
#OMPLIB       = /usr/local/Cellar/libomp/9.0.1/lib/libomp.a
LIBS         =  $(LAPACKLIB) $(BLASLIB) -lm -lpthread#-lprofiler -ltcmalloc_and_profiler

ARCH         = ar
ARCHFLAGS    = cr
RANLIB       = ranlib

DEFS         =
CC           = gcc -O2 -Wall
CFLAGS       = $(INCLUDE) $(DEFS) -O2 -Wall
#CFLAGS       = $(INCLUDE) $(DEFS) -g -Wall
FC           = gfortran -O2
#FFLAGS       = -cpp $(DEFS) -O2 \
#               -Wall -Wextra -fimplicit-none -finit-real=nan -fbounds-check
FFLAGS       = -cpp $(DEFS) -g -fbacktrace \
               -Wall -Wextra -fimplicit-none -finit-real=nan -fbounds-check
#NOOPTS       =
NOOPTS       = -g
LOADER       = $(FC)
LDFLAGS      = #-L/usr/local/opt/openblas/lib
