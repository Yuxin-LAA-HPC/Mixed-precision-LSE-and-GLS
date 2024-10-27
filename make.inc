BLASLIB      = -lopenblas
LAPACKLIB    = -llapack
BLASLIB      = -lopenblas
INCLUDE      = -I../include
LIBS         =  $(LAPACKLIB) $(BLASLIB) -lm -lpthread

ARCH         = ar
ARCHFLAGS    = cr
RANLIB       = ranlib

DEFS         =
CC           = gcc -O2 -Wall
CFLAGS       = $(INCLUDE) $(DEFS) -O2 -Wall
FC           = gfortran -O2
FFLAGS       = -cpp $(DEFS) -g -fbacktrace \
               -Wall -Wextra -fimplicit-none -finit-real=nan -fbounds-check
NOOPTS       = -g
LOADER       = $(FC)
LDFLAGS      = #-L/usr/local/opt/openblas/lib
