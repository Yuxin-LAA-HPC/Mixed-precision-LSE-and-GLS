include ../make.inc

lib = ../lib/libmpls.a

OBJS = my_wtime.o dlaord.o dqrt15.o dqrt13.o sgelsy_qz.o util.o mplsy.o mplse.o mplse_noaug.o mpgls.o sggglm_qz.o sgglse_qz.o

.PHONY: my_wtime.o dlaord.o dqrt15.o dqrt13.o sgelsy_qz.o util.o mplsy.o mplse.o mplse_noaug.o mpgls.o sggglm_qz.o sgglse_qz.o

all: $(lib)

dlaord.o:
	$(FC) -o dlaord.o -c dlaord.f

dqrt15.o:
	$(FC) -o dqrt15.o -c dqrt15.f

dqrt13.o:
	$(FC) -o dqrt13.o -c dqrt13.f

my_wtime.o:
	$(CC) -o my_wtime.o -c my_wtime.c

util.o:
	$(CC) -o util.o -c util.c

mplsy.o:
	$(CC) -o mplsy.o -c mplsy.c

mplse.o:
	$(CC) -o mplse.o -c mplse.c

mplse_noaug.o:
	$(CC) -o mplse_noaug.o -c mplse_noaug.c

mpgls.o:
	$(CC) -o mpgls.o -c mpgls.c

sgelsy_qz.o:
	$(FC) -o sgelsy_qz.o -c sgelsy_qz.f

sgglse_qz.o:
	$(FC) -o sgglse_qz.o -c sgglse_qz.f

sggglm_qz.o:
	$(FC) -o sggglm_qz.o -c sggglm_qz.f

$(lib): $(OBJS)
	ar cr $@ $(OBJS)

clean:
	rm -f *.o *.exe

.SUFFIXES: .o .c
.c.o:
	$(CC) $(CFLAGS) -c $<
