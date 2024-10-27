include ../make.inc

lib = ../lib/libmpls.a

OBJS = my_wtime.o gmres_lse.o gmres_lse_twoside.o gmres_gls_twoside.o dlaord.o dqrt15.o dqrt13.o sgelsy_qz.o util.o mplse.o mplse_gmres.o mpgls_gmres_twoside.o mplse_gmres_twoside.o mpgls.o sggglm_qz.o sgglse_qz.o sggglm_time.o dggglm_time.o sggqrf_time.o dggqrf_time.o sggqrf_ql.o sggrqf_ql.o

.PHONY: my_wtime.o gmres_lse.o gmres_lse_twoside.o gmres_gls_twoside.o dlaord.o dqrt15.o dqrt13.o sgelsy_qz.o util.o mplsy.o mplse.o mplse_gmres.o mplse_gmres_twoside.o mplse_gmres_twoside.o mplse_noaug.o mpgls.o sggglm_qz.o sgglse_qz.o sggglm_time.o dggglm_time.o sggqrf_time.o dggqrf_time.o sggqrf_ql.o sggrqf_ql.o

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

gmres_lse.o:
	$(CC) -o gmres_lse.o -c gmres_lse.c

gmres_lse_twoside.o:
	$(CC) -o gmres_lse_twoside.o -c gmres_lse_twoside.c

gmres_gls_twoside.o:
	$(CC) -o gmres_gls_twoside.o -c gmres_gls_twoside.c

mplse.o:
	$(CC) -o mplse.o -c mplse.c

mplse_gmres.o:
	$(CC) -o mplse_gmres.o -c mplse_gmres.c

mplse_gmres_twoside.o:
	$(CC) -o mplse_gmres_twoside.o -c mplse_gmres_twoside.c

mpgls_gmres_twoside.o:
	$(CC) -o mpgls_gmres_twoside.o -c mpgls_gmres_twoside.c

mpgls.o:
	$(CC) -o mpgls.o -c mpgls.c

sgelsy_qz.o:
	$(FC) -o sgelsy_qz.o -c sgelsy_qz.f

sgglse_qz.o:
	$(FC) -o sgglse_qz.o -c sgglse_qz.f

sggglm_qz.o:
	$(FC) -o sggglm_qz.o -c sggglm_qz.f

sggglm_time.o:
	$(FC) -o sggglm_time.o -c sggglm_time.f

dggglm_time.o:
	$(FC) -o dggglm_time.o -c dggglm_time.f

sggqrf_time.o:
	$(FC) -o sggqrf_time.o -c sggqrf_time.f

dggqrf_time.o:
	$(FC) -o dggqrf_time.o -c dggqrf_time.f

sggqrf_ql.o:
	$(FC) -o sggqrf_ql.o -c sggqrf_ql.f

sggrqf_ql.o:
	$(FC) -o sggrqf_ql.o -c sggrqf_ql.f

$(lib): $(OBJS)
	ar cr $@ $(OBJS)

clean:
	rm -f *.o *.exe

.SUFFIXES: .o .c
.c.o:
	$(CC) $(CFLAGS) -c $<
