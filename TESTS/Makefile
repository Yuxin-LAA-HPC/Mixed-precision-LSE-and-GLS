include ../make.inc

LIB = ../lib/libmpls.a

.PHONY: all clean

all: test_lse_gmres.o test_lse_gmres.exe test_gls.o test_gls.exe

test_lse_gmres.o: test_lse_gmres.c
	$(CC) -o test_lse_gmres.o -c test_lse_gmres.c

test_lse_gmres.exe: test_lse_gmres.o
	$(LOADER) $(LDFLAGS) -o $@ $^ $(LIB) $(LIBS)

test_gls.o: test_gls.c
	$(CC) -o test_gls.o -c test_gls.c

test_gls.exe: test_gls.o
	$(LOADER) $(LDFLAGS) -o $@ $^ $(LIB) $(LIBS)

clean:
	rm -f *.o *.exe

.SUFFIXES: .o .c
.c.o:
	$(CC) $(CFLAGS) -c $<
