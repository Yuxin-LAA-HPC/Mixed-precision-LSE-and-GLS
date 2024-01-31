include ../make.inc

LIB = ../lib/libmpls.a

.PHONY: all clean

all: test_lsy.o test_lsy.exe test_lse.o test_lse.exe test_gls.o test_gls.exe

test_lsy.o: test_lsy.c
	$(CC) -o test_lsy.o -c test_lsy.c

test_lsy.exe: test_lsy.o
	$(LOADER) $(LDFLAGS) -o $@ $^ $(LIB) $(LIBS)

test_lse.o: test_lse.c
	$(CC) -o test_lse.o -c test_lse.c

test_lse.exe: test_lse.o
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
