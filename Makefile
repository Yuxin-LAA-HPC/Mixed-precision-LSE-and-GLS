include make.inc

.PHONY: all lib tests clean

all: lib tests
	ctags SRC/*.c TESTS/*.c include/*.h

lib:
	cd SRC && make && cd ..

tests: lib
	cd TESTS && make && cd ..

clean:
	cd SRC && make clean && cd ..
	cd TESTS && make clean && cd ..
