OBJS	= akira.o cmatrix.o mmatrix.o
SOURCE	= src/akira.c cmatrix/cmatrix.c mmatrix/mmatrix.m
HEADER	= src/akira.h cmatrix/cmatrix.h mmatrix/mmatrix.h
OUT	= example

# Build through xcrun so the Metal/Foundation headers resolve against the
# active Xcode SDK even when a non-Apple clang comes first on PATH.
CC	= xcrun clang
CXX	= xcrun clang

# apple-m1 is the baseline for every Apple Silicon part and still runs on the
# newer ones. Override to pin the host, e.g. `make CPU=apple-m2`.
CPU	?= apple-m1

# ACCELERATE_NEW_LAPACK selects the current (non-deprecated) CBLAS headers.
OPT	= -O3 -mcpu=$(CPU) -fno-math-errno -flto -DACCELERATE_NEW_LAPACK
FLAGS	= -g $(OPT) -c
LFLAGS	= -g $(OPT) -lm -framework Accelerate -framework Metal -framework Foundation

all: example example_mnist

example: example.o $(OBJS)
	$(CXX) example.o $(OBJS) -o $(OUT) $(LFLAGS)

example.o: examples/example.c $(HEADER)
	$(CC) $(FLAGS) examples/example.c -o example.o

example_mnist: example_mnist.o $(OBJS)
	$(CXX) example_mnist.o $(OBJS) -o example_mnist $(LFLAGS)

example_mnist.o: examples/example_mnist.c $(HEADER)
	$(CC) $(FLAGS) examples/example_mnist.c -o example_mnist.o

akira.o: src/akira.c $(HEADER)
	$(CC) $(FLAGS) src/akira.c -o akira.o

cmatrix.o: cmatrix/cmatrix.c cmatrix/cmatrix.h
	$(CC) $(FLAGS) cmatrix/cmatrix.c -o cmatrix.o

# ARC keeps the retained command buffers / encoders held across calls correct;
# the matrix struct is calloc'd so its strong buffer field starts out nil.
mmatrix.o: mmatrix/mmatrix.m $(HEADER)
	$(CXX) -ObjC -fobjc-arc $(FLAGS) mmatrix/mmatrix.m -o mmatrix.o

clean:
	rm -f example.o example_mnist.o example_mnist $(OBJS) $(OUT) *.o cmatrix/cmatrix.o

run: $(OUT)
	./$(OUT)

.PHONY: all clean run
