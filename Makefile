OBJS	= akira.o cmatrix/cmatrix.o mmatrix.o
SOURCE	= src/akira.c cmatrix/cmatrix.c mmatrix/mmatrix.m
HEADER	= src/akira.h cmatrix/cmatrix.h mmatrix/mmatrix.h
OUT	= example
CC	= gcc
CXX	= clang
FLAGS	= -g -c
LFLAGS	= -lm -framework Metal -framework Foundation

all: example example_mnist

example: example.o $(OBJS)
	$(CXX) -g example.o $(OBJS) -o $(OUT) $(LFLAGS)

example.o: examples/example.c
	$(CC) $(FLAGS) examples/example.c

example_mnist: example_mnist.o $(OBJS)
	$(CXX) -g example_mnist.o $(OBJS) -o example_mnist $(LFLAGS)

example_mnist.o: examples/example_mnist.c
	$(CC) $(FLAGS) examples/example_mnist.c

akira.o: src/akira.c
	$(CC) $(FLAGS) src/akira.c 

cmatrix.o: cmatrix/cmatrix.c
	$(CC) $(FLAGS) cmatrix/cmatrix.c 

mmatrix.o: mmatrix/mmatrix.m mmatrix/mmatrix.h
	$(CXX) -ObjC -c mmatrix/mmatrix.m -o mmatrix.o

clean:
	rm -f example.o example_mnist.o  example_mnist $(OBJ) $(OUT) *.o

run: $(OUT)
	./$(OUT)
