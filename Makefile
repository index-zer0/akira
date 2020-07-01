OBJS	= akira.o cmatrix/cmatrix.o
SOURCE	= src/akira.c cmatrix/cmatrix.c
HEADER	= src/akira.h cmatrix/cmatrix.h
OUT	= example
CC	= gcc
FLAGS	= -g -c # -O3
SHARED = -fPIC -shared
LFLAGS	= -lm
all: example example_mnist akira.so

example: example.o $(OBJS)
	$(CC) -g example.o $(OBJS) -o $(OUT) $(LFLAGS)

example.o: examples/example.c
	$(CC) $(FLAGS) examples/example.c

example_mnist: example_mnist.o $(OBJS)
	$(CC) -g example_mnist.o $(OBJS) -o example_mnist $(LFLAGS)

example_mnist.o: examples/example_mnist.c
	$(CC) $(FLAGS) examples/example_mnist.c

akira.o: src/akira.c
	$(CC) $(FLAGS) src/akira.c 

cmatrix.o: cmatrix/cmatrix.c
	$(CC) $(FLAGS) cmatrix/cmatrix.c 

# shared libraries

akira.so: cmatrix.so src/akira.c
	cc $(SHARED) -o akira.so src/akira.c -L. -l:./cmatrix.so

cmatrix.so: cmatrix/cmatrix.c
	cc $(SHARED) -o cmatrix.so cmatrix/cmatrix.c

clean:
	rm -f example.o example_mnist.o example_mnist $(OBJ) $(OUT) *.o *.so cmatrix/*.o cmatrix/*.so

run: $(OUT)
	./$(OUT)
