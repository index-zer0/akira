OBJS	= akira.o cmatrix.o
SOURCE	= src/akira.c cmatrix/cmatrix.c
HEADER	= src/akira.h cmatrix/cmatrix.h examples/stb_image.h
OUT	= example
CC	= gcc
FLAGS	= -pg
LFLAGS	= -lm -fopenmp

all: example example_mnist example_image

example: example.o $(OBJS)
	$(CC) $(FLAGS) example.o $(OBJS) -o $(OUT) $(LFLAGS)

example.o: examples/example.c
	$(CC) $(FLAGS) -c examples/example.c $(LFLAGS)

example_mnist: example_mnist.o $(OBJS)
	$(CC) $(FLAGS) example_mnist.o $(OBJS) -o example_mnist $(LFLAGS)

example_mnist.o: examples/example_mnist.c
	$(CC) $(FLAGS) -c examples/example_mnist.c $(LFLAGS)

example_image: example_image.o $(OBJS)
	$(CC) $(FLAGS) example_image.o $(OBJS) -o example_image $(LFLAGS)

example_image.o: examples/example_image.c
	$(CC) $(FLAGS) -c examples/example_image.c $(LFLAGS)


akira.o: src/akira.c
	$(CC) $(FLAGS) -c src/akira.c $(LFLAGS)

cmatrix.o: cmatrix/cmatrix.c
	$(CC) $(FLAGS) -c cmatrix/cmatrix.c $(LFLAGS)

clean:
	rm -f example.o example_mnist.o  example_mnist $(OBJ) $(OUT) *.o cmatrix.o

run: $(OUT)
	./$(OUT)
