OBJS	= neural_network.o cmatrix/cmatrix.o
SOURCE	= neural_network.c cmatrix/cmatrix.c
HEADER	= neural_network.h cmatrix/cmatrix.h
OUT	= example
CC	= gcc
FLAGS	= -g -c
LFLAGS	= -lm

all: example example_mnist

example: example.o $(OBJS)
	$(CC) -g example.o $(OBJS) -o $(OUT) $(LFLAGS)

example.o: example.c
	$(CC) $(FLAGS) example.c

example_mnist: example_mnist.o $(OBJS)
	$(CC) -g example_mnist.o $(OBJS) -o example_mnist $(LFLAGS)

example_mnist.o: example_mnist.c
	$(CC) $(FLAGS) example_mnist.c

neural_network.o: neural_network.c
	$(CC) $(FLAGS) neural_network.c 

cmatrix.o: cmatrix/cmatrix.c
	$(CC) $(FLAGS) cmatrix/cmatrix.c 

clean:
	rm -f example.o example_mnist.o $(OBJS) $(OUT)

run: $(OUT)
	./$(OUT)