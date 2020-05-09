OBJS	= example.o neural_network.o cmatrix/cmatrix.o
SOURCE	= example.c neural_network.c cmatrix/cmatrix.c
HEADER	= neural_network.h cmatrix/cmatrix.h
OUT	= example
CC	= gcc
FLAGS	= -g -c -Wall
LFLAGS	= -lm

all: $(OBJS)
	$(CC) -g $(OBJS) -o $(OUT) $(LFLAGS)

example.o: example.c
	$(CC) $(FLAGS) example.c 

neural_network.o: neural_network.c
	$(CC) $(FLAGS) neural_network.c 

cmatrix.o: cmatrix/cmatrix.c
	$(CC) $(FLAGS) cmatrix/cmatrix.c 

clean:
	rm -f $(OBJS) $(OUT)

run: $(OUT)
	./$(OUT)