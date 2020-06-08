OBJS	= akira.o cmatrix/cmatrix.o
SOURCE	= src/akira.c cmatrix/cmatrix.c
HEADER	= src/akira.h cmatrix/cmatrix.h
OUT	= example
CC	= gcc
FLAGS	= -g -c
LFLAGS	= -lm

all: example example_mnist example_face_recognition

example: example.o $(OBJS)
	$(CC) -g example.o $(OBJS) -o $(OUT) $(LFLAGS)

example.o: examples/example.c
	$(CC) $(FLAGS) examples/example.c

example_mnist: example_mnist.o $(OBJS)
	$(CC) -g example_mnist.o $(OBJS) -o example_mnist $(LFLAGS)

example_mnist.o: examples/example_mnist.c
	$(CC) $(FLAGS) examples/example_mnist.c

example_face_recognition: example_face_recognition.o $(OBJS)
	$(CC) -g example_face_recognition.o $(OBJS) -o example_face_recognition $(LFLAGS)

example_face_recognition.o: examples/recognition/example_face_recognition.c
	$(CC) $(FLAGS) examples/recognition/example_face_recognition.c

akira.o: src/akira.c
	$(CC) $(FLAGS) src/akira.c 

cmatrix.o: cmatrix/cmatrix.c
	$(CC) $(FLAGS) cmatrix/cmatrix.c 

clean:
	rm -f example.o example_mnist.o example_face_recognition.o example_mnist example_face_recognition $(OBJ) $(OUT) *.o

run: $(OUT)
	./$(OUT)
