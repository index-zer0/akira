all: example

example: example.c brain.c
	gcc -g -o example example.c brain.c -lm

clean:
	rm -rf *.o
	rm -rf brain example