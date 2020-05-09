#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

typedef struct perceptron {
    double *weights;
    int size;
} perceptron;

double get_random(double min, double max) {
    if ((rand() % 2) == 0) {
        return ((min + 1) + (((double) rand()) / (double) RAND_MAX) * (max - (min + 1)));
    }
    return -1.0 * ((min + 1) + (((double) rand()) / (double) RAND_MAX) * (max - (min + 1)));
}

double activation(double num) {
    return num >= 0.0 ? 1.0 : -1.0;
}

perceptron* perceptron_constructor(int num) {
    int i;
    perceptron *p = malloc(sizeof(perceptron));
    p->size = num;
    p->weights = malloc(sizeof(double) * p->size + 1); // +1 for bias
    for (i = 0; i <= num; i++) {
        p->weights[i] = get_random(-1.0, 1.0);
    }
    return p;
}

void delete_perceptron(perceptron *p) {
    free(p->weights);
    free(p);
}

double perceptron_guess(perceptron *p, double *inputs, double bias) {
    int i;
    double sum = 0.0;
    for (i = 0; i < p->size; i++) {
        sum += inputs[i] * p->weights[i];
    }
    sum += bias * p->weights[p->size];
    return activation(sum);
}

void train(perceptron *p, double *inputs, double outputs, double lr, double bias) {
    int i;
    double guess = perceptron_guess(p, inputs, bias);
    double error = outputs - guess;
    for (i = 0; i < p->size; i++) {
        p->weights[i] += error * inputs[i] * lr;
    }
}

int main(void) {
    srand(time(0));
    int bias = 1.0;
    //xor
    double training_input[4 * 2] = { 1, 0,
                                     1, 1,
                                     0, 0,
                                     0, 1 };
    double training_output[4] = { 1, -1, -1, 1 };
    int i, j;
    perceptron *p = perceptron_constructor(2);
    for (j = 0; j < 1000; j++)
    for (i = 0; i < 4; i++) {
        //printf("Training: %lf xor %lf = %lf\n", *(training_input + i * 2), *(training_input + i * 2 + 1), *(training_output  + i * 1));
        train(p, training_input + i * 2, *(training_output  + i * 1), 0.1, bias);
    }
    for (i = 0; i < 4; i++) {
        printf("Guess: %lf xor %lf = %lf\n", *(training_input + i * 2), *(training_input + i * 2 + 1), perceptron_guess(p, training_input + i * 2, bias));
    }
    
    delete_perceptron(p);
}