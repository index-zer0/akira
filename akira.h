#include "./cmatrix/cmatrix.h"

typedef struct nn {
    int input, hidden, output;
    matrix weights_ih, weights_ho;
    matrix bias_ih, bias_ho;
    double lr;
} _nn, *nn;

double sigmoid(double);
double sigmoid_derivative(double);
nn nn_constructor(int, int, int);
void nn_delete(nn);
matrix run(nn, matrix);
void train(nn, matrix, matrix);