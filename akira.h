#include "./cmatrix/cmatrix.h"

typedef struct nn {
    int input, hidden, output;
    // matrix weights_ih, weights_ho;
    // matrix bias_ih, bias_ho;
    matrix *weights;
    matrix *bias;
    int hidden_num;
    double lr;
} _nn, *nn;

nn nn_constructor(const int, const int *);
void nn_delete(nn);
matrix run(nn, matrix);
void train(nn, matrix, matrix);