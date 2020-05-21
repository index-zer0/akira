#include "./cmatrix/cmatrix.h"

typedef struct nn {
    matrix *weights;
    matrix *bias;
    int hidden_num;
    double lr;
} _nn, *nn;

nn nn_constructor(const int, const int *);
void nn_delete(nn);
matrix run(nn, matrix);
void train(nn, matrix, matrix);