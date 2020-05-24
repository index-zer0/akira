#define AKIRA_VERSION "0.1.0"
#define FILE_VERSION "0.1.0"

#include "../cmatrix/cmatrix.h"

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
int save(nn , const char *, const char *, const char *);
nn load(const char *);