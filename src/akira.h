#define AKIRA_VERSION "0.1.0"
#define FILE_VERSION "0.1.0"

#include "../cmatrix/cmatrix.h"
#include "../mmatrix/mmatrix.h"

typedef enum {
    AKIRA_BACKEND_CPU = 0,
    AKIRA_BACKEND_METAL = 1
} akira_backend;

typedef struct nn {
    void **weights; // can be matrix* or mmatrix*
    void **bias;    // can be matrix* or mmatrix*
    int hidden_num;
    double lr;
    akira_backend backend;
    char notes[256];
} _nn, *nn;

nn nn_constructor(const int, const int *, akira_backend backend);
void nn_delete(nn);
void *run(nn, void *input); // returns matrix* or mmatrix*
void train(nn, void *training_input, void *training_output);
int save(nn , const char *, const char *, const char *);
nn load(const char *, int backend);
void nn_print_weights_biases(nn network);

extern int g_debug;