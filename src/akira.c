#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <unistd.h>
#include "akira.h"

int g_debug = 0;

int save_0_1_0(nn, const char *, const char *);
nn load_0_1_0(const char *, int);

/* Scratch buffers for the activations and error signals of one sample. They
 * are sized once from the weight shapes and then reused, so a training step
 * performs no allocation at all. */
typedef struct {
    void **hidden;      // hidden_num activations, matrix* or mmatrix*
    void **error;       // hidden_num error signals, same shapes as hidden
    void *output;       // output activation
    void *output_error; // output error signal
} workspace;

static void ws_delete(nn network) {
    workspace *ws = (workspace *)network->ws;
    int i;
    if (!ws) {
        return;
    }
    for (i = 0; i < network->hidden_num; i++) {
        if (network->backend == AKIRA_BACKEND_CPU) {
            matrix_delete((matrix)ws->hidden[i]);
            matrix_delete((matrix)ws->error[i]);
        } else {
            mmatrix_delete((mmatrix)ws->hidden[i]);
            mmatrix_delete((mmatrix)ws->error[i]);
        }
    }
    if (network->backend == AKIRA_BACKEND_CPU) {
        matrix_delete((matrix)ws->output);
        matrix_delete((matrix)ws->output_error);
    } else {
        mmatrix_delete((mmatrix)ws->output);
        mmatrix_delete((mmatrix)ws->output_error);
    }
    free(ws->hidden);
    free(ws->error);
    free(ws);
    network->ws = NULL;
}

static void ws_init(nn network) {
    workspace *ws = malloc(sizeof(workspace));
    const int n = network->hidden_num;
    int i, rows;
    ws->hidden = malloc(sizeof(void *) * n);
    ws->error = malloc(sizeof(void *) * n);
    for (i = 0; i < n; i++) {
        if (network->backend == AKIRA_BACKEND_CPU) {
            rows = ((matrix)network->weights[i])->rows;
            ws->hidden[i] = matrix_constructor(rows, 1);
            ws->error[i] = matrix_constructor(rows, 1);
        } else {
            rows = mmatrix_get_rows((mmatrix)network->weights[i]);
            ws->hidden[i] = mmatrix_constructor(rows, 1);
            ws->error[i] = mmatrix_constructor(rows, 1);
        }
    }
    if (network->backend == AKIRA_BACKEND_CPU) {
        rows = ((matrix)network->weights[n])->rows;
        ws->output = matrix_constructor(rows, 1);
        ws->output_error = matrix_constructor(rows, 1);
    } else {
        rows = mmatrix_get_rows((mmatrix)network->weights[n]);
        ws->output = mmatrix_constructor(rows, 1);
        ws->output_error = mmatrix_constructor(rows, 1);
    }
    network->ws = ws;
}

nn nn_constructor(const int hidden_num, const int *sizes, akira_backend backend) {
    int i;
    nn network = malloc(sizeof(_nn));
    network->hidden_num = hidden_num;
    network->backend = backend;
    network->ws = NULL;
    network->weights = malloc(sizeof(void *) * (network->hidden_num+1));
    network->bias = malloc(sizeof(void *) * (network->hidden_num+1));
    for (i = 0; i < network->hidden_num + 1; i++) {
        if (backend == AKIRA_BACKEND_CPU) {
            network->weights[i] = matrix_constructor(sizes[i+1], sizes[i]);
            network->bias[i] = matrix_constructor(sizes[i+1], 1);
            matrix_randomize((matrix)network->weights[i], -1, 1);
            matrix_randomize((matrix)network->bias[i], -1, 1);
        } else {
            network->weights[i] = mmatrix_constructor(sizes[i+1], sizes[i]);
            network->bias[i] = mmatrix_constructor(sizes[i+1], 1);
            mmatrix_randomize((mmatrix)network->weights[i], -1, 1);
            mmatrix_randomize((mmatrix)network->bias[i], -1, 1);
        }
    }
    network->lr = 0.1;
    return network;
}

void nn_delete(nn network) {
    int i;
    ws_delete(network);
    for (i = 0; i < network->hidden_num + 1; i++) {
        if (network->backend == AKIRA_BACKEND_CPU) {
            matrix_delete((matrix)network->weights[i]);
            matrix_delete((matrix)network->bias[i]);
        } else {
            mmatrix_delete((mmatrix)network->weights[i]);
            mmatrix_delete((mmatrix)network->bias[i]);
        }
    }
    free(network->weights);
    free(network->bias);
    free(network);
}

/* Forward pass into the workspace activations. Each layer is a single BLAS
 * call: seeding the destination with the bias and accumulating (beta = 1)
 * folds the bias add into the matrix product. */
static void forward_cpu(nn network, matrix input) {
    workspace *ws = (workspace *)network->ws;
    const int n = network->hidden_num;
    int i;
    for (i = 0; i < n; i++) {
        matrix h = (matrix)ws->hidden[i];
        matrix_copy(h, (matrix)network->bias[i]);
        matrix_gemm(h, (matrix)network->weights[i], 0,
                    i == 0 ? input : (matrix)ws->hidden[i-1], 0, 1.0, 1.0);
        matrix_sigmoid(h);
    }
    matrix out = (matrix)ws->output;
    matrix_copy(out, (matrix)network->bias[n]);
    matrix_gemm(out, (matrix)network->weights[n], 0, (matrix)ws->hidden[n-1], 0, 1.0, 1.0);
    matrix_sigmoid(out);
}

/* Same forward pass on the GPU. Each layer is one fused kernel that does the
 * product, the bias add and the activation in a single dispatch. */
static void forward_metal(nn network, mmatrix input) {
    workspace *ws = (workspace *)network->ws;
    const int n = network->hidden_num;
    int i;
    for (i = 0; i < n; i++) {
        mmatrix_layer_forward((mmatrix)ws->hidden[i], (mmatrix)network->weights[i],
                              i == 0 ? input : (mmatrix)ws->hidden[i-1],
                              (mmatrix)network->bias[i]);
    }
    mmatrix_layer_forward((mmatrix)ws->output, (mmatrix)network->weights[n],
                          (mmatrix)ws->hidden[n-1], (mmatrix)network->bias[n]);
}

void *run(nn network, void *input) {
    void *output_layer = NULL;
    if (!network->ws) {
        ws_init(network);
    }
    workspace *ws = (workspace *)network->ws;
    /* The caller owns the result, so hand back a copy of the scratch output
     * rather than the workspace buffer itself. */
    if (network->backend == AKIRA_BACKEND_CPU) {
        forward_cpu(network, (matrix)input);
        output_layer = matrix_constructor(((matrix)ws->output)->rows, 1);
        matrix_copy((matrix)output_layer, (matrix)ws->output);
    } else {
        forward_metal(network, (mmatrix)input);
        output_layer = mmatrix_constructor(mmatrix_get_rows((mmatrix)ws->output), 1);
        mmatrix_copy((mmatrix)output_layer, (mmatrix)ws->output);
    }
    return output_layer;
}

void train(nn network, void *training_input, void *training_output) {
    if (!network->ws) {
        ws_init(network);
    }
    if (network->backend == AKIRA_BACKEND_CPU) {
        workspace *ws = (workspace *)network->ws;
        const int n = network->hidden_num;
        int i;

        forward_cpu(network, (matrix)training_input);

        matrix output_layer = (matrix)ws->output;
        matrix output_layer_error = (matrix)ws->output_error;
        matrix_copy(output_layer_error, (matrix)training_output);
        matrix_sub(output_layer_error, output_layer);

        /* output_layer becomes the scaled output gradient. */
        matrix_sigmoid_derivative(output_layer);
        matrix_hadamard(output_layer, output_layer_error);
        matrix_scalar_mult(output_layer, network->lr);

        /* W += grad * h**T is a rank-1 update; passing the transpose flag
         * means the transpose is never materialized. */
        matrix_gemm((matrix)network->weights[n], output_layer, 0, (matrix)ws->hidden[n-1], 1, 1.0, 1.0);
        matrix_add((matrix)network->bias[n], output_layer);

        /* Backpropagate through the just-updated weights, as before. */
        matrix last_layer_error = (matrix)ws->error[n-1];
        matrix_gemm(last_layer_error, (matrix)network->weights[n], 1, output_layer_error, 0, 1.0, 0.0);

        for (i = n - 1; i >= 0; i--) {
            matrix h = (matrix)ws->hidden[i];
            matrix_sigmoid_derivative(h);
            matrix_hadamard(h, last_layer_error);
            matrix_scalar_mult(h, network->lr);
            matrix_gemm((matrix)network->weights[i], h, 0,
                        i == 0 ? (matrix)training_input : (matrix)ws->hidden[i-1], 1, 1.0, 1.0);
            matrix_add((matrix)network->bias[i], h);
            /* The error signal below the first layer is never read, so the
             * widest gemv of the step (input-sized) is simply skipped. */
            if (i > 0) {
                matrix next_error = (matrix)ws->error[i-1];
                matrix_gemm(next_error, (matrix)network->weights[i], 1, last_layer_error, 0, 1.0, 0.0);
                last_layer_error = next_error;
            }
        }
    } else {
        /* Mirror of the CPU path. Nothing is allocated, no transpose is
         * materialized, and the whole step is encoded into one command buffer
         * that is submitted once, when the next sample overwrites the input. */
        workspace *ws = (workspace *)network->ws;
        const int n = network->hidden_num;
        int i;

        forward_metal(network, (mmatrix)training_input);

        mmatrix output_layer = (mmatrix)ws->output;
        mmatrix output_layer_error = (mmatrix)ws->output_error;
        mmatrix_sub_into(output_layer_error, (mmatrix)training_output, output_layer);

        /* output_layer becomes the scaled output gradient. */
        mmatrix_gradient(output_layer, output_layer_error, (float)network->lr);

        mmatrix_gemm((mmatrix)network->weights[n], output_layer, 0, (mmatrix)ws->hidden[n-1], 1, 1.0f, 1.0f);
        mmatrix_add((mmatrix)network->bias[n], output_layer);

        mmatrix last_layer_error = (mmatrix)ws->error[n-1];
        mmatrix_gemm(last_layer_error, (mmatrix)network->weights[n], 1, output_layer_error, 0, 1.0f, 0.0f);

        for (i = n - 1; i >= 0; i--) {
            mmatrix h = (mmatrix)ws->hidden[i];
            mmatrix_gradient(h, last_layer_error, (float)network->lr);
            mmatrix_gemm((mmatrix)network->weights[i], h, 0,
                         i == 0 ? (mmatrix)training_input : (mmatrix)ws->hidden[i-1], 1, 1.0f, 1.0f);
            mmatrix_add((mmatrix)network->bias[i], h);
            if (i > 0) {
                mmatrix next_error = (mmatrix)ws->error[i-1];
                mmatrix_gemm(next_error, (mmatrix)network->weights[i], 1, last_layer_error, 0, 1.0f, 0.0f);
                last_layer_error = next_error;
            }
        }
    }
}

int save(nn network, const char *f_name, const char *notes, const char *file_version) {
    char *filename = malloc(sizeof(char) * (strlen(f_name) + 1));
    strcpy(filename, f_name);
    char *ext = strrchr(filename, '.');
    char a;
    int rtn = 1;
    char *temp;
    if (ext && (strcmp(ext + 1, ".akr") != 0)) {
        printf("Warning: akira models use the .akr extension\n");
    } else if (!ext) {
        filename = realloc(filename, strlen(filename) + 5);
        strcat(filename, ".akr");
    }
    while (access(filename, F_OK) != -1) {
        printf("File %s exists\nDo you want to overwrite it? (y/n)\n", filename);
        scanf(" %c", &a);
        if (a == 'y' || a == 'Y') {
            break;
        }
        if (a != 'y') {
            temp = malloc(sizeof(char) * (strlen(filename) + 2));
            strcpy(temp, filename);
            filename = realloc(filename, strlen(filename) + strlen("copy_") + 1);
            strcpy(filename, "copy_");
            strcat(filename, temp);
            free(temp);
        }
    }
    
    if (file_version == NULL || strcmp(file_version, FILE_VERSION) == 0) {
        rtn = save_0_1_0(network, filename, notes);
    } else {
        printf("Unknown file vesrion %s. Use latest %s instead?\n", file_version, FILE_VERSION);
        rtn = save_0_1_0(network, filename, notes);
    }
    free(filename);
    return rtn;
}

int save_0_1_0(nn network, const char *filename, const char *notes) {
    FILE *fp;
    int i, j;
    char akira_version[6], file_version[6];
    strcpy(akira_version, AKIRA_VERSION);
    strcpy(file_version, FILE_VERSION);
    if ((fp = fopen(filename, "w")) == NULL) {
        printf("ERROR: Could not open file %s\n", filename);
        return 1;
    }
    fprintf(fp, "%s %s\n", file_version, akira_version);
    fprintf(fp, "%d %lf\n", network->hidden_num, network->lr);
    for (i = 0; i < network->hidden_num + 1; i++) {
        if (network->backend == AKIRA_BACKEND_CPU) {
            matrix w = (matrix)network->weights[i];
            matrix b = (matrix)network->bias[i];
            fprintf(fp, "%d %d\n", w->rows, w->columns);
            for (j = 0; j < w->rows * w->columns; j++) {
                fprintf(fp, "%lf ", w->p[j]);
            }
            fprintf(fp, "\n%d %d\n", b->rows, b->columns);
            for (j = 0; j < b->rows * b->columns; j++) {
                fprintf(fp, "%lf ", b->p[j]);
            }
        } else {
            mmatrix w = (mmatrix)network->weights[i];
            mmatrix b = (mmatrix)network->bias[i];
            int w_rows = mmatrix_get_rows(w);
            int w_cols = mmatrix_get_cols(w);
            float *w_data = malloc(sizeof(float) * w_rows * w_cols);
            mmatrix_copy_to_host(w, w_data);
            fprintf(fp, "%d %d\n", w_rows, w_cols);
            for (j = 0; j < w_rows * w_cols; j++) {
                fprintf(fp, "%f ", w_data[j]);
            }
            free(w_data);

            fprintf(fp, "\n");

            int b_rows = mmatrix_get_rows(b);
            int b_cols = mmatrix_get_cols(b);
            float *b_data = malloc(sizeof(float) * b_rows * b_cols);
            mmatrix_copy_to_host(b, b_data);
            fprintf(fp, "%d %d\n", b_rows, b_cols);
            for (j = 0; j < b_rows * b_cols; j++) {
                fprintf(fp, "%f ", b_data[j]);
            }
            free(b_data);
        }
    }
    fprintf(fp, "\n%s\n", notes);
    fclose(fp);
    return 0;
}

nn load(const char *filename, int backend) {
    FILE *fp;
    char file_version[6], akira_version[6];
    if (access(filename, F_OK) == -1) {
        printf("File %s does not exist\n", filename);
        return NULL;
    }
    if ((fp = fopen(filename, "r")) == NULL) {
        printf("ERROR: Could not open file %s\n", filename);
        return NULL;
    }
    fscanf(fp, "%s %s\n", file_version, akira_version);
    fclose(fp);
    if (strcmp(akira_version, AKIRA_VERSION) != 0) {
        printf("ERROR: the file is meant for akira version %s\n", akira_version);
    }
    if (strcmp(file_version, FILE_VERSION) != 0) {
        printf("Warning: Not the current file version (%s)\n", file_version);
    }
    if (strcmp(file_version, "0.1.0") == 0) {
        return load_0_1_0(filename, backend);
    } else {
        printf("ERROR: Unknown file version %s.\n", file_version);
        return NULL;
    }
    return NULL;
}

nn load_0_1_0(const char *filename, int backend) {
    FILE *fp;
    int i, j;
    char c;
    char akira_version[6], file_version[6];
    if ((fp = fopen(filename, "r")) == NULL) {
        printf("ERROR: Could not open file %s\n", filename);
        return NULL;
    }
    nn network = malloc(sizeof(_nn));
    fscanf(fp, "%s %s\n", file_version, akira_version);
    fscanf(fp, "%d %lf\n", &(network->hidden_num), &(network->lr));
    network->backend = backend;
    network->ws = NULL;
    network->weights = malloc(sizeof(void *) * (network->hidden_num + 1));
    network->bias = malloc(sizeof(void *) * (network->hidden_num + 1));
    for (i = 0; i < network->hidden_num + 1; i++) {
        if (network->backend == AKIRA_BACKEND_CPU) {
            int rows, cols;
            fscanf(fp, "%d %d\n", &rows, &cols);
            matrix w = matrix_constructor(rows, cols);
            for (j = 0; j < rows * cols; j++) {
                fscanf(fp, "%lf ", &(w->p[j]));
            }
            network->weights[i] = w;
            fscanf(fp, "\n");
            fscanf(fp, "%d %d\n", &rows, &cols);
            matrix b = matrix_constructor(rows, cols);
            for (j = 0; j < rows * cols; j++) {
                fscanf(fp, "%lf ", &(b->p[j]));
            }
            network->bias[i] = b;
        } else { // AKIRA_BACKEND_METAL
            int rows, cols;
            fscanf(fp, "%d %d\n", &rows, &cols);
            float *w_data = malloc(sizeof(float) * rows * cols);
            for (j = 0; j < rows * cols; j++) {
                double val;
                fscanf(fp, "%lf ", &val);
                w_data[j] = (float)val;
            }
            mmatrix w = mmatrix_constructor(rows, cols);
            mmatrix_copy_from_host(w, w_data);
            network->weights[i] = w;
            free(w_data);

            fscanf(fp, "\n");

            fscanf(fp, "%d %d\n", &rows, &cols);
            float *b_data = malloc(sizeof(float) * rows * cols);
            for (j = 0; j < rows * cols; j++) {
                double val;
                fscanf(fp, "%lf ", &val);
                b_data[j] = (float)val;
            }
            mmatrix b = mmatrix_constructor(rows, cols);
            mmatrix_copy_from_host(b, b_data);
            network->bias[i] = b;
            free(b_data);
        }
    }
    while ((c = fgetc(fp)) != '\n' && c != EOF); // Consume rest of the line
    fgets(network->notes, 255, fp);
    fclose(fp);
    return network;
}

void nn_print_weights_biases(nn network) {
    int i;
    printf("--- Weights ---\n");
    for (i = 0; i < network->hidden_num + 1; i++) {
        printf("Layer %d weights:\n", i);
        if (network->backend == AKIRA_BACKEND_CPU) {
            matrix_print((matrix)network->weights[i]);
        } else {
            mmatrix_print((mmatrix)network->weights[i]);
        }
    }
}
