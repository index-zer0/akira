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

static inline double sigmoid(double number) {
    return (1 / (1 + exp(-number)));
}

static inline double sigmoid_derivative(double number) {
    return number * (1 - number);
}

nn nn_constructor(const int hidden_num, const int *sizes, akira_backend backend) {
    int i;
    nn network = malloc(sizeof(_nn));
    network->hidden_num = hidden_num;
    network->backend = backend;
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

void *run(nn network, void *input) {
    int i = 0;
    void **hidden_layer = malloc(sizeof(void *) * network->hidden_num);
    void *output_layer = NULL;
    if (network->backend == AKIRA_BACKEND_CPU) {
        for (i = 0; i < network->hidden_num; i++) {
            if (i != 0) {
                hidden_layer[i] = matrix_mult((matrix)network->weights[i], (matrix)hidden_layer[i-1]);
            } else {
                hidden_layer[0] = matrix_mult((matrix)network->weights[0], (matrix)input);
            }
            matrix_add((matrix)hidden_layer[i], (matrix)network->bias[i]);
            matrix_apply((matrix)hidden_layer[i], sigmoid);
        }
        output_layer = matrix_mult((matrix)network->weights[network->hidden_num], (matrix)hidden_layer[network->hidden_num-1]);
        matrix_add((matrix)output_layer, (matrix)network->bias[network->hidden_num]);
        matrix_apply((matrix)output_layer, sigmoid);
        for (i = 0; i < network->hidden_num; i++) {
            matrix_delete((matrix)hidden_layer[i]);
        }
    } else {
        for (i = 0; i < network->hidden_num; i++) {
            if (i != 0) {
                hidden_layer[i] = mmatrix_mult((mmatrix)network->weights[i], (mmatrix)hidden_layer[i-1]);
            } else {
                hidden_layer[0] = mmatrix_mult((mmatrix)network->weights[0], (mmatrix)input);
            }
            mmatrix_add((mmatrix)hidden_layer[i], (mmatrix)network->bias[i]);
            mmatrix_apply((mmatrix)hidden_layer[i], 0); // 0: sigmoid
        }
        output_layer = mmatrix_mult((mmatrix)network->weights[network->hidden_num], (mmatrix)hidden_layer[network->hidden_num-1]);
        mmatrix_add((mmatrix)output_layer, (mmatrix)network->bias[network->hidden_num]);
        mmatrix_apply((mmatrix)output_layer, 0); // 0: sigmoid
        for (i = 0; i < network->hidden_num; i++) {
            mmatrix_delete((mmatrix)hidden_layer[i]);
        }
    }
    free(hidden_layer);
    return output_layer;
}

void train(nn network, void *training_input, void *training_output) {
    if (network->backend == AKIRA_BACKEND_CPU) {
        int i = 0;
        matrix weights_T, weights_ho_delta, hidden_layer_T, last_layer_error, last_layer_error_temp, weights_delta, layer_T, output_layer, output_layer_error;
        matrix hidden_layer[network->hidden_num];
        for (i = 0; i < network->hidden_num; i++) {
            if (i != 0) {
                hidden_layer[i] = matrix_mult((matrix)network->weights[i], hidden_layer[i-1]);
            } else {
                hidden_layer[0] = matrix_mult((matrix)network->weights[0], (matrix)training_input);
            }
            matrix_add(hidden_layer[i], (matrix)network->bias[i]);
            matrix_apply(hidden_layer[i], sigmoid);
        }
        output_layer = matrix_mult((matrix)network->weights[network->hidden_num], hidden_layer[network->hidden_num-1]);
        matrix_add(output_layer, (matrix)network->bias[network->hidden_num]);
        matrix_apply(output_layer, sigmoid);

        output_layer_error = matrix_constructor(((matrix)training_output)->rows, ((matrix)training_output)->columns);
        memcpy(output_layer_error->p, ((matrix)training_output)->p, sizeof(double) * output_layer_error->rows * output_layer_error->columns);
        matrix_sub(output_layer_error, output_layer);

        matrix_apply(output_layer, sigmoid_derivative);
        matrix_hadamard(output_layer, output_layer_error);
        matrix_scalar_mult(output_layer, network->lr);
        hidden_layer_T = transpose(hidden_layer[network->hidden_num-1]);
        weights_ho_delta = matrix_mult(output_layer, hidden_layer_T);
        matrix_delete(hidden_layer_T);
        matrix_add((matrix)network->weights[network->hidden_num], weights_ho_delta);
        matrix_add((matrix)network->bias[network->hidden_num], output_layer);

        weights_T = transpose((matrix)network->weights[network->hidden_num]);
        last_layer_error = matrix_mult(weights_T, output_layer_error);
        matrix_delete(weights_T);
        for (i = network->hidden_num-1; i >= 0; i--) {
            matrix_apply(hidden_layer[i], sigmoid_derivative);
            matrix_hadamard(hidden_layer[i], last_layer_error);
            matrix_scalar_mult(hidden_layer[i], network->lr);
            if (i == 0) {
                layer_T = transpose((matrix)training_input);
            } else {
                layer_T = transpose(hidden_layer[i-1]);
            }
            weights_delta = matrix_mult(hidden_layer[i], layer_T);
            matrix_delete(layer_T);
            matrix_add((matrix)network->weights[i], weights_delta);
            matrix_add((matrix)network->bias[i], hidden_layer[i]);
            matrix_delete(weights_delta);
            weights_T = transpose((matrix)network->weights[i]);
            last_layer_error_temp = matrix_constructor(last_layer_error->rows, last_layer_error->columns);
            memcpy(last_layer_error_temp->p, last_layer_error->p, sizeof(double) * last_layer_error_temp->rows * last_layer_error_temp->columns);
            matrix_delete(last_layer_error);
            last_layer_error = matrix_mult(weights_T, last_layer_error_temp);
            matrix_delete(last_layer_error_temp);
            matrix_delete(weights_T);
        }
        for (i = 0; i < network->hidden_num; i++) {
            matrix_delete(hidden_layer[i]);
        }
        matrix_delete(last_layer_error);
        matrix_delete(weights_ho_delta);
        matrix_delete(output_layer);
        matrix_delete(output_layer_error);
    } else {
        int i = 0;
        mmatrix weights_T, weights_ho_delta, hidden_layer_T, last_layer_error, last_layer_error_temp, weights_delta, layer_T, output_layer, output_layer_error;
        mmatrix hidden_layer[network->hidden_num];
        for (i = 0; i < network->hidden_num; i++) {
            if (i != 0) {
                hidden_layer[i] = mmatrix_mult((mmatrix)network->weights[i], hidden_layer[i-1]);
            } else {
                hidden_layer[0] = mmatrix_mult((mmatrix)network->weights[0], (mmatrix)training_input);
            }
            mmatrix_add((mmatrix)hidden_layer[i], (mmatrix)network->bias[i]);
            mmatrix_apply((mmatrix)hidden_layer[i], 0); // 0: sigmoid
        }
        output_layer = mmatrix_mult((mmatrix)network->weights[network->hidden_num], hidden_layer[network->hidden_num-1]);
        mmatrix_add((mmatrix)output_layer, (mmatrix)network->bias[network->hidden_num]);
        mmatrix_apply((mmatrix)output_layer, 0); // 0: sigmoid

        output_layer_error = mmatrix_constructor(mmatrix_get_rows((mmatrix)training_output), mmatrix_get_cols((mmatrix)training_output));
        mmatrix_copy(output_layer_error, (mmatrix)training_output);
        mmatrix_sub(output_layer_error, output_layer);

        mmatrix_apply(output_layer, 1); // 1: sigmoid_derivative
        mmatrix_hadamard(output_layer, output_layer_error);
        mmatrix_scalar_mult(output_layer, network->lr);
        hidden_layer_T = mmatrix_transpose(hidden_layer[network->hidden_num-1]);
        weights_ho_delta = mmatrix_mult(output_layer, hidden_layer_T);
        mmatrix_add((mmatrix)network->weights[network->hidden_num], weights_ho_delta);
        mmatrix_add((mmatrix)network->bias[network->hidden_num], output_layer);

        weights_T = mmatrix_transpose((mmatrix)network->weights[network->hidden_num]);
        last_layer_error = mmatrix_mult(weights_T, output_layer_error);
        for (i = network->hidden_num-1; i >= 0; i--) {
            mmatrix_apply(hidden_layer[i], 1); // 1: sigmoid_derivative
            mmatrix_hadamard(hidden_layer[i], last_layer_error);
            mmatrix_scalar_mult(hidden_layer[i], network->lr);
            if (i == 0) {
                layer_T = mmatrix_transpose((mmatrix)training_input);
            } else {
                layer_T = mmatrix_transpose(hidden_layer[i-1]);
            }
            weights_delta = mmatrix_mult(hidden_layer[i], layer_T);
            mmatrix_add((mmatrix)network->weights[i], weights_delta);
            mmatrix_add((mmatrix)network->bias[i], hidden_layer[i]);
            mmatrix_delete(weights_delta);
            mmatrix_delete(layer_T);
            weights_T = mmatrix_transpose((mmatrix)network->weights[i]);
            last_layer_error_temp = mmatrix_mult(weights_T, last_layer_error);
            mmatrix_delete(last_layer_error);
            mmatrix_delete(weights_T);
            last_layer_error = last_layer_error_temp;
        }

        for (i = 0; i < network->hidden_num; i++) {
            mmatrix_delete(hidden_layer[i]);
        }
        mmatrix_delete(hidden_layer_T);
        mmatrix_delete(weights_ho_delta);
        mmatrix_delete(last_layer_error);
        mmatrix_delete(output_layer);
        mmatrix_delete(output_layer_error);
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
    network->weights = malloc(sizeof(void *) * (network->hidden_num + 1));
    network->bias = malloc(sizeof(void *) * (network->hidden_num + 1));
    for (i = 0; i < network->hidden_num + 1; i++) {
        if (network->backend == AKIRA_BACKEND_CPU) {
            matrix w = malloc(sizeof(_matrix));
            fscanf(fp, "%d %d\n", &(w->rows), &(w->columns));
            w->p = malloc(sizeof(double) * w->rows * w->columns);
            for (j = 0; j < w->rows * w->columns; j++) {
                fscanf(fp, "%lf ", &(w->p[j]));
            }
            network->weights[i] = w;
            fscanf(fp, "\n");
            matrix b = malloc(sizeof(_matrix));
            fscanf(fp, "%d %d\n", &(b->rows), &(b->columns));
            b->p = malloc(sizeof(double) * b->rows * b->columns);
            for (j = 0; j < b->rows * b->columns; j++) {
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
