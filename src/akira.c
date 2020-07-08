#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <unistd.h>
#include "akira.h"

int save_0_1_0(nn, const char *, const char *);
nn load_0_1_0(const char *);

static inline double sigmoid(double number) {
    return (1 / (1 + exp(-number)));
}

static inline double sigmoid_derivative(double number) {
    return number * (1 - number);
}

nn nn_constructor(const int hidden_num, const int *sizes) {
    int i;
    nn network = malloc(sizeof(_nn));
    network->hidden_num = hidden_num;

    network->weights = malloc(sizeof(matrix) * (network->hidden_num+1));
    network->bias = malloc(sizeof(matrix) * (network->hidden_num+1));

    for (i = 0; i < network->hidden_num + 1; i++) { // + 1 because of the input layer
        network->weights[i] = matrix_constructor(sizes[i+1], sizes[i]);
        network->bias[i] = matrix_constructor(sizes[i+1], 1);
        matrix_randomize(network->weights[i], -1, 1);
        matrix_randomize(network->bias[i], -1, 1);
    }
    network->lr = 0.1;

    return network;
}

void nn_delete(nn network) {
    int i;
    for (i = 0; i < network->hidden_num + 1; i++) {
        matrix_delete(network->weights[i]);
        matrix_delete(network->bias[i]);
    }
    free(network->weights);
    free(network->bias);
    free(network);
}

matrix run(nn network, matrix input) {
    int i = 0;
    matrix hidden_layer[network->hidden_num];
    for (i = 0; i < network->hidden_num; i++) {
        if (i != 0) {
            hidden_layer[i] = matrix_mult(network->weights[i], hidden_layer[i-1]);
        } else {
            hidden_layer[0] = matrix_mult(network->weights[0], input);
        }
        matrix_add(hidden_layer[i], network->bias[i]);
        matrix_apply(hidden_layer[i], sigmoid);
    }

    matrix output_layer = matrix_mult(network->weights[network->hidden_num], hidden_layer[network->hidden_num-1]);
    matrix_add(output_layer, network->bias[network->hidden_num]);
    matrix_apply(output_layer, sigmoid);
    for (i = 0; i < network->hidden_num; i++) {
        matrix_delete(hidden_layer[i]);
    }
    return output_layer;
}

void train(nn network, matrix training_input, matrix training_output) {
    int i = 0;
    matrix weights_T, weights_ho_delta, hidden_layer_T, last_layer_error, last_layer_error_temp, weights_delta, layer_T, output_layer, output_layer_error;
    matrix hidden_layer[network->hidden_num];
    for (i = 0; i < network->hidden_num; i++) {
        if (i != 0) {
            hidden_layer[i] = matrix_mult(network->weights[i], hidden_layer[i-1]);
        } else {
            hidden_layer[0] = matrix_mult(network->weights[0], training_input);
        }
        matrix_add(hidden_layer[i], network->bias[i]);
        matrix_apply(hidden_layer[i], sigmoid);
    }
    output_layer = matrix_mult(network->weights[network->hidden_num], hidden_layer[network->hidden_num-1]);
    matrix_add(output_layer, network->bias[network->hidden_num]);
    matrix_apply(output_layer, sigmoid);

    output_layer_error = matrix_constructor(training_output->rows, training_output->columns);
    memcpy(output_layer_error->p, training_output->p, sizeof(double) * output_layer_error->rows * output_layer_error->columns);
    // output output_layer_error
    matrix_sub(output_layer_error, output_layer);

    // gradient
    matrix_apply(output_layer, sigmoid_derivative);
    matrix_hadamard(output_layer, output_layer_error);
    matrix_scalar_mult(output_layer, network->lr);
    // delta
    hidden_layer_T = transpose(hidden_layer[network->hidden_num-1]);
    weights_ho_delta = matrix_mult(output_layer, hidden_layer_T);
    matrix_delete(hidden_layer_T);
    // adjustments
    matrix_add(network->weights[network->hidden_num], weights_ho_delta);
    matrix_add(network->bias[network->hidden_num], output_layer);

    weights_T = transpose(network->weights[network->hidden_num]);
    last_layer_error = matrix_mult(weights_T, output_layer_error);
    matrix_delete(weights_T);
    for (i = network->hidden_num-1; i >= 0; i--) {
        // gradient
        matrix_apply(hidden_layer[i], sigmoid_derivative);
        matrix_hadamard(hidden_layer[i], last_layer_error);
        matrix_scalar_mult(hidden_layer[i], network->lr);

        // deltas
        if (i == 0) {
            layer_T = transpose(training_input);
        } else {
            layer_T = transpose(hidden_layer[i-1]);
        }
        weights_delta = matrix_mult(hidden_layer[i], layer_T);
        matrix_delete(layer_T);
        // adjustments
        matrix_add(network->weights[i], weights_delta);
        matrix_add(network->bias[i], hidden_layer[i]);
        matrix_delete(weights_delta);

        weights_T = transpose(network->weights[i]);
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
        if (scanf(" %c", &a) < 0) {
            exit(-1);
        }
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
        fprintf(fp, "%d %d\n", network->weights[i]->rows, network->weights[i]->columns);
        for (j = 0; j < network->weights[i]->rows * network->weights[i]->columns; j++) {
            fprintf(fp, "%lf ", network->weights[i]->p[j]);
        }
        fprintf(fp, "\n%d %d\n", network->bias[i]->rows, network->bias[i]->columns);
        for (j = 0; j < network->bias[i]->rows * network->bias[i]->columns; j++) {
            fprintf(fp, "%lf ", network->bias[i]->p[j]);
        }
    }
    fprintf(fp, "\n%s\n", notes);
    fclose(fp);
    return 0;
}

nn load(const char *filename) {
    int i;
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
    if (fscanf(fp, "%s %s\n", file_version, akira_version) < 0) {
        exit(-1);
    }
    fclose(fp);
    if (strcmp(akira_version, AKIRA_VERSION) != 0) {
        printf("ERROR: the file is ment for akira version %s\n", akira_version);
    }
    if (strcmp(file_version, FILE_VERSION) != 0) {
        printf("Warning: Not the current file version (%s)\n", file_version);
    }
    if (strcmp(file_version, "0.1.0") == 0) {
        return load_0_1_0(filename);
    } else {
        printf("ERROR: Unknown file vesrion %s.\n", file_version);
        return NULL;
    }
    return NULL;
}

nn load_0_1_0(const char *filename) {
    FILE *fp;
    int i, j;
    char c;
    char akira_version[6], file_version[6];
    if ((fp = fopen(filename, "r")) == NULL) {
        printf("ERROR: Could not open file %s\n", filename);
        return NULL;
    }
    nn network = malloc(sizeof(_nn));
    if (fscanf(fp, "%s %s\n", file_version, akira_version) < 0) {
        exit(-1);
    }
    if (fscanf(fp, "%d %lf\n", &(network->hidden_num), &(network->lr)) < 0) {
        exit(-1);
    }

    network->weights = malloc(sizeof(matrix) * (network->hidden_num+1));
    network->bias = malloc(sizeof(matrix) * (network->hidden_num+1));
    printf("%s %s %d %lf\n", file_version, akira_version, network->hidden_num, network->lr);
    for (i = 0; i < network->hidden_num + 1; i++) {
        network->weights[i] = malloc(sizeof(_matrix));
        if (fscanf(fp, "%d %d\n", &(network->weights[i]->rows), &(network->weights[i]->columns)) < 0) {
            exit(-1);
        }
        network->weights[i]->p = malloc(sizeof(double) * network->weights[i]->rows * network->weights[i]->columns);
        for (j = 0; j < network->weights[i]->rows * network->weights[i]->columns; j++) {
            fscanf(fp, "%lf ", &(network->weights[i]->p[j]));
        }
        network->bias[i] = malloc(sizeof(_matrix));
        if (fscanf(fp, "\n%d %d\n", &(network->bias[i]->rows), &(network->bias[i]->columns)) < 0) {
            exit(-1);
        }
        network->bias[i]->p = malloc(sizeof(double) * network->bias[i]->rows * network->bias[i]->columns);
        for (j = 0; j < network->bias[i]->rows * network->bias[i]->columns; j++) {
            if (fscanf(fp, "%lf ", &(network->bias[i]->p[j])) < 0) {
                exit(-1);
            }
        }
    }
    printf("#############\nNote:\n");
    while(fscanf(fp, "%c", &c) != EOF) {
        printf("%c", c);
    }
    printf("\n#############\n");
    fclose(fp);
    return network;
}