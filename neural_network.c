#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "./cmatrix/cmatrix.h"

typedef struct nn {
    int input, hidden, output;
    matrix weights_ih, weights_ho;
    matrix bias_ih, bias_ho;
    double lr;
} _nn, *nn;

double sigmoid(double number) {
    return (1 / (1 + exp(-number)));
}

double sigmoid_derivative(double number) {
    return number * (1 - number);
}

nn nn_constructor(int si, int sh, int so) {
    nn network = malloc(sizeof(_nn));
    network->input = si;
    network->hidden = sh;
    network->output = so;

    network->weights_ih = matrix_constructor(sh, si);
    network->weights_ho = matrix_constructor(so, sh);

    network->bias_ih = matrix_constructor(sh, 1);
    network->bias_ho = matrix_constructor(so, 1);
    //memset(network->bias_ih, 1.0, sizeof(double) * sh);
    //memset(network->bias_ho, 1.0, sizeof(double) * so);

    matrix_randomize(network->weights_ih, -1, 1);
    matrix_randomize(network->weights_ho, -1, 1);
    matrix_randomize(network->bias_ih, -1, 1);
    matrix_randomize(network->bias_ho, -1, 1);
    network->lr = 0.1;

    return network;
}

void nn_delete(nn network) {
    matrix_delete(network->weights_ih);
    matrix_delete(network->weights_ho);
    matrix_delete(network->bias_ih);
    matrix_delete(network->bias_ho);
    free(network);
}

matrix run(nn network, matrix input) {
    matrix hidden_layer = matrix_mult(network->weights_ih, input);
    matrix_add(hidden_layer, network->bias_ih);

    matrix_apply(hidden_layer, sigmoid);

    matrix output_layer = matrix_mult(network->weights_ho, hidden_layer);
    matrix_add(output_layer, network->bias_ho);
    matrix_apply(output_layer, sigmoid);
    matrix_delete(hidden_layer);
    return output_layer;
}

void train(nn network, matrix training_input, matrix training_output) {
    matrix hidden_layer = matrix_mult(network->weights_ih, training_input);
    matrix_add(hidden_layer, network->bias_ih);

    matrix_apply(hidden_layer, sigmoid);

    matrix output_layer = matrix_mult(network->weights_ho, hidden_layer);
    matrix_add(output_layer, network->bias_ho);
    matrix_apply(output_layer, sigmoid);

    matrix output_layer_error = matrix_constructor(training_output->rows, training_output->columns);
    memcpy(output_layer_error->p, training_output->p, sizeof(double) * output_layer_error->rows * output_layer_error->columns);
    // output output_layer_error
    matrix_sub(output_layer_error, output_layer);
    // gradient
    matrix_apply(output_layer, sigmoid_derivative);
    matrix_hadamard(output_layer, output_layer_error);
    matrix_scalar_mult(output_layer, network->lr);

    // deltas
    matrix hidden_layer_T = transpose(hidden_layer);
    matrix weights_ho_delta = matrix_mult(output_layer, hidden_layer_T);

    // adjustments
    matrix_add(network->weights_ho, weights_ho_delta);
    matrix_add(network->bias_ho, output_layer);

    matrix weights_ho_T = transpose(network->weights_ho);
    matrix hidden_layer_error = matrix_mult(weights_ho_T, output_layer_error);

    // gradient hidden
    matrix_apply(hidden_layer, sigmoid_derivative);
    matrix_hadamard(hidden_layer, hidden_layer_error);
    matrix_scalar_mult(hidden_layer, network->lr);

    // deltas hidden
    matrix input_layer_T = transpose(training_input);
    matrix weights_ih_delta = matrix_mult(hidden_layer, input_layer_T);

    // adjustments hidden
    matrix_add(network->weights_ih, weights_ih_delta);
    matrix_add(network->bias_ih, hidden_layer);

    matrix_delete(output_layer);
    matrix_delete(output_layer_error);
    matrix_delete(weights_ho_T);
    matrix_delete(hidden_layer_error);
    matrix_delete(hidden_layer);
    matrix_delete(hidden_layer_T);
    matrix_delete(weights_ho_delta);
    matrix_delete(input_layer_T);
    matrix_delete(weights_ih_delta);
}

int main(void) {
    srand(time(0));
    //xor
    double dtraining_input[] = { 1, 0,
                                1, 1,
                                0, 0,
                                0, 1 };
    double dtraining_output[] = { 1, 0, 0, 1 };
    int i, j;
    nn network = nn_constructor(2, 2, 1);
    matrix training_input = matrix_constructor(2, 1);
    matrix training_output = matrix_constructor(1, 1);
    for (i = 0; i < 1000000; i++) {
        for (j = 0; j < 4; j++) {
            // randomize order please
            memcpy(training_input->p, dtraining_input + 2 * j, sizeof(double) * 2);
            memcpy(training_output->p, dtraining_output + 1 * j, sizeof(double) * 1);
            train(network, training_input, training_output);
        }
    }

    matrix output;
    for (j = 0; j < 4; j++) {
        memcpy(training_input->p, dtraining_input + 2 * j, sizeof(double) * 2);
        memcpy(training_output->p, dtraining_output + 1 * j, sizeof(double) * 1);
        output = run(network, training_input);
        printf("input (%lf %lf):\n", training_input->p[0], training_input->p[1]);
        matrix_print(output);
        printf("\n");
        matrix_delete(output);
    }

    matrix_delete(training_input);
    matrix_delete(training_output);
    nn_delete(network);
}