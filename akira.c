#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "akira.h"

static inline double sigmoid(double number) {
    return (1 / (1 + exp(-number)));
}

static inline double sigmoid_derivative(double number) {
    return number * (1 - number);
}

nn nn_constructor(const int hidden_num, const int *sizes) { //(int si, int sh, int so) {
    int i;
    nn network = malloc(sizeof(_nn));
    network->hidden_num = hidden_num;
    network->input = sizes[0];
    network->hidden = sizes[1];
    network->output = sizes[2];

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
    //matrix hidden_layer = matrix_mult(network->weights[0], input);
    //matrix_add(hidden_layer, network->bias[0]);

    //matrix_apply(hidden_layer, sigmoid);

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
    matrix layer_error[network->hidden_num];
    for (i = 0; i < network->hidden_num; i++) {
        if (i != 0) {
            hidden_layer[i] = matrix_mult(network->weights[i], hidden_layer[i-1]);
        } else {
            hidden_layer[0] = matrix_mult(network->weights[0], training_input);
        }
        matrix_add(hidden_layer[i], network->bias[i]);
        matrix_apply(hidden_layer[i], sigmoid);
    }
    /*matrix hidden_layer = matrix_mult(network->weights[0], training_input);
    matrix_add(hidden_layer, network->bias[0]);

    matrix_apply(hidden_layer, sigmoid);*/
    output_layer = matrix_mult(network->weights[network->hidden_num], hidden_layer[network->hidden_num-1]); //////
    matrix_add(output_layer, network->bias[network->hidden_num]);
    matrix_apply(output_layer, sigmoid);
    /*matrix output_layer = matrix_mult(network->weights[1], hidden_layer);
    matrix_add(output_layer, network->bias[1]);
    matrix_apply(output_layer, sigmoid);*/

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

    for (i = network->hidden_num-1; i >= 0; i++) {
        printf("%d\n", i);
        // gradient
        matrix_apply(hidden_layer[i], sigmoid_derivative);
        matrix_hadamard(hidden_layer[i], last_layer_error);
        matrix_scalar_mult(hidden_layer[i], network->lr);

        // deltas
        layer_T = transpose(hidden_layer[i]);
        weights_delta = matrix_mult(hidden_layer[i], layer_T);
        matrix_delete(layer_T);
        // adjustments
        matrix_add(network->weights[i], weights_delta);
        matrix_add(network->bias[i], hidden_layer[i]);
        matrix_delete(weights_delta);

        weights_T = transpose(network->weights[i]);
        last_layer_error_temp = matrix_constructor(last_layer_error->rows, last_layer_error->columns);
        memcpy(last_layer_error_temp->p, last_layer_error->p, sizeof(double) * last_layer_error_temp->rows * last_layer_error_temp->columns);
        last_layer_error = matrix_mult(weights_T, last_layer_error_temp);
        matrix_delete(last_layer_error_temp);
        matrix_delete(weights_T);
    }
    for (i = 0; i < network->hidden_num; i++) {
        matrix_delete(hidden_layer[i]);
        matrix_delete(layer_error[i]);
    }
    matrix_delete(layer_error[network->hidden_num]);
    matrix_delete(last_layer_error);
    matrix_delete(weights_T);
    matrix_delete(weights_ho_delta);
    matrix_delete(last_layer_error);
    matrix_delete(last_layer_error_temp);
    matrix_delete(weights_T);
    matrix_delete(weights_delta);
    matrix_delete(layer_T);
    matrix_delete(output_layer);
    matrix_delete(output_layer_error);
}
    // gradient
    /*matrix_apply(output_layer, sigmoid_derivative);
    matrix_hadamard(output_layer, output_layer_error);
    matrix_scalar_mult(output_layer, network->lr);*/

    // deltas
    /*matrix hidden_layer_T = transpose(hidden_layer);
    matrix weights_ho_delta = matrix_mult(output_layer, hidden_layer_T);*/

    // adjustments
    /*matrix_add(network->weights[1], weights_ho_delta);
    matrix_add(network->bias[1], output_layer);*/
    
    /*matrix weights_ho_T = transpose(network->weights[1]);
    matrix hidden_layer_error = matrix_mult(weights_ho_T, output_layer_error);*/

    // gradient hidden
    /*matrix_apply(hidden_layer, sigmoid_derivative);
    matrix_hadamard(hidden_layer, hidden_layer_error);
    matrix_scalar_mult(hidden_layer, network->lr);

    // deltas hidden
    matrix input_layer_T = transpose(training_input);
    matrix weights_ih_delta = matrix_mult(hidden_layer, input_layer_T);

    // adjustments hidden
    matrix_add(network->weights[0], weights_ih_delta);
    matrix_add(network->bias[0], hidden_layer);*/