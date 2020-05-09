#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include "neural_network.h"

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