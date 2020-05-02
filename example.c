#include <stdio.h>
#include <stdlib.h>
#include "brain.h"

int main(int argc, char **argv) {
    int i;
    const int size_of_input = 2;
    const int size_of_output = 1;
    const int train_number = 4;
    int number_of_iteration = 100000;
    if (argc == 2) {
        number_of_iteration = atoi(argv[1]);
    }
    double dtraining_input[] = {
        0, 0,
        1, 1,
        1, 0,
        0, 1
    };
    double dtraining_output[] = {
        0,
        1,
        1,
        0
    };
    double dtest_input[] = {
        1,
        1
    };
    matrix_t training_input = { train_number, size_of_input, dtraining_input};
    matrix_t training_output = { train_number, size_of_output, dtraining_output};
    matrix_t test_input = { 1, size_of_input, dtest_input};
    double dsynaptic_weights[training_input.w];
    for (i = 0; i < training_input.w; i++) {
        dsynaptic_weights[i] = get_random(-1.0, 1.0);
    }
    matrix_t synaptic_weights = { training_input.w, 1, dsynaptic_weights};

    train(&training_input, &training_output, &synaptic_weights, number_of_iteration);
    matrix output = run(&synaptic_weights, &test_input);
    mat_show(output);
    free(output);
}