#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "akira.h"
//https://github.com/takafumihoriuchi/MNIST_for_C/
#include "MNIST_for_C/mnist.h"

double zero[10] =  {1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
double one[10] =   {0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
double two[10] =   {0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
double three[10] = {0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
double four[10] =  {0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0};
double five[10] =  {0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0};
double six[10] =   {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0};
double seven[10] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0};
double eight[10] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0};
double nine[10] =  {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0};

int main(int argc, char **argv) {
    srand(time(0));
    int i, j;
    const int size_of_input = 784;
    const int size_of_output = 10;
    const int train_number = 1000;
    int number_of_iteration = 10;
    if (argc == 2) {
        number_of_iteration = atoi(argv[1]);
    }

    load_mnist();
    double *dtraining_output = malloc(sizeof(double) * size_of_output * train_number);
    
    for (i = 0; i < train_number; i++) {
        //populate dtraining_output
        switch(train_label[i]) {
            case 0:
                memcpy(dtraining_output + i*size_of_output, zero, sizeof(double) * size_of_output);
                break;
            case 1:
                memcpy(dtraining_output + i*size_of_output, one, sizeof(double) * size_of_output);
                break;
            case 2:
                memcpy(dtraining_output + i*size_of_output, two, sizeof(double) * size_of_output);
                break;
            case 3:
                memcpy(dtraining_output + i*size_of_output, three, sizeof(double) * size_of_output);
                break;
            case 4:
                memcpy(dtraining_output + i*size_of_output, four, sizeof(double) * size_of_output);
                break;
            case 5:
                memcpy(dtraining_output + i*size_of_output, five, sizeof(double) * size_of_output);
                break;
            case 6:
                memcpy(dtraining_output + i*size_of_output, six, sizeof(double) * size_of_output);
                break;
            case 7:
                memcpy(dtraining_output + i*size_of_output, seven, sizeof(double) * size_of_output);
                break;
            case 8:
                memcpy(dtraining_output + i*size_of_output, eight, sizeof(double) * size_of_output);
                break;
            case 9:
                memcpy(dtraining_output + i*size_of_output, nine, sizeof(double) * size_of_output);
                break;
        }
        
    }
    int sizes[] = {size_of_input, 128, 128, size_of_output};
    //int sizes[] = {size_of_input, 64, size_of_output};
    nn network = nn_constructor(2, sizes);
    matrix training_input = matrix_constructor(size_of_input, 1);
    matrix training_output = matrix_constructor(size_of_output, 1);
    for (i = 0; i < number_of_iteration; i++) {
        for (j = 0; j < train_number; j++) {
            // randomize order please
            memcpy(training_input->p, &train_image[j][0], sizeof(double) * size_of_input);
            memcpy(training_output->p, dtraining_output + size_of_output * j, sizeof(double) * size_of_output);
            train(network, training_input, training_output);
        }
        //if ((i % 100) == 0) {
        printf("%lf%%\r", (double)((double)i/(double)number_of_iteration * 100.0));
        fflush(stdout);
        //}
    }
    printf("Done training\n");
    //int correct = 0, wrong = 0;
    matrix output;
    for (j = 0; j < 10; j++) {
        memcpy(training_input->p, &test_image[j][0], sizeof(double) * size_of_input);
        memcpy(training_output->p, dtraining_output + size_of_output * j, sizeof(double) * size_of_output);
        output = run(network, training_input);
        printf("\ninput %d:\n", test_label[j]);
        //matrix_print(output);
        double max = -100.0;
        int max_index = -1;
        for (i = 0; i < output->rows; i++) {
            if (output->p[i] > max) {
                max = output->p[i];
                max_index = i;
            }
        }
        printf("prediction: %d\n", max_index);
        matrix_delete(output);
    }
    free(dtraining_output);
    matrix_delete(training_input);
    matrix_delete(training_output);
    nn_delete(network);
}