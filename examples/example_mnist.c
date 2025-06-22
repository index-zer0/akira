#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "../src/akira.h"
//https://github.com/takafumihoriuchi/MNIST_for_C/
#include "../MNIST_for_C/mnist.h"

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
    g_debug = 1;
    srand(time(0));
    int i, j, backend_id;
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
        switch(train_label[i]) {
            case 0: memcpy(dtraining_output + i*size_of_output, zero, sizeof(double) * size_of_output); break;
            case 1: memcpy(dtraining_output + i*size_of_output, one, sizeof(double) * size_of_output); break;
            case 2: memcpy(dtraining_output + i*size_of_output, two, sizeof(double) * size_of_output); break;
            case 3: memcpy(dtraining_output + i*size_of_output, three, sizeof(double) * size_of_output); break;
            case 4: memcpy(dtraining_output + i*size_of_output, four, sizeof(double) * size_of_output); break;
            case 5: memcpy(dtraining_output + i*size_of_output, five, sizeof(double) * size_of_output); break;
            case 6: memcpy(dtraining_output + i*size_of_output, six, sizeof(double) * size_of_output); break;
            case 7: memcpy(dtraining_output + i*size_of_output, seven, sizeof(double) * size_of_output); break;
            case 8: memcpy(dtraining_output + i*size_of_output, eight, sizeof(double) * size_of_output); break;
            case 9: memcpy(dtraining_output + i*size_of_output, nine, sizeof(double) * size_of_output); break;
        }
    }
    int sizes[] = {size_of_input, 128, 128, size_of_output};
    for (backend_id = 0; backend_id < 2; backend_id++) {
        akira_backend backend = (backend_id == 0) ? AKIRA_BACKEND_CPU : AKIRA_BACKEND_METAL;
        printf("\n==============================\n");
        printf("Using backend: %s\n", backend == AKIRA_BACKEND_CPU ? "CPU (cmatrix)" : "Metal (mmatrix)");
        fflush(stdout);
        srand(42);
        clock_t start = clock();
        nn network = nn_constructor(2, sizes, backend);
        if (backend == AKIRA_BACKEND_CPU) {
            network->lr = 0.5;
        } else {
            network->lr = 0.1;
        }
        void *training_input;
        void *training_output;
        float *ftraining_input = NULL;
        float *ftraining_output = NULL;
        if (backend == AKIRA_BACKEND_CPU) {
            training_input = matrix_constructor(size_of_input, 1);
            training_output = matrix_constructor(size_of_output, 1);
        } else {
            training_input = mmatrix_constructor(size_of_input, 1);
            training_output = mmatrix_constructor(size_of_output, 1);
            ftraining_input = malloc(sizeof(float) * size_of_input);
            ftraining_output = malloc(sizeof(float) * size_of_output);
        }
        // TRAINING
        for (i = 0; i < number_of_iteration; i++) {
            for (j = 0; j < train_number; j++) {
                if (backend == AKIRA_BACKEND_CPU) {
                    memcpy(((matrix)training_input)->p, &train_image[j][0], sizeof(double) * size_of_input);
                    memcpy(((matrix)training_output)->p, dtraining_output + size_of_output * j, sizeof(double) * size_of_output);
                    train(network, training_input, training_output);
                } else {
                    for (int k = 0; k < size_of_input; ++k) ftraining_input[k] = (float)train_image[j][k];
                    for (int k = 0; k < size_of_output; ++k) ftraining_output[k] = (float)dtraining_output[size_of_output * j + k];
                    mmatrix_copy_from_host((mmatrix)training_input, ftraining_input);
                    mmatrix_copy_from_host((mmatrix)training_output, ftraining_output);
                    train(network, training_input, training_output);
                }
            }
            printf("%lf%%\r", (double)((double)i/(double)number_of_iteration * 100.0));
            fflush(stdout);
        }
        printf("Done training\n");
        // TESTING
        void *output;
        for (j = 0; j < 10; j++) {
            if (backend == AKIRA_BACKEND_CPU) {
                memcpy(((matrix)training_input)->p, &test_image[j][0], sizeof(double) * size_of_input);
                memcpy(((matrix)training_output)->p, dtraining_output + size_of_output * j, sizeof(double) * size_of_output);
                output = run(network, training_input);
                printf("\ninput %d:\n", test_label[j]);
                double max = -100.0;
                int max_index = -1;
                for (i = 0; i < ((matrix)output)->rows; i++) {
                    if (((matrix)output)->p[i] > max) {
                        max = ((matrix)output)->p[i];
                        max_index = i;
                    }
                }
                printf("prediction: %d\n", max_index);
                matrix_delete((matrix)output);
            } else {
                for (int k = 0; k < size_of_input; ++k) ftraining_input[k] = (float)test_image[j][k];
                mmatrix_copy_from_host((mmatrix)training_input, ftraining_input);
                output = run(network, training_input);
                float outbuf[10];
                mmatrix_copy_to_host((mmatrix)output, outbuf);
                printf("\ninput %d:\n", test_label[j]);
                float max = -100.0f;
                int max_index = -1;
                for (i = 0; i < 10; i++) {
                    if (outbuf[i] > max) {
                        max = outbuf[i];
                        max_index = i;
                    }
                }
                printf("prediction: %d\n", max_index);
                mmatrix_delete((mmatrix)output);
            }
        }
        if (save(network, backend == AKIRA_BACKEND_CPU ? "model" : "model_metal", "just some notes", FILE_VERSION) != 0) {
            printf("File was not saved\n");
        }
        nn_delete(network);
        if (backend == AKIRA_BACKEND_CPU) {
            matrix_delete((matrix)training_input);
            matrix_delete((matrix)training_output);
        } else {
            mmatrix_delete((mmatrix)training_input);
            mmatrix_delete((mmatrix)training_output);
            free(ftraining_input);
            free(ftraining_output);
        }
        clock_t end = clock();
        double elapsed = (double)(end - start) / CLOCKS_PER_SEC;
        printf("Elapsed time for %s: %.3f seconds\n", backend == AKIRA_BACKEND_CPU ? "CPU" : "Metal", elapsed);
    }
    free(dtraining_output);
    return 0;
}