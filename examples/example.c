#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include "../src/akira.h"
#include <math.h>

static const int TRAIN_SIZE = 10000;

static int cmp_double(const void *a, const void *b) { return (*(double*)a > *(double*)b) - (*(double*)a < *(double*)b); }
static int cmp_float(const void *a, const void *b) { return (*(float*)a > *(float*)b) - (*(float*)a < *(float*)b); }

double percentile_double(double *arr, int n, double p) {
    qsort(arr, n, sizeof(double), cmp_double);
    double idx = p * (n - 1);
    int i = (int)idx;
    double frac = idx - i;
    if (i + 1 < n)
        return arr[i] * (1 - frac) + arr[i + 1] * frac;
    else
        return arr[i];
}

float percentile_float(float *arr, int n, float p) {
    qsort(arr, n, sizeof(float), cmp_float);
    float idx = p * (n - 1);
    int i = (int)idx;
    float frac = idx - i;
    if (i + 1 < n)
        return arr[i] * (1 - frac) + arr[i + 1] * frac;
    else
        return arr[i];
}

int main(int argc, char **argv) {
    g_debug = 1;
    // Run both backends for comparison
    for (int backend_id = 0; backend_id < 2; backend_id++) {
        akira_backend backend = (backend_id == 0) ? AKIRA_BACKEND_CPU : AKIRA_BACKEND_METAL;
        printf("\n==============================\n");
        printf("Using backend: %s\n", backend == AKIRA_BACKEND_CPU ? "CPU (cmatrix)" : "Metal (mmatrix)");
        fflush(stdout);
        srand(42); // fixed seed for reproducibility
        int sizes[] = {2, 2, 1};
        nn network = nn_constructor(1, sizes, backend);
        if (backend == AKIRA_BACKEND_CPU) {
            network->lr = 0.5;
        } else {
            network->lr = 0.1;
        }
        void *training_input;
        void *training_output;
        double dtraining_input[] = { 1, 0,
                                    1, 1,
                                    0, 0,
                                    0, 1 };
        double dtraining_output[] = { 1, 0, 0, 1 };
        float ftraining_input[8];
        float ftraining_output[4];
        for (int k = 0; k < 8; ++k) ftraining_input[k] = (float)dtraining_input[k];
        for (int k = 0; k < 4; ++k) ftraining_output[k] = (float)dtraining_output[k];
        if (backend == AKIRA_BACKEND_CPU) {
            training_input = matrix_constructor(2, 1);
            training_output = matrix_constructor(1, 1);
        } else {
            training_input = mmatrix_constructor(2, 1);
            training_output = mmatrix_constructor(1, 1);
        }
        // TRAINING START
        for (int i = 0; i < TRAIN_SIZE; i++) {
            for (int j = 0; j < 4; j++) {
                if (backend == AKIRA_BACKEND_CPU) {
                    memcpy(((matrix)training_input)->p, dtraining_input + 2 * j, sizeof(double) * 2);
                    memcpy(((matrix)training_output)->p, dtraining_output + 1 * j, sizeof(double) * 1);
                    train(network, training_input, training_output);
                } else {
                    mmatrix_copy_from_host((mmatrix)training_input, ftraining_input + 2 * j);
                    mmatrix_copy_from_host((mmatrix)training_output, ftraining_output + 1 * j);
                    train(network, training_input, training_output);
                }
            }
            if (i % 100 == 0) {
                printf("\r%d/%d", i, TRAIN_SIZE);
                fflush(stdout);
            }
        }
        printf("\r\n");
        fflush(stdout);
        // TRAINING END

        void *output;
        for (int j = 0; j < 4; j++) {
            if (backend == AKIRA_BACKEND_CPU) {
                memcpy(((matrix)training_input)->p, dtraining_input + 2 * j, sizeof(double) * 2);
                output = run(network, training_input);
                printf("input (%lf %lf):\n", ((matrix)training_input)->p[0], ((matrix)training_input)->p[1]);
                printf("%lf\n", ((matrix)output)->p[0]);
                matrix_delete((matrix)output);
            } else {
                mmatrix_copy_from_host((mmatrix)training_input, ftraining_input + 2 * j);
                output = run(network, training_input);
                float out;
                mmatrix_copy_to_host((mmatrix)output, &out);
                float in[2];
                mmatrix_copy_to_host((mmatrix)training_input, in);
                printf("input (%f %f):\n", in[0], in[1]);
                printf("%f\n", out);
                mmatrix_delete((mmatrix)output);
            }
        }
        if (backend == AKIRA_BACKEND_CPU) {
            matrix_delete((matrix)training_input);
            matrix_delete((matrix)training_output);
        } else {
            mmatrix_delete((mmatrix)training_input);
            training_input = NULL;
            mmatrix_delete((mmatrix)training_output);
            training_output = NULL;
        }
        nn_delete(network);
    }
    return 0;
}
