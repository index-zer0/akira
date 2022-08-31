#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#if defined(_OPENMP)
    #include <omp.h>
#endif
#define STB_IMAGE_IMPLEMENTATION
#include "../src/akira.h"
//https://github.com/takafumihoriuchi/MNIST_for_C/
#include "stb_image.h"

int main(int argc, char **argv) {
    srand(time(0));
    #if defined(_OPENMP)
         double start = omp_get_wtime();
    #else
        clock_t t;
        t = clock();
    #endif
    int i, j;
    const int size_of_input = 224 * 224;
    const int size_of_output = 2;
    int train_number = 100;
    int number_of_iteration = 10;
    if (argc >= 2) {
        number_of_iteration = atoi(argv[1]);
    }
    if (argc >= 3) {
        train_number = atoi(argv[2]);
    }
    int width, height, bpp;

    int sizes[] = {
        size_of_input,
        128,
        128,
        size_of_output};
    nn network = nn_constructor(2, sizes);
    matrix training_input = matrix_constructor(size_of_input, 1);
    matrix training_output = matrix_constructor(size_of_output, 1);
    int cat = 0, dog = 0;
    for (i = 0; i < number_of_iteration; i++) {
        for (j = 0; j < train_number; j++) {
            char filename[100];

            int index = rand() % 12000 + 0;
            if (index%2 == 0) {
                snprintf(filename, 100, "data/Cat/%d.jpg", index);
                training_output->p[0] = 1.0;
                training_output->p[1] = 0.0;
            } else {
                snprintf(filename, 100, "data/Dog/%d.jpg", index);
                training_output->p[0] = 0.0;
                training_output->p[1] = 1.0;
            }
            float* rgb_image = stbi_loadf(filename, &width, &height, &bpp, 1);
             if (rgb_image == NULL) {
                continue;
            }
            // better way?
            #pragma omp parallel
            #pragma omp for
            for (int k = 0; k < size_of_input; k++) {
                // memcpy(training_input->p, rgb_image, sizeof(double) * size_of_input);
                 training_input->p[k] = (double)rgb_image[k];   
            }
            stbi_image_free(rgb_image);
            train(network, training_input, training_output);
            printf("%d/%d - %lf%%\r", i, number_of_iteration, (double)((double)j/(double)train_number * 100.0));
            fflush(stdout);
        }
        //if ((i % 100) == 0) {
        // printf("%lf%%\r", (double)((double)i/(double)number_of_iteration * 100.0));
        // fflush(stdout);
        //}
    }
    #if defined(_OPENMP)
        double end = omp_get_wtime();
        printf("Parallel Work took %f seconds\n", end - start);
    #else
        t = clock() - t;
        double time_taken = ((double)t)/CLOCKS_PER_SEC;
        printf("Work took %f seconds\n", time_taken);
    #endif
    printf("Done training\n");
    matrix output;
    int correct = 0;
    cat = 0;
    dog = 0;
    for (j = 0; j < 10; j++) {
        char filename[100];
        if (j%2 == 0) {
            snprintf(filename, 100, "data/Cat/%d.jpg", train_number + cat++);
        } else {
            snprintf(filename, 100, "data/Dog/%d.jpg", train_number + dog++);
        }
        float* rgb_image = stbi_loadf(filename, &width, &height, &bpp, 1);
        for (int k = 0; k < size_of_input; k++) {
            // memcpy(training_input->p, rgb_image, sizeof(double) * size_of_input);
            training_input->p[k] = (double)rgb_image[k];   
        }
        stbi_image_free(rgb_image);
        output = run(network, training_input);
        printf("\ninput (%s):\n", filename);
        if (output->p[0] >= output->p[1]) {
            printf("prediction: Cat\n");
            if (j%2 == 0) {
                correct++;
            }
        } else {
            printf("prediction: Dog\n");
            if (j%2 == 1) {
                correct++;
            }
        }
        printf("%lf - %lf\n", output->p[0], output->p[1]);
        matrix_delete(output);
    }
    printf("Accuracy: %d/%d (%f%%)\n", correct, 10, (double)correct*10.0);
    if (save(network, "model", "just some notes", FILE_VERSION) != 0) {
        printf("File was not saved\n");
    }
    
    nn_delete(network);
     return 0;
}
