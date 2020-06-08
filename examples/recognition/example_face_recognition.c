#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "../../src/akira.h"

int main(int argc, char **argv) {
    int i, j, value_int;
    double value;
    char number[5] = {'\0'}, image[150] = {'\0'};
    FILE *fp, *img;
    const int size_of_input = 400*400*3;
    const int size_of_output = 5;
    const int train_number = 100;
    int number_of_iteration = 10;
    int face_exists = 0;
    matrix training_input = matrix_constructor(size_of_input, 1);
    matrix training_output = matrix_constructor(size_of_output, 1);
    memset(training_output->p, 0.0, sizeof(double) * size_of_output);
    if (argc == 2) {
        number_of_iteration = atoi(argv[1]);
    }
    int sizes[] = {size_of_input, 64, 192, 128, 256, 256, 512, 256, 512, 256, 512, 256, 512, 256, 512, 512, 1024, 512, \
        1024, 512, 1024, 1024, 1024, 1024, 256, size_of_output};
    nn network = nn_constructor(24, sizes);

    if ((fp = fopen("./examples/recognition/images_info.txt", "r")) == NULL) {
        printf("ERROR: Could not open file images_info.txt\n");
        return 1;
    }
    fscanf(fp, "%4s", number);
    do {
        strcpy(image, "./examples/recognition/images/img");
        strcat(image, number);
        strcat(image, ".txt");
        //printf("%s\n", image);
        i = 1;
        while (fscanf(fp, ",%lf", &value) != 0) {
            //printf(", %lf", value);
            if (i < 5) {
                training_output->p[i] = value;
                i++;
            }
        }
        if (i > 5) {
            training_output->p[0] = 1;
        } else {
            training_output->p[0] = 0;
        }
        printf("%s: %d\n", image, number_of_iteration);
        if ((img = fopen(image, "r")) == NULL) {
            printf("ERROR: Could not open file %s\n", image);
            return 1;
        }
        i = 0;
        while (fscanf(img, "%d\n", &value_int) != 0 && i < size_of_input) {
            training_input->p[i] = (double)value_int/255.0;
            /*if (i%100000 == 0) {
                printf("%d: %lf\n", i, training_input->p[i]);
            }*/
            i++;
        }
        fclose(img);
        train(network, training_input, training_output);
    } while (fscanf(fp, "\n%4s", number) != EOF && number_of_iteration--);
    fclose(fp);
    if ((img = fopen("./examples/recognition/images/img0000.txt", "r")) == NULL) {
        printf("ERROR: Could not open file %s\n", image);
        return 1;
    }
    i = 0;
    while (fscanf(img, "%d\n", &value_int) != 0 && i < size_of_input) {
        training_input->p[i] = (double)value_int/255.0;
        i++;
    }
    fclose(img);
    matrix output = run(network, training_input);
    for (i = 0; i < output->rows * output->columns; i++) {
        printf("%lf\n", output->p[i]);
    }
    nn_delete(network);
    matrix_delete(training_input);
    matrix_delete(training_output);
    return 0;
}