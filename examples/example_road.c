#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "../src/akira.h"

#define MAX_SPEED 50

const char* getfield(char* line, int num)
{
    const char* tok;
    for (tok = strtok(line, ",");
            tok && *tok;
            tok = strtok(NULL, ",\n"))
    {
        if (!--num)
            return tok;
    }
    return NULL;
}

int main(int argc, char **argv) {
    FILE *csv, *img;
    char* tmp;
    char line[1024], center[256], left[256], right[256];
    double steering, throttle, reverse, speed;
    int i, j, number_of_iteration, value_int, x = 0;
    const char* tok;
    nn network;
    const int size_of_input = 320*160*3 * 3;
    const int size_of_output = 4;
    matrix training_input = matrix_constructor(size_of_input, 1);
    matrix training_output = matrix_constructor(size_of_output, 1);

    int sizes[] = {size_of_input, 72, 108, 144, 192, 192, 100, 50, 10, size_of_output};
    network = nn_constructor(8, sizes);

    if (argc == 2) {
        number_of_iteration = atoi(argv[1]);
    }

    if ((csv = fopen("./examples/drive_data/driving_log.csv", "r")) == NULL) {
        printf("ERROR: Could not open file driving_log.csv\n");
        return 1;
    }
    while (fgets(line, 1024, csv)) {
        tok = strtok(line, ",");
        strcpy(center, tok);
        tok = strtok(NULL, ",\n");
        strcpy(left, tok);
        tok = strtok(NULL, ",\n");
        strcpy(right, tok);
        tok = strtok(NULL, ",\n");
        sscanf(tok, "%lf", &steering);
        tok = strtok(NULL, ",\n");
        sscanf(tok, "%lf", &throttle);
        tok = strtok(NULL, ",\n");
        sscanf(tok, "%lf", &reverse);
        tok = strtok(NULL, ",\n");
        sscanf(tok, "%lf", &speed);
        printf("%d\n", x++);
        // printf("%lf %lf %lf %lf\n", steering, throttle, reverse, speed);
        training_output->p[0] = steering;
        training_output->p[1] = throttle;
        training_output->p[2] = reverse;
        training_output->p[3] = speed / MAX_SPEED;
        i = 0;
        if ((img = fopen(center, "r")) == NULL) {
            printf("ERROR: Could not open image %s\n", center);
            return 1;
        }
        while (fscanf(img, "%d\n", &value_int) != 0 && i < size_of_input / 3) {
            training_input->p[i] = (double)value_int/127.5-1.0;
            i++;
        }
        fclose(img);
        if ((img = fopen(left, "r")) == NULL) {
            printf("ERROR: Could not open image %s\n", left);
            return 1;
        }
        j = 0;
        while (fscanf(img, "%d\n", &value_int) != 0 && j < size_of_input / 3) {
            training_input->p[i] = (double)value_int/127.5-1.0;
            // printf("%d/%d\n", i, size_of_input);
            i++;
            j++;
        }
        fclose(img);
        if ((img = fopen(right, "r")) == NULL) {
            printf("ERROR: Could not open image %s\n", right);
            return 1;
        }
        j = 0;
        while (fscanf(img, "%d\n", &value_int) != 0 && j < size_of_input / 3) {
            training_input->p[i] = (double)value_int/127.5-1.0;
            i++;
            j++;
        }
        fclose(img);
        train(network, training_input, training_output);
    }

    if (save(network, "model_road", "", FILE_VERSION) != 0) {
        printf("File was not saved\n");
    }

    fclose(csv);
    matrix_delete(training_input);
    matrix_delete(training_output);
    nn_delete(network);

    return 0;
}