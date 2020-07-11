#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "../src/akira.h"

#define MAX_SPEED 50

int main(int argc, char **argv) {
    FILE *csv, *img;
    int i, j, x = 0, value_int;
    nn network;
    matrix output;
    const char* tok;
    char line[1024], center[256], left[256], right[256];
    double steering, throttle, reverse, speed;
    double steering_loss, throttle_loss, reverse_loss, speed_loss;
    const int size_of_input = 320*160*3 * 3;
    matrix training_input = matrix_constructor(size_of_input, 1);

    network = load("model_road.akr");

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
        
        output = run(network, training_input);
        x++;
        steering_loss += fabs((output->p[0]) - (steering));
        throttle_loss += fabs((output->p[1]) - (throttle));
        reverse_loss += fabs((output->p[2]) - (reverse));
        speed_loss += fabs((output->p[3]) - ((speed / MAX_SPEED)));
        printf("%d: %lf %lf %lf %lf\n", x, steering_loss / x, throttle_loss / x, reverse_loss / x, speed_loss / x);
        matrix_delete(output);
    }
    fclose(csv);
    matrix_delete(training_input);
    nn_delete(network);
    return 0;
}