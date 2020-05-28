#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "../../src/akira.h"

int main(int argc, char **argv) {
    int i;
    double value;
    char number[5] = {'\0'}, image[150] = {'\0'};
    FILE *fp;
    if ((fp = fopen("images_info.txt", "r")) == NULL) {
        printf("ERROR: Could not open file images_info.txt\n");
        return 1;
    }
    fscanf(fp, "%4s", number);
    do {
        strcpy(image, "./images/img");
        strcat(image, number);
        strcat(image, ".txt");
        printf("%s\n", image);
        while (fscanf(fp, ",%lf", &value) != 0) {
            printf(",%lf", value);
        }
        printf("\n");
    } while (fscanf(fp, "\n%4s", number) != EOF);
    fclose(fp);
    return 0;
}