#include <stdio.h>
#include <stdlib.h>
#include <math.h> 
#include <time.h>

typedef struct { 
    int h, w; 
    double *x;
} matrix_t, *matrix;

typedef struct neuron {
    matrix inputs;
    matrix weights;
    double bias;
    double output;
} neuron;

static inline double sigmoid(double number) {
    return (1 / (1 + exp(-number)));
}

static inline double sigmoid_derivative(double number) {
    return number * (1 - number);
}

void apply_sigmoid(matrix a) {
    int i;
    for (i = 0; i < a->h * a->w; i++) {
        a->x[i] = sigmoid(a->x[i]);
    }
}

static inline double get_random(double min, double max) {
    return (double)rand()/RAND_MAX*max+min;
}
 
static inline double dot(double *a, double *b, int len, int step) {
	double r = 0;
	while (len--) {
		r += *a++ * *b;
		b += step;
	}
	return r;
}
 
matrix mat_new(int h, int w) {
	matrix r = malloc(sizeof(matrix_t) + sizeof(double) * w * h);
	r->h = h, r->w = w;
	r->x = (double*)(r + 1);
	return r;
}
 
matrix mat_mul(matrix a, matrix b) {
	matrix r;
	double *p, *pa;
	int i, j;
	if (a->w != b->h) return 0;
 
	r = mat_new(a->h, b->w);
	p = r->x;
	for (pa = a->x, i = 0; i < a->h; i++, pa += a->w)
		for (j = 0; j < b->w; j++)
			*p++ = dot(pa, b->x + j, a->w, b->w);
	return r;
}
 
void mat_show(matrix a) {
	int i, j;
	double *p = a->x;
	for (i = 0; i < a->h; i++, putchar('\n'))
		for (j = 0; j < a->w; j++)
			printf("  %lf", *p++);
	putchar('\n');
}

matrix calculate_error(matrix a, matrix b) {
    matrix r;
    int i = 0;
    if (a->h != b->h || a->w != b->w) {
        return NULL;
    }
    r = mat_new(a->h, a->w);
    for (i = 0; i < r->h * r->w; i++) {
        r->x[i] = a->x[i] - b->x[i];
    }
    return r;
}

matrix calculate_adjustments(matrix a, matrix b) {
    matrix r;
    int i = 0;
    if (a->h != b->h || a->w != b->w) {
        return NULL;
    }
    r = mat_new(a->h, a->w);
    for (i = 0; i < r->h * r->w; i++) {
        r->x[i] = a->x[i] * sigmoid_derivative(b->x[i]);
    }
    return r;
}

void update_synaptic_weights(matrix weights, matrix input, matrix adjustments) {
    matrix change = mat_mul(input, adjustments);
    int i = 0;
    if (weights->h != change->h || weights->w != change->w) {
        return;
    }
    for (i = 0; i < change->h * change->w; i++) {
        weights->x[i] += change->x[i];
    }
    free(change);
}

matrix transpose(matrix a) {
    matrix tr = mat_new(a->w, a->h);
    int i, j;
    for (i = 0; i < a->w; i++) {
        for (j = 0; j < a->h; j++) {
            tr->x[j + i*a->h] = a->x[i + j*a->w];
        }
    }
    return tr;
}

void validate_matrix(matrix a) {
    if (a->h <= 0) {
        printf("ERROR: Height of matrix should be greater than 0 (%d)\n", a->h);
        exit(1);
    }
    if (a->w <= 0) {
        printf("ERROR: Width of matrix should be greater than 0 (%d)\n", a->w);
        exit(1);
    }
}

void train(matrix training_input, matrix training_output, matrix synaptic_weights, int iterations) {
    int i;
    matrix output = NULL, error = NULL, adjustments = NULL, tr = NULL;
    srand(time(0));
    validate_matrix(training_input);
    validate_matrix(training_output);
    if (training_input->h != training_output->h) {
        printf("ERROR: Number of training input (%d) is different than the number of training output (%d)\n", training_input->h, training_output->h);
        exit(1);
    }
    if (iterations <= 0) {
        printf("ERROR: Number of iterations should be be greater than 0 (Asked for %d iterations)\n", iterations);
        exit(1);
    }

    for (i = 0; i < iterations; i++) {
        if (output != NULL) {
            free(output);
        }
        output = mat_mul(training_input, synaptic_weights);
        apply_sigmoid(output);
        if (error != NULL) {
            free(error);
        }
        error = calculate_error(training_output, output);
        if (adjustments != NULL) {
            free(adjustments);
        }
        adjustments = calculate_adjustments(error, output);
        if (tr != NULL) {
            free(tr);
        }
        tr = transpose(training_input);
        update_synaptic_weights(synaptic_weights, tr, adjustments);
    }
    free(output);
    free(error);
    free(adjustments);
    free(tr);
}

matrix run(matrix synaptic_weights, matrix test_input) {
    matrix output = mat_mul(test_input, synaptic_weights);
    apply_sigmoid(output);
    return output;
}

int main(void) {
    int i;
    double dtraining_input[] = {
        0, 0, 1,
        1, 1, 1,
        1, 0, 1,
        0, 1, 1
    };
    double dtraining_output[] = {
        0,
        1,
        1,
        0
    };
    double dtest_input[] = {
        1,
        1,
        1
    };
    matrix_t training_input = { 4, 3, dtraining_input};
    matrix_t training_output = { 4, 1, dtraining_output};
    matrix_t test_input = { 1, 3, dtest_input};
    double dsynaptic_weights[training_input.w];
    for (i = 0; i < training_input.w; i++) {
        dsynaptic_weights[i] = get_random(-1.0, 1.0);
    }
    matrix_t synaptic_weights = { training_input.w, 1, dsynaptic_weights};

    train(&training_input, &training_output, &synaptic_weights, 20000);
    matrix output = run(&synaptic_weights, &test_input);
    mat_show(output);
    free(output);
}