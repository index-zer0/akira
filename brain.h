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

double sigmoid(double number);
double sigmoid_derivative(double number);
void apply_sigmoid(matrix a);
double get_random(double min, double max);
double dot(double *a, double *b, int len, int step);
matrix mat_new(int h, int w);
matrix mat_mul(matrix a, matrix b);
void mat_show(matrix a);
matrix calculate_error(matrix a, matrix b);
matrix calculate_adjustments(matrix a, matrix b);
void update_synaptic_weights(matrix weights, matrix input, matrix adjustments);
matrix transpose(matrix a);
void validate_matrix(matrix a);
void train(matrix training_input, matrix training_output, matrix synaptic_weights, int iterations);
matrix run(matrix synaptic_weights, matrix test_input);