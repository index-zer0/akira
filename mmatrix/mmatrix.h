#pragma once

#ifdef __cplusplus
extern "C" {
#endif

// Opaque pointer type for the matrix structure
typedef struct _mmatrix* mmatrix;

// Constructor and Destructor
mmatrix mmatrix_constructor(int rows, int columns);
void mmatrix_delete(mmatrix m);

// Getters for matrix properties
int mmatrix_get_rows(mmatrix m);
int mmatrix_get_cols(mmatrix m);

// Data transfer functions
void mmatrix_copy(mmatrix dest, mmatrix src);
void mmatrix_copy_from_host(mmatrix m, float *host_data);
void mmatrix_copy_to_host(mmatrix m, float *host_data);

// Matrix arithmetic and operations
void mmatrix_add(mmatrix m, mmatrix a);
void mmatrix_sub(mmatrix m, mmatrix a);
void mmatrix_hadamard(mmatrix m, mmatrix a);
void mmatrix_scalar_mult(mmatrix m, float scalar);
void mmatrix_apply(mmatrix m, int op);

mmatrix mmatrix_mult(mmatrix a, mmatrix b);
mmatrix mmatrix_transpose(mmatrix m);

// Utility functions
void mmatrix_randomize(mmatrix m, float min, float max);
void mmatrix_print(mmatrix m);

#ifdef __cplusplus
}
#endif