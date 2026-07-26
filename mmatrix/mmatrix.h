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

/* --- batching --------------------------------------------------------------
 * Operations are encoded into a shared command buffer and only submitted when
 * a result is actually needed on the CPU. One training step therefore costs a
 * single GPU round trip instead of one per operation. The functions that read
 * device memory (mmatrix_copy_to_host, mmatrix_print) flush implicitly, so
 * callers only need mmatrix_flush() before touching a buffer by other means.
 */
void mmatrix_flush(void);

/* --- allocation-free / fused variants --------------------------------------
 * These write into a caller-owned destination and fold several dispatches into
 * one kernel, which is what the per-dispatch overhead makes worthwhile.
 */

/* out = alpha * op(a) * op(b) + beta * out, op(x) is x or x**T. */
void mmatrix_gemm(mmatrix out, mmatrix a, int a_t, mmatrix b, int b_t, float alpha, float beta);
void mmatrix_mult_into(mmatrix out, mmatrix a, mmatrix b);
void mmatrix_transpose_into(mmatrix out, mmatrix m);

/* out = sigmoid(w * x + bias), the whole forward step of one layer. */
void mmatrix_layer_forward(mmatrix out, mmatrix w, mmatrix x, mmatrix bias);

/* a = a * (1 - a) * e * lr, the scaled gradient for an activated layer. */
void mmatrix_gradient(mmatrix a, mmatrix e, float lr);

/* err = target - activation */
void mmatrix_sub_into(mmatrix out, mmatrix target, mmatrix activation);

#ifdef __cplusplus
}
#endif
