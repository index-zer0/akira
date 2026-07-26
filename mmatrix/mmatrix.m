#import "mmatrix.h"
#import "../src/akira.h"
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <string.h>
#include <stdint.h>

/* Kernels are written for the Apple Silicon GPU: every global read is indexed
 * so that adjacent threads touch adjacent addresses, and the reductions use
 * simdgroup intrinsics rather than threadgroup memory. */
static const char *kMatrixShaders =
"#include <metal_stdlib>\n"
"using namespace metal;\n"
"\n"
"#define FLAG_BIAS    1u\n"
"#define FLAG_SIGMOID 2u\n"
"\n"
/* y = A(MxN) * x(N), one simdgroup per output row. Lanes walk the row so the
 * loads coalesce, then simd_sum folds the partial products in registers. */
"kernel void gemv(const device float* A [[buffer(0)]],\n"
"                 const device float* x [[buffer(1)]],\n"
"                 const device float* bias [[buffer(2)]],\n"
"                 device float* y [[buffer(3)]],\n"
"                 constant uint& M [[buffer(4)]],\n"
"                 constant uint& N [[buffer(5)]],\n"
"                 constant uint& flags [[buffer(6)]],\n"
"                 uint tg [[threadgroup_position_in_grid]],\n"
"                 uint sg [[simdgroup_index_in_threadgroup]],\n"
"                 uint sgs [[simdgroups_per_threadgroup]],\n"
"                 uint lane [[thread_index_in_simdgroup]],\n"
"                 uint w [[threads_per_simdgroup]]) {\n"
"    uint row = tg * sgs + sg;\n"
"    if (row >= M) return;\n"
"    const device float* Arow = A + (ulong)row * N;\n"
"    float acc = 0.0f;\n"
"    for (uint n = lane; n < N; n += w) acc += Arow[n] * x[n];\n"
"    float s = simd_sum(acc);\n"
"    if (lane == 0) {\n"
"        if (flags & FLAG_BIAS) s += bias[row];\n"
"        if (flags & FLAG_SIGMOID) s = 1.0f / (1.0f + exp(-s));\n"
"        y[row] = s;\n"
"    }\n"
"}\n"
"\n"
/* y = A(MxN)**T * x(M), one thread per output element. Threads are indexed by
 * column so the strided walk down A still reads consecutive addresses. */
"kernel void gemv_t(const device float* A [[buffer(0)]],\n"
"                   const device float* x [[buffer(1)]],\n"
"                   device float* y [[buffer(2)]],\n"
"                   constant uint& M [[buffer(3)]],\n"
"                   constant uint& N [[buffer(4)]],\n"
"                   uint n [[thread_position_in_grid]]) {\n"
"    if (n >= N) return;\n"
"    float s = 0.0f;\n"
"    for (uint m = 0; m < M; ++m) s += A[(ulong)m * N + n] * x[m];\n"
"    y[n] = s;\n"
"}\n"
"\n"
/* A(MxN) += alpha * x(M) * y(N)**T -- the rank-1 weight update. */
"kernel void ger(device float* A [[buffer(0)]],\n"
"                const device float* x [[buffer(1)]],\n"
"                const device float* y [[buffer(2)]],\n"
"                constant uint& M [[buffer(3)]],\n"
"                constant uint& N [[buffer(4)]],\n"
"                constant float& alpha [[buffer(5)]],\n"
"                uint2 gid [[thread_position_in_grid]]) {\n"
"    uint n = gid.x, m = gid.y;\n"
"    if (m >= M || n >= N) return;\n"
"    A[(ulong)m * N + n] += alpha * x[m] * y[n];\n"
"}\n"
"\n"
/* C(MxK) = A(MxN) * B(NxK), general case. */
"kernel void matmul(const device float* A [[buffer(0)]],\n"
"                   const device float* B [[buffer(1)]],\n"
"                   device float* C [[buffer(2)]],\n"
"                   constant uint& M [[buffer(3)]],\n"
"                   constant uint& N [[buffer(4)]],\n"
"                   constant uint& K [[buffer(5)]],\n"
"                   uint2 gid [[thread_position_in_grid]]) {\n"
"    uint col = gid.x, row = gid.y;\n"
"    if (row >= M || col >= K) return;\n"
"    float sum = 0.0f;\n"
"    for (uint i = 0; i < N; ++i) sum += A[(ulong)row * N + i] * B[(ulong)i * K + col];\n"
"    C[(ulong)row * K + col] = sum;\n"
"}\n"
"\n"
"kernel void madd(device float* A [[buffer(0)]], const device float* B [[buffer(1)]],\n"
"                 constant uint& size [[buffer(2)]], uint gid [[thread_position_in_grid]]) {\n"
"    if (gid >= size) return;\n"
"    A[gid] += B[gid];\n"
"}\n"
"\n"
"kernel void msub(device float* A [[buffer(0)]], const device float* B [[buffer(1)]],\n"
"                 constant uint& size [[buffer(2)]], uint gid [[thread_position_in_grid]]) {\n"
"    if (gid >= size) return;\n"
"    A[gid] -= B[gid];\n"
"}\n"
"\n"
"kernel void msub_into(device float* O [[buffer(0)]], const device float* A [[buffer(1)]],\n"
"                      const device float* B [[buffer(2)]], constant uint& size [[buffer(3)]],\n"
"                      uint gid [[thread_position_in_grid]]) {\n"
"    if (gid >= size) return;\n"
"    O[gid] = A[gid] - B[gid];\n"
"}\n"
"\n"
"kernel void mhadamard(device float* A [[buffer(0)]], const device float* B [[buffer(1)]],\n"
"                      constant uint& size [[buffer(2)]], uint gid [[thread_position_in_grid]]) {\n"
"    if (gid >= size) return;\n"
"    A[gid] *= B[gid];\n"
"}\n"
"\n"
"kernel void mscalar_mult(device float* A [[buffer(0)]], constant float& scalar [[buffer(1)]],\n"
"                         constant uint& size [[buffer(2)]], uint gid [[thread_position_in_grid]]) {\n"
"    if (gid >= size) return;\n"
"    A[gid] *= scalar;\n"
"}\n"
"\n"
/* op 0: sigmoid. op 1: derivative expressed on the already-activated value,
 * x * (1 - x), matching the CPU backend. */
"kernel void mapply(device float* A [[buffer(0)]], constant uint& size [[buffer(1)]],\n"
"                   constant int& op [[buffer(2)]], uint gid [[thread_position_in_grid]]) {\n"
"    if (gid >= size) return;\n"
"    float x = A[gid];\n"
"    if (op == 0) A[gid] = 1.0f / (1.0f + exp(-x));\n"
"    else if (op == 1) A[gid] = x * (1.0f - x);\n"
"}\n"
"\n"
/* a = a * (1 - a) * e * lr: derivative, error and learning rate in one pass. */
"kernel void mgradient(device float* A [[buffer(0)]], const device float* E [[buffer(1)]],\n"
"                      constant uint& size [[buffer(2)]], constant float& lr [[buffer(3)]],\n"
"                      uint gid [[thread_position_in_grid]]) {\n"
"    if (gid >= size) return;\n"
"    float v = A[gid];\n"
"    A[gid] = v * (1.0f - v) * E[gid] * lr;\n"
"}\n"
"\n"
"kernel void mtranspose(const device float* A [[buffer(0)]], device float* T [[buffer(1)]],\n"
"                       constant uint& rows [[buffer(2)]], constant uint& cols [[buffer(3)]],\n"
"                       uint2 gid [[thread_position_in_grid]]) {\n"
"    uint col = gid.x, row = gid.y;\n"
"    if (row >= rows || col >= cols) return;\n"
"    T[(ulong)col * rows + row] = A[(ulong)row * cols + col];\n"
"}\n";

/* Pipelines are compiled once at startup. Building them per call, as this
 * previously did, cost far more than the kernels themselves. */
typedef enum {
    PIPE_GEMV = 0, PIPE_GEMV_T, PIPE_GER, PIPE_MATMUL,
    PIPE_ADD, PIPE_SUB, PIPE_SUB_INTO, PIPE_HADAMARD,
    PIPE_SCALAR_MULT, PIPE_APPLY, PIPE_GRADIENT, PIPE_TRANSPOSE,
    PIPE_COUNT
} pipe_id;

static const char *kPipeNames[PIPE_COUNT] = {
    "gemv", "gemv_t", "ger", "matmul",
    "madd", "msub", "msub_into", "mhadamard",
    "mscalar_mult", "mapply", "mgradient", "mtranspose"
};

@interface MMMetalContext : NSObject
@property (nonatomic, strong) id<MTLDevice> device;
@property (nonatomic, strong) id<MTLLibrary> library;
@property (nonatomic, strong) id<MTLCommandQueue> queue;
@property (nonatomic, strong) NSArray<id<MTLComputePipelineState>> *pipelines;
+ (instancetype)shared;
@end

@implementation MMMetalContext
+ (instancetype)shared {
    static MMMetalContext *ctx;
    static dispatch_once_t onceToken;
    dispatch_once(&onceToken, ^{
        ctx = [MMMetalContext new];
        ctx.device = MTLCreateSystemDefaultDevice();
        if (!ctx.device) {
            NSLog(@"Failed to create Metal device");
            return;
        }
        NSError *err = nil;
        NSString *src = [NSString stringWithUTF8String:kMatrixShaders];
        MTLCompileOptions *opts = [MTLCompileOptions new];
        if (@available(macOS 15.0, *)) {
            opts.mathMode = MTLMathModeFast;
        } else {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
            opts.fastMathEnabled = YES;
#pragma clang diagnostic pop
        }
        ctx.library = [ctx.device newLibraryWithSource:src options:opts error:&err];
        if (!ctx.library) {
            NSLog(@"Metal library error: %@", err);
            return;
        }
        ctx.queue = [ctx.device newCommandQueue];
        NSMutableArray *pipes = [NSMutableArray arrayWithCapacity:PIPE_COUNT];
        for (int i = 0; i < PIPE_COUNT; i++) {
            id<MTLFunction> fn = [ctx.library newFunctionWithName:
                                  [NSString stringWithUTF8String:kPipeNames[i]]];
            if (!fn) {
                NSLog(@"Metal function %s not found", kPipeNames[i]);
                return;
            }
            id<MTLComputePipelineState> p = [ctx.device newComputePipelineStateWithFunction:fn error:&err];
            if (!p) {
                NSLog(@"Pipeline error for %s: %@", kPipeNames[i], err);
                return;
            }
            [pipes addObject:p];
        }
        ctx.pipelines = pipes;
    });
    return ctx;
}
@end

struct _mmatrix {
    int rows;
    int columns;
    id<MTLBuffer> metalBuffer;
};

/* One command buffer collects many dispatches; it is submitted only when the
 * host needs a result. Dispatches inside a single compute encoder are ordered
 * with automatic barriers, so the data dependencies between them hold. */
static id<MTLCommandBuffer> g_cmd = nil;
static id<MTLComputeCommandEncoder> g_enc = nil;
static id<MTLCommandBuffer> g_inflight = nil;
static int g_encoded = 0;

/* Cap on dispatches per command buffer, so a long run of unflushed work does
 * not pin an unbounded number of resources. */
#define MM_MAX_ENCODED 1024

static id<MTLCommandBuffer> mm_command_buffer(MMMetalContext *ctx) {
    if (!g_cmd) {
        g_cmd = [ctx.queue commandBuffer];
        g_encoded = 0;
    }
    return g_cmd;
}

static id<MTLComputeCommandEncoder> mm_encoder(MMMetalContext *ctx) {
    id<MTLCommandBuffer> cmd = mm_command_buffer(ctx);
    if (!g_enc) {
        g_enc = [cmd computeCommandEncoder];
    }
    return g_enc;
}

static void mm_submit(void) {
    if (g_enc) {
        [g_enc endEncoding];
        g_enc = nil;
    }
    if (g_cmd) {
        [g_cmd commit];
        g_inflight = g_cmd;
        g_cmd = nil;
    }
}

static void mm_encoded_one(void) {
    if (++g_encoded >= MM_MAX_ENCODED) {
        mm_submit();
    }
}

void mmatrix_flush(void) {
    mm_submit();
    if (g_inflight) {
        /* The queue is serial, so waiting on the newest command buffer waits
         * for every one before it too. */
        [g_inflight waitUntilCompleted];
        g_inflight = nil;
    }
}

/* Elementwise kernels are all launched the same way. */
static void mm_dispatch_1d(id<MTLComputeCommandEncoder> enc,
                           id<MTLComputePipelineState> pipeline, NSUInteger n) {
    NSUInteger tg = MIN(pipeline.maxTotalThreadsPerThreadgroup, (NSUInteger)256);
    if (tg > n) tg = n;
    if (tg == 0) return;
    [enc dispatchThreads:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
}

mmatrix mmatrix_constructor(int rows, int columns) {
    if (rows <= 0 || columns <= 0) {
        printf("[ERROR] Invalid matrix dimensions: rows=%d, cols=%d\n", rows, columns);
        return NULL;
    }

    MMMetalContext *ctx = [MMMetalContext shared];
    if (!ctx.device || !ctx.pipelines) {
        printf("[ERROR] Metal device not available\n");
        return NULL;
    }

    mmatrix m = calloc(1, sizeof(struct _mmatrix));
    if (!m) {
        printf("[ERROR] Failed to allocate memory for matrix\n");
        return NULL;
    }

    m->rows = rows;
    m->columns = columns;
    NSUInteger size = sizeof(float) * (NSUInteger)rows * columns;

    /* Shared storage is the right mode on Apple Silicon: CPU and GPU address
     * the same physical pages, so there is nothing to stage or synchronise. */
    m->metalBuffer = [ctx.device newBufferWithLength:size options:MTLResourceStorageModeShared];
    if (!m->metalBuffer) {
        printf("[ERROR] Failed to create Metal buffer\n");
        free(m);
        return NULL;
    }
    return m;
}

void mmatrix_delete(mmatrix m) {
    if (m) {
        /* The buffer may still be referenced by work that has not run yet. */
        mmatrix_flush();
        m->metalBuffer = nil;
        free(m);
    }
}

void mmatrix_copy_from_host(mmatrix m, float *host_data) {
    if (!m || !m->metalBuffer || !host_data) {
        printf("[ERROR] Invalid parameters for mmatrix_copy_from_host\n");
        return;
    }
    /* Overwriting a buffer the GPU may still be reading. */
    mmatrix_flush();
    memcpy([m->metalBuffer contents], host_data, sizeof(float) * m->rows * m->columns);
}

void mmatrix_copy_to_host(mmatrix m, float *host_data) {
    if (!m || !m->metalBuffer || !host_data) {
        printf("[ERROR] Invalid parameters for mmatrix_copy_to_host\n");
        return;
    }
    mmatrix_flush();
    memcpy(host_data, [m->metalBuffer contents], sizeof(float) * m->rows * m->columns);
}

/* --- elementwise ---------------------------------------------------------- */

static void mm_binary_op(mmatrix m, mmatrix a, pipe_id pipe, const char *what) {
    if (!m || !a || !m->metalBuffer || !a->metalBuffer) {
        printf("[ERROR] Invalid matrices for %s\n", what);
        return;
    }
    if (m->rows != a->rows || m->columns != a->columns) {
        printf("[ERROR] Matrix size mismatch for %s\n", what);
        return;
    }
    MMMetalContext *ctx = [MMMetalContext shared];
    id<MTLComputePipelineState> pipeline = ctx.pipelines[pipe];
    id<MTLComputeCommandEncoder> enc = mm_encoder(ctx);
    [enc setComputePipelineState:pipeline];
    [enc setBuffer:m->metalBuffer offset:0 atIndex:0];
    [enc setBuffer:a->metalBuffer offset:0 atIndex:1];
    uint size = (uint)(m->rows * m->columns);
    [enc setBytes:&size length:sizeof(uint) atIndex:2];
    mm_dispatch_1d(enc, pipeline, size);
    mm_encoded_one();
}

void mmatrix_add(mmatrix m, mmatrix a) { mm_binary_op(m, a, PIPE_ADD, "addition"); }
void mmatrix_sub(mmatrix m, mmatrix a) { mm_binary_op(m, a, PIPE_SUB, "subtraction"); }
void mmatrix_hadamard(mmatrix m, mmatrix a) { mm_binary_op(m, a, PIPE_HADAMARD, "Hadamard product"); }

void mmatrix_sub_into(mmatrix out, mmatrix target, mmatrix activation) {
    if (!out || !target || !activation) {
        printf("[ERROR] Invalid matrices for subtraction\n");
        return;
    }
    if (out->rows != target->rows || out->columns != target->columns ||
        out->rows != activation->rows || out->columns != activation->columns) {
        printf("[ERROR] Matrix size mismatch for subtraction\n");
        return;
    }
    MMMetalContext *ctx = [MMMetalContext shared];
    id<MTLComputePipelineState> pipeline = ctx.pipelines[PIPE_SUB_INTO];
    id<MTLComputeCommandEncoder> enc = mm_encoder(ctx);
    [enc setComputePipelineState:pipeline];
    [enc setBuffer:out->metalBuffer offset:0 atIndex:0];
    [enc setBuffer:target->metalBuffer offset:0 atIndex:1];
    [enc setBuffer:activation->metalBuffer offset:0 atIndex:2];
    uint size = (uint)(out->rows * out->columns);
    [enc setBytes:&size length:sizeof(uint) atIndex:3];
    mm_dispatch_1d(enc, pipeline, size);
    mm_encoded_one();
}

void mmatrix_scalar_mult(mmatrix m, float number) {
    if (!m || !m->metalBuffer) {
        printf("[ERROR] Invalid matrix for scalar multiplication\n");
        return;
    }
    MMMetalContext *ctx = [MMMetalContext shared];
    id<MTLComputePipelineState> pipeline = ctx.pipelines[PIPE_SCALAR_MULT];
    id<MTLComputeCommandEncoder> enc = mm_encoder(ctx);
    [enc setComputePipelineState:pipeline];
    [enc setBuffer:m->metalBuffer offset:0 atIndex:0];
    [enc setBytes:&number length:sizeof(float) atIndex:1];
    uint size = (uint)(m->rows * m->columns);
    [enc setBytes:&size length:sizeof(uint) atIndex:2];
    mm_dispatch_1d(enc, pipeline, size);
    mm_encoded_one();
}

void mmatrix_apply(mmatrix m, int op) {
    if (!m || !m->metalBuffer) {
        printf("[ERROR] Invalid matrix for apply operation\n");
        return;
    }
    MMMetalContext *ctx = [MMMetalContext shared];
    id<MTLComputePipelineState> pipeline = ctx.pipelines[PIPE_APPLY];
    id<MTLComputeCommandEncoder> enc = mm_encoder(ctx);
    [enc setComputePipelineState:pipeline];
    [enc setBuffer:m->metalBuffer offset:0 atIndex:0];
    uint size = (uint)(m->rows * m->columns);
    [enc setBytes:&size length:sizeof(uint) atIndex:1];
    [enc setBytes:&op length:sizeof(int) atIndex:2];
    mm_dispatch_1d(enc, pipeline, size);
    mm_encoded_one();
}

void mmatrix_gradient(mmatrix a, mmatrix e, float lr) {
    if (!a || !e || a->rows != e->rows || a->columns != e->columns) {
        printf("[ERROR] Matrix size mismatch for gradient\n");
        return;
    }
    MMMetalContext *ctx = [MMMetalContext shared];
    id<MTLComputePipelineState> pipeline = ctx.pipelines[PIPE_GRADIENT];
    id<MTLComputeCommandEncoder> enc = mm_encoder(ctx);
    [enc setComputePipelineState:pipeline];
    [enc setBuffer:a->metalBuffer offset:0 atIndex:0];
    [enc setBuffer:e->metalBuffer offset:0 atIndex:1];
    uint size = (uint)(a->rows * a->columns);
    [enc setBytes:&size length:sizeof(uint) atIndex:2];
    [enc setBytes:&lr length:sizeof(float) atIndex:3];
    mm_dispatch_1d(enc, pipeline, size);
    mm_encoded_one();
}

/* --- products ------------------------------------------------------------- */

/* y = op(A) * x with x a column vector, optionally adding bias and applying
 * the sigmoid in the same kernel. */
static void mm_gemv(mmatrix out, mmatrix A, int a_t, mmatrix x, mmatrix bias, uint flags) {
    MMMetalContext *ctx = [MMMetalContext shared];
    id<MTLComputeCommandEncoder> enc = mm_encoder(ctx);
    uint M = (uint)A->rows, N = (uint)A->columns;

    if (a_t) {
        id<MTLComputePipelineState> pipeline = ctx.pipelines[PIPE_GEMV_T];
        [enc setComputePipelineState:pipeline];
        [enc setBuffer:A->metalBuffer offset:0 atIndex:0];
        [enc setBuffer:x->metalBuffer offset:0 atIndex:1];
        [enc setBuffer:out->metalBuffer offset:0 atIndex:2];
        [enc setBytes:&M length:sizeof(uint) atIndex:3];
        [enc setBytes:&N length:sizeof(uint) atIndex:4];
        mm_dispatch_1d(enc, pipeline, N);
    } else {
        id<MTLComputePipelineState> pipeline = ctx.pipelines[PIPE_GEMV];
        [enc setComputePipelineState:pipeline];
        [enc setBuffer:A->metalBuffer offset:0 atIndex:0];
        [enc setBuffer:x->metalBuffer offset:0 atIndex:1];
        /* buffer 2 is only read when FLAG_BIAS is set, but it must still be
         * bound, so point it at the output when there is no bias. */
        [enc setBuffer:(bias ? bias->metalBuffer : out->metalBuffer) offset:0 atIndex:2];
        [enc setBuffer:out->metalBuffer offset:0 atIndex:3];
        [enc setBytes:&M length:sizeof(uint) atIndex:4];
        [enc setBytes:&N length:sizeof(uint) atIndex:5];
        [enc setBytes:&flags length:sizeof(uint) atIndex:6];
        /* One simdgroup per row; the threadgroup width must be a multiple of
         * the simd width for the row indexing above to be exact. */
        NSUInteger w = pipeline.threadExecutionWidth;
        NSUInteger tgSize = MIN(pipeline.maxTotalThreadsPerThreadgroup, (NSUInteger)256);
        tgSize = (tgSize / w) * w;
        if (tgSize == 0) tgSize = w;
        NSUInteger rowsPerGroup = tgSize / w;
        NSUInteger groups = (M + rowsPerGroup - 1) / rowsPerGroup;
        [enc dispatchThreadgroups:MTLSizeMake(groups, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(tgSize, 1, 1)];
    }
    mm_encoded_one();
}

/* A += alpha * x * y**T */
static void mm_ger(mmatrix A, mmatrix x, mmatrix y, float alpha) {
    MMMetalContext *ctx = [MMMetalContext shared];
    id<MTLComputePipelineState> pipeline = ctx.pipelines[PIPE_GER];
    id<MTLComputeCommandEncoder> enc = mm_encoder(ctx);
    uint M = (uint)A->rows, N = (uint)A->columns;
    [enc setComputePipelineState:pipeline];
    [enc setBuffer:A->metalBuffer offset:0 atIndex:0];
    [enc setBuffer:x->metalBuffer offset:0 atIndex:1];
    [enc setBuffer:y->metalBuffer offset:0 atIndex:2];
    [enc setBytes:&M length:sizeof(uint) atIndex:3];
    [enc setBytes:&N length:sizeof(uint) atIndex:4];
    [enc setBytes:&alpha length:sizeof(float) atIndex:5];
    NSUInteger w = pipeline.threadExecutionWidth;
    NSUInteger h = MIN(pipeline.maxTotalThreadsPerThreadgroup / w, (NSUInteger)8);
    if (h == 0) h = 1;
    [enc dispatchThreads:MTLSizeMake(N, M, 1) threadsPerThreadgroup:MTLSizeMake(w, h, 1)];
    mm_encoded_one();
}

void mmatrix_gemm(mmatrix out, mmatrix a, int a_t, mmatrix b, int b_t, float alpha, float beta) {
    if (!out || !a || !b) {
        printf("[ERROR] Invalid matrices for multiplication\n");
        return;
    }
    const int M = a_t ? a->columns : a->rows;
    const int N = a_t ? a->rows : a->columns;
    const int K = b_t ? b->rows : b->columns;
    if (N != (b_t ? b->columns : b->rows)) {
        printf("ERROR: Incompatible matrix sizes for multiplication\n");
        return;
    }
    if (out->rows != M || out->columns != K) {
        printf("[ERROR] destination matrix has the wrong dimensions\n");
        return;
    }

    /* Matrix times column vector, the shape every dense layer produces. */
    if (K == 1 && !b_t && beta == 0.0f && alpha == 1.0f) {
        mm_gemv(out, a, a_t, b, NULL, 0);
        return;
    }
    /* Outer product accumulated in place, i.e. the weight update. */
    if (N == 1 && beta == 1.0f) {
        mm_ger(out, a, b, alpha);
        return;
    }
    if (a_t || b_t || alpha != 1.0f || beta != 0.0f) {
        printf("[ERROR] unsupported gemm variant on the Metal backend\n");
        return;
    }

    MMMetalContext *ctx = [MMMetalContext shared];
    id<MTLComputePipelineState> pipeline = ctx.pipelines[PIPE_MATMUL];
    id<MTLComputeCommandEncoder> enc = mm_encoder(ctx);
    uint um = (uint)M, un = (uint)N, uk = (uint)K;
    [enc setComputePipelineState:pipeline];
    [enc setBuffer:a->metalBuffer offset:0 atIndex:0];
    [enc setBuffer:b->metalBuffer offset:0 atIndex:1];
    [enc setBuffer:out->metalBuffer offset:0 atIndex:2];
    [enc setBytes:&um length:sizeof(uint) atIndex:3];
    [enc setBytes:&un length:sizeof(uint) atIndex:4];
    [enc setBytes:&uk length:sizeof(uint) atIndex:5];
    NSUInteger w = pipeline.threadExecutionWidth;
    NSUInteger h = MIN(pipeline.maxTotalThreadsPerThreadgroup / w, (NSUInteger)8);
    if (h == 0) h = 1;
    [enc dispatchThreads:MTLSizeMake(K, M, 1) threadsPerThreadgroup:MTLSizeMake(w, h, 1)];
    mm_encoded_one();
}

void mmatrix_mult_into(mmatrix out, mmatrix a, mmatrix b) {
    mmatrix_gemm(out, a, 0, b, 0, 1.0f, 0.0f);
}

mmatrix mmatrix_mult(mmatrix a, mmatrix b) {
    if (!a || !b) {
        printf("[ERROR] Invalid matrices for multiplication\n");
        return NULL;
    }
    if (a->columns != b->rows) {
        printf("ERROR: Incompatible matrix sizes for multiplication\n");
        return NULL;
    }
    mmatrix c = mmatrix_constructor(a->rows, b->columns);
    if (!c) {
        printf("[ERROR] Failed to create result matrix\n");
        return NULL;
    }
    mmatrix_mult_into(c, a, b);
    return c;
}

void mmatrix_layer_forward(mmatrix out, mmatrix w, mmatrix x, mmatrix bias) {
    if (!out || !w || !x || !bias) {
        printf("[ERROR] Invalid matrices for layer forward\n");
        return;
    }
    if (w->columns != x->rows || x->columns != 1 ||
        out->rows != w->rows || out->columns != 1 ||
        bias->rows != w->rows || bias->columns != 1) {
        printf("[ERROR] Matrix size mismatch for layer forward\n");
        return;
    }
    mm_gemv(out, w, 0, x, bias, 3u /* bias | sigmoid */);
}

void mmatrix_transpose_into(mmatrix out, mmatrix m) {
    if (!out || !m || out->rows != m->columns || out->columns != m->rows) {
        printf("[ERROR] destination matrix has the wrong dimensions\n");
        return;
    }
    MMMetalContext *ctx = [MMMetalContext shared];
    id<MTLComputePipelineState> pipeline = ctx.pipelines[PIPE_TRANSPOSE];
    id<MTLComputeCommandEncoder> enc = mm_encoder(ctx);
    [enc setComputePipelineState:pipeline];
    [enc setBuffer:m->metalBuffer offset:0 atIndex:0];
    [enc setBuffer:out->metalBuffer offset:0 atIndex:1];
    uint rows = (uint)m->rows, cols = (uint)m->columns;
    [enc setBytes:&rows length:sizeof(uint) atIndex:2];
    [enc setBytes:&cols length:sizeof(uint) atIndex:3];
    NSUInteger w = pipeline.threadExecutionWidth;
    NSUInteger h = MIN(pipeline.maxTotalThreadsPerThreadgroup / w, (NSUInteger)8);
    if (h == 0) h = 1;
    [enc dispatchThreads:MTLSizeMake(cols, rows, 1) threadsPerThreadgroup:MTLSizeMake(w, h, 1)];
    mm_encoded_one();
}

mmatrix mmatrix_transpose(mmatrix m) {
    if (!m) {
        printf("[ERROR] Invalid matrix for transpose\n");
        return NULL;
    }
    mmatrix t = mmatrix_constructor(m->columns, m->rows);
    if (!t) {
        printf("[ERROR] Failed to create transpose matrix\n");
        return NULL;
    }
    mmatrix_transpose_into(t, m);
    return t;
}

void mmatrix_copy(mmatrix dest, mmatrix src) {
    if (!dest || !src || !dest->metalBuffer || !src->metalBuffer) {
        printf("[ERROR] Invalid matrices for copy\n");
        return;
    }
    if (dest->rows != src->rows || dest->columns != src->columns) {
        printf("[ERROR] Matrix size mismatch for copy\n");
        return;
    }
    /* Blit needs its own encoder, so close the compute one first. */
    MMMetalContext *ctx = [MMMetalContext shared];
    if (g_enc) {
        [g_enc endEncoding];
        g_enc = nil;
    }
    id<MTLCommandBuffer> cmd = mm_command_buffer(ctx);
    id<MTLBlitCommandEncoder> blit = [cmd blitCommandEncoder];
    [blit copyFromBuffer:src->metalBuffer sourceOffset:0
                toBuffer:dest->metalBuffer destinationOffset:0
                    size:sizeof(float) * dest->rows * dest->columns];
    [blit endEncoding];
    mm_encoded_one();
}

void mmatrix_randomize(mmatrix m, float min, float max) {
    if (!m || !m->metalBuffer) {
        printf("[ERROR] Invalid matrix for randomization\n");
        return;
    }
    float *tmp = malloc(sizeof(float) * m->rows * m->columns);
    if (!tmp) {
        printf("[ERROR] Failed to allocate temporary memory for randomization\n");
        return;
    }
    for (int i = 0; i < m->rows * m->columns; i++) {
        tmp[i] = ((float)rand() / (float)RAND_MAX) * (max - min) + min;
    }
    mmatrix_copy_from_host(m, tmp);
    free(tmp);
}

int mmatrix_get_rows(mmatrix m) {
    return m ? m->rows : 0;
}

int mmatrix_get_cols(mmatrix m) {
    return m ? m->columns : 0;
}

void mmatrix_print(mmatrix m) {
    if (!g_debug) return;
    if (!m || !m->metalBuffer) {
        printf("[ERROR] Invalid matrix for printing\n");
        return;
    }
    float *tmp = malloc(sizeof(float) * m->rows * m->columns);
    if (!tmp) {
        printf("[ERROR] Failed to allocate temporary memory for printing\n");
        return;
    }
    mmatrix_copy_to_host(m, tmp);
    for (int i = 0; i < m->rows * m->columns; i++) {
        printf("%f", tmp[i]);
        if ((i + 1) % m->columns == 0) {
            printf("\n");
        } else {
            printf(" ");
        }
    }
    free(tmp);
}

float mmatrix_get_random(float min, float max) {
    return ((float)rand() / (float)RAND_MAX) * (max - min) + min;
}
