#import "mmatrix.h"
#import "../src/akira.h"
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <string.h>
#include <stdint.h>

static NSString * const kMatrixShaders = @"\
#include <metal_stdlib>\n\
using namespace metal;\n\
kernel void matmul(const device float* A [[ buffer(0) ]],\n                   const device float* B [[ buffer(1) ]],\n                   device float* C [[ buffer(2) ]],\n                   constant int& M [[ buffer(3) ]],\n                   constant int& N [[ buffer(4) ]],\n                   constant int& K [[ buffer(5) ]],\n                   uint2 gid [[thread_position_in_grid]]) {\n    int row = gid.x;\n    int col = gid.y;\n    if (row >= M || col >= K) return;\n    float sum = 0.0f;\n    for (int i = 0; i < N; ++i) {\n        sum += A[row * N + i] * B[i * K + col];\n    }\n    C[row * K + col] = sum;\n}\n\
kernel void madd(device float* A, const device float* B, constant uint& size, uint gid [[thread_position_in_grid]]) {\n    if (gid >= size) return;\n    A[gid] += B[gid];\n}\n\
kernel void msub(device float* A, const device float* B, constant uint& size, uint gid [[thread_position_in_grid]]) {\n    if (gid >= size) return;\n    A[gid] -= B[gid];\n}\n\
kernel void mhadamard(device float* A, const device float* B, constant uint& size, uint gid [[thread_position_in_grid]]) {\n    if (gid >= size) return;\n    A[gid] *= B[gid];\n}\n\
kernel void mscalar_mult(device float* A, constant float& scalar, constant uint& size, uint gid [[thread_position_in_grid]]) {\n    if (gid >= size) return;\n    A[gid] *= scalar;\n}\n\
kernel void mapply(device float* A, constant uint& size, constant int& op, uint gid [[thread_position_in_grid]]) {\n    if (gid >= size) return;\n    float x = A[gid];\n    // Clamp x to avoid overflow\n    x = clamp(x, -30.0f, 30.0f);\n    if (op == 0) {\n        // sigmoid\n        A[gid] = 1.0f / (1.0f + exp(-x));\n    } else if (op == 1) {\n        // sigmoid_derivative\n        float s = 1.0f / (1.0f + exp(-x));\n        A[gid] = s * (1.0f - s);\n    }\n}\n\
kernel void mtranspose(const device float* A, device float* T, constant uint& rows, constant uint& cols, uint2 gid [[thread_position_in_grid]]) {\n    uint row = gid.x;\n    uint col = gid.y;\n    if (row >= rows || col >= cols) return;\n    T[col * rows + row] = A[row * cols + col];\n}\n";

@interface MMMetalContext : NSObject
@property (nonatomic, strong) id<MTLDevice> device;
@property (nonatomic, strong) id<MTLLibrary> library;
@property (nonatomic, strong) id<MTLCommandQueue> queue;
+ (instancetype)shared;
@end

@implementation MMMetalContext
+ (instancetype)shared {
    fflush(stdout);
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
        ctx.library = [ctx.device newLibraryWithSource:kMatrixShaders options:nil error:&err];
        if (!ctx.library) {
            NSLog(@"Metal library error: %@", err);
            return;
        }
        ctx.queue = [ctx.device newCommandQueue];
    });
    return ctx;
}
@end

struct _mmatrix {
    int rows;
    int columns;
    id<MTLBuffer> metalBuffer;
};

mmatrix mmatrix_constructor(int rows, int columns) {
    
    if (rows <= 0 || columns <= 0) {
        printf("[ERROR] Invalid matrix dimensions: rows=%d, cols=%d\n", rows, columns);
        return NULL;
    }
    
    MMMetalContext *ctx = [MMMetalContext shared];
    
    if (!ctx.device) {
        printf("[ERROR] Metal device not available\n");
        return NULL;
    }
    
    mmatrix m = malloc(sizeof(struct _mmatrix));
    
    if (!m) {
        printf("[ERROR] Failed to allocate memory for matrix\n");
        return NULL;
    }
    
    m->rows = rows;
    m->columns = columns;
    NSUInteger size = sizeof(float) * rows * columns;
    
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
        m->metalBuffer = nil;
        free(m);
    }
}

void mmatrix_copy_from_host(mmatrix m, float *host_data) {
    if (!m || !m->metalBuffer || !host_data) {
        printf("[ERROR] Invalid parameters for mmatrix_copy_from_host\n");
        return;
    }
    memcpy([m->metalBuffer contents], host_data, sizeof(float) * m->rows * m->columns);
}

void mmatrix_copy_to_host(mmatrix m, float *host_data) {
    if (!m || !m->metalBuffer || !host_data) {
        printf("[ERROR] Invalid parameters for mmatrix_copy_to_host\n");
        return;
    }
    memcpy(host_data, [m->metalBuffer contents], sizeof(float) * m->rows * m->columns);
}

#define CHECK_PIPELINE(pipeline, fn) \
    if (!(pipeline)) { NSLog(@"Pipeline error for %@: %@", fn.name, err); return; }

void mmatrix_add(mmatrix m, mmatrix a) {
    if (!m || !a || !m->metalBuffer || !a->metalBuffer) {
        printf("[ERROR] Invalid matrices for addition\n");
        return;
    }
    if (m->rows != a->rows || m->columns != a->columns) {
        printf("[ERROR] Matrix size mismatch for addition\n");
        return;
    }
    
    MMMetalContext *ctx = [MMMetalContext shared];
    id<MTLFunction> fn = [ctx.library newFunctionWithName:@"madd"];
    NSError *err = nil;
    id<MTLComputePipelineState> pipeline = [ctx.device newComputePipelineStateWithFunction:fn error:&err];
    CHECK_PIPELINE(pipeline, fn);
    
    id<MTLCommandBuffer> cmd = [ctx.queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setBuffer:m->metalBuffer offset:0 atIndex:0];
    [enc setBuffer:a->metalBuffer offset:0 atIndex:1];
    uint size = m->rows * m->columns;
    [enc setBytes:&size length:sizeof(uint) atIndex:2];
    [enc setComputePipelineState:pipeline];
    
    NSUInteger threadsPerThreadgroup = MIN(pipeline.maxTotalThreadsPerThreadgroup, 256);
    NSUInteger threadgroups = (size + threadsPerThreadgroup - 1) / threadsPerThreadgroup;
    
    [enc dispatchThreadgroups:MTLSizeMake(threadgroups, 1, 1) 
                threadsPerThreadgroup:MTLSizeMake(threadsPerThreadgroup, 1, 1)];
    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];
}

void mmatrix_sub(mmatrix m, mmatrix a) {
    if (!m || !a || !m->metalBuffer || !a->metalBuffer) {
        printf("[ERROR] Invalid matrices for subtraction\n");
        return;
    }
    if (m->rows != a->rows || m->columns != a->columns) {
        printf("[ERROR] Matrix size mismatch for subtraction\n");
        return;
    }
    
    MMMetalContext *ctx = [MMMetalContext shared];
    id<MTLFunction> fn = [ctx.library newFunctionWithName:@"msub"];
    NSError *err = nil;
    id<MTLComputePipelineState> pipeline = [ctx.device newComputePipelineStateWithFunction:fn error:&err];
    CHECK_PIPELINE(pipeline, fn);
    
    id<MTLCommandBuffer> cmd = [ctx.queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setBuffer:m->metalBuffer offset:0 atIndex:0];
    [enc setBuffer:a->metalBuffer offset:0 atIndex:1];
    uint size = m->rows * m->columns;
    [enc setBytes:&size length:sizeof(uint) atIndex:2];
    [enc setComputePipelineState:pipeline];
    
    NSUInteger threadsPerThreadgroup = MIN(pipeline.maxTotalThreadsPerThreadgroup, 256);
    NSUInteger threadgroups = (size + threadsPerThreadgroup - 1) / threadsPerThreadgroup;
    
    [enc dispatchThreadgroups:MTLSizeMake(threadgroups, 1, 1) 
                threadsPerThreadgroup:MTLSizeMake(threadsPerThreadgroup, 1, 1)];
    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];
}

void mmatrix_hadamard(mmatrix m, mmatrix a) {
    if (!m || !a || !m->metalBuffer || !a->metalBuffer) {
        printf("[ERROR] Invalid matrices for Hadamard product\n");
        return;
    }
    if (m->rows != a->rows || m->columns != a->columns) {
        printf("[ERROR] Matrix size mismatch for Hadamard product\n");
        return;
    }
    
    MMMetalContext *ctx = [MMMetalContext shared];
    id<MTLFunction> fn = [ctx.library newFunctionWithName:@"mhadamard"];
    NSError *err = nil;
    id<MTLComputePipelineState> pipeline = [ctx.device newComputePipelineStateWithFunction:fn error:&err];
    CHECK_PIPELINE(pipeline, fn);
    
    id<MTLCommandBuffer> cmd = [ctx.queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setBuffer:m->metalBuffer offset:0 atIndex:0];
    [enc setBuffer:a->metalBuffer offset:0 atIndex:1];
    uint size = m->rows * m->columns;
    [enc setBytes:&size length:sizeof(uint) atIndex:2];
    [enc setComputePipelineState:pipeline];
    
    NSUInteger threadsPerThreadgroup = MIN(pipeline.maxTotalThreadsPerThreadgroup, 256);
    NSUInteger threadgroups = (size + threadsPerThreadgroup - 1) / threadsPerThreadgroup;
    
    [enc dispatchThreadgroups:MTLSizeMake(threadgroups, 1, 1) 
                threadsPerThreadgroup:MTLSizeMake(threadsPerThreadgroup, 1, 1)];
    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];
}

void mmatrix_scalar_mult(mmatrix m, float number) {
    if (!m || !m->metalBuffer) {
        printf("[ERROR] Invalid matrix for scalar multiplication\n");
        return;
    }
    
    MMMetalContext *ctx = [MMMetalContext shared];
    id<MTLFunction> fn = [ctx.library newFunctionWithName:@"mscalar_mult"];
    NSError *err = nil;
    id<MTLComputePipelineState> pipeline = [ctx.device newComputePipelineStateWithFunction:fn error:&err];
    CHECK_PIPELINE(pipeline, fn);
    
    id<MTLCommandBuffer> cmd = [ctx.queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setBuffer:m->metalBuffer offset:0 atIndex:0];
    [enc setBytes:&number length:sizeof(float) atIndex:1];
    uint size = m->rows * m->columns;
    [enc setBytes:&size length:sizeof(uint) atIndex:2];
    [enc setComputePipelineState:pipeline];
    
    NSUInteger threadsPerThreadgroup = MIN(pipeline.maxTotalThreadsPerThreadgroup, 256);
    NSUInteger threadgroups = (size + threadsPerThreadgroup - 1) / threadsPerThreadgroup;
    
    [enc dispatchThreadgroups:MTLSizeMake(threadgroups, 1, 1) 
                threadsPerThreadgroup:MTLSizeMake(threadsPerThreadgroup, 1, 1)];
    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];
}

void mmatrix_apply(mmatrix m, int op) {
    if (!m || !m->metalBuffer) {
        printf("[ERROR] Invalid matrix for apply operation\n");
        return;
    }
    
    MMMetalContext *ctx = [MMMetalContext shared];
    id<MTLFunction> fn = [ctx.library newFunctionWithName:@"mapply"];
    NSError *err = nil;
    id<MTLComputePipelineState> pipeline = [ctx.device newComputePipelineStateWithFunction:fn error:&err];
    CHECK_PIPELINE(pipeline, fn);
    
    id<MTLCommandBuffer> cmd = [ctx.queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setBuffer:m->metalBuffer offset:0 atIndex:0];
    uint size = m->rows * m->columns;
    [enc setBytes:&size length:sizeof(uint) atIndex:1];
    [enc setBytes:&op length:sizeof(int) atIndex:2];
    [enc setComputePipelineState:pipeline];
    
    NSUInteger threadsPerThreadgroup = MIN(pipeline.maxTotalThreadsPerThreadgroup, 256);
    NSUInteger threadgroups = (size + threadsPerThreadgroup - 1) / threadsPerThreadgroup;
    
    [enc dispatchThreadgroups:MTLSizeMake(threadgroups, 1, 1) 
                threadsPerThreadgroup:MTLSizeMake(threadsPerThreadgroup, 1, 1)];
    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];
}

mmatrix mmatrix_mult(mmatrix a, mmatrix b) {
    if (!a || !b || !a->metalBuffer || !b->metalBuffer) {
        printf("[ERROR] Invalid matrices for multiplication\n");
        return NULL;
    }
    
    MMMetalContext *ctx = [MMMetalContext shared];
    if (a->columns != b->rows) {
        printf("ERROR: Incompatible matrix sizes for multiplication\n");
        return NULL;
    }
    
    int M = a->rows, N = a->columns, K = b->columns;

    if (M <= 0 || N <= 0 || K <= 0) {
        printf("[ERROR] Invalid matmul dimensions: M=%d, N=%d, K=%d\n", M, N, K);
        fflush(stdout);
        return NULL;
    }
    
    mmatrix c = mmatrix_constructor(M, K);
    if (!c) {
        printf("[ERROR] Failed to create result matrix\n");
        return NULL;
    }
    
    id<MTLBuffer> A = a->metalBuffer;
    id<MTLBuffer> B = b->metalBuffer;
    id<MTLBuffer> C = c->metalBuffer;
    
    id<MTLFunction> fn = [ctx.library newFunctionWithName:@"matmul"];
    if (!fn) { 
        printf("[ERROR] Metal function matmul is NULL!\n"); 
        mmatrix_delete(c);
        return NULL; 
    }
    
    NSError *err = nil;
    id<MTLComputePipelineState> pipeline = [ctx.device newComputePipelineStateWithFunction:fn error:&err];
    if (!pipeline) { 
        printf("[ERROR] Metal pipeline for matmul is NULL! Error: %s\n", [[err localizedDescription] UTF8String]); 
        mmatrix_delete(c);
        return NULL; 
    }
    
    id<MTLCommandBuffer> cmd = [ctx.queue commandBuffer];
    
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    
    [enc setBuffer:A offset:0 atIndex:0];
    [enc setBuffer:B offset:0 atIndex:1];
    [enc setBuffer:C offset:0 atIndex:2];
    
    [enc setBytes:&M length:sizeof(int) atIndex:3];
    [enc setBytes:&N length:sizeof(int) atIndex:4];
    [enc setBytes:&K length:sizeof(int) atIndex:5];
    
    [enc setComputePipelineState:pipeline];
    
    NSUInteger maxThreadsPerGroup = pipeline.maxTotalThreadsPerThreadgroup;
    NSUInteger threadsPerGroup = MIN(maxThreadsPerGroup, 256);
    
    MTLSize threadGroupSize = MTLSizeMake(16, 16, 1);  // 16x16 = 256 threads per group
    if (threadsPerGroup < 256) {
        NSUInteger side = (NSUInteger)sqrt(threadsPerGroup);
        threadGroupSize = MTLSizeMake(side, side, 1);
    }
    
    MTLSize gridSize = MTLSizeMake(M, K, 1);
    
    [enc dispatchThreads:gridSize threadsPerThreadgroup:threadGroupSize];
    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];
    return c;
}

mmatrix mmatrix_transpose(mmatrix m) {
    if (!m || !m->metalBuffer) {
        printf("[ERROR] Invalid matrix for transpose\n");
        return NULL;
    }
    
    MMMetalContext *ctx = [MMMetalContext shared];
    mmatrix t = mmatrix_constructor(m->columns, m->rows);
    if (!t) {
        printf("[ERROR] Failed to create transpose matrix\n");
        return NULL;
    }
    
    id<MTLFunction> fn = [ctx.library newFunctionWithName:@"mtranspose"];
    NSError *err = nil;
    id<MTLComputePipelineState> pipeline = [ctx.device newComputePipelineStateWithFunction:fn error:&err];
    if (!pipeline) {
        printf("[ERROR] Failed to create transpose pipeline\n");
        mmatrix_delete(t);
        return NULL;
    }
    
    id<MTLCommandBuffer> cmd = [ctx.queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setBuffer:m->metalBuffer offset:0 atIndex:0];
    [enc setBuffer:t->metalBuffer offset:0 atIndex:1];
    uint rows = m->rows;
    uint cols = m->columns;
    [enc setBytes:&rows length:sizeof(uint) atIndex:2];
    [enc setBytes:&cols length:sizeof(uint) atIndex:3];
    [enc setComputePipelineState:pipeline];
    
    NSUInteger maxThreadsPerGroup = pipeline.maxTotalThreadsPerThreadgroup;
    NSUInteger threadsPerGroup = MIN(maxThreadsPerGroup, 256);
    NSUInteger side = (NSUInteger)sqrt(threadsPerGroup);
    
    MTLSize threadGroupSize = MTLSizeMake(side, side, 1);
    MTLSize gridSize = MTLSizeMake(rows, cols, 1);
    
    [enc dispatchThreads:gridSize threadsPerThreadgroup:threadGroupSize];
    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];
    return t;
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
        tmp[i] = ((float)rand()/RAND_MAX)*(max-min)+min;
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

void mmatrix_copy(mmatrix dest, mmatrix src) {
    if (!dest || !src || !dest->metalBuffer || !src->metalBuffer) {
        printf("[ERROR] Invalid matrices for copy\n");
        return;
    }
    if (dest->rows != src->rows || dest->columns != src->columns) {
        printf("[ERROR] Matrix size mismatch for copy\n");
        return;
    }

    MMMetalContext *ctx = [MMMetalContext shared];
    id<MTLCommandBuffer> cmd = [ctx.queue commandBuffer];
    id<MTLBlitCommandEncoder> enc = [cmd blitCommandEncoder];
    [enc copyFromBuffer:src->metalBuffer sourceOffset:0 toBuffer:dest->metalBuffer destinationOffset:0 size:sizeof(float) * dest->rows * dest->columns];
    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];
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
    return ((float)rand()/RAND_MAX)*(max-min)+min; 
}
