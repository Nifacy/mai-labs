#ifndef _UTILS_H_
#define _UTILS_H_

#define EXIT_WITH_ERROR(message)                                   \
{                                                                  \
    fprintf(stderr, "ERROR: [line %d] %s\n", __LINE__, (message)); \
    exit(0);                                                       \
}                                                                  \


#define SAVE_CUDA(call)                              \
{                                                    \
    cudaError_t result = call;                       \
    if (result != cudaSuccess) {                     \
        EXIT_WITH_ERROR(cudaGetErrorString(result)); \
    }                                                \
}

/* Common types */

typedef enum {
    CPU,
    GPU
} DeviceType;

/* Help Methods */

template<typename T>
__host__ __device__ T Max(T a, T b) {
    if (a > b) return a;
    return b;
}

template<typename T>
__host__ __device__ T Min(T a, T b) {
    if (a < b) return a;
    return b;
}

#endif // _UTILS_H_
