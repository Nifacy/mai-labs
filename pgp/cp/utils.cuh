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

#endif // _UTILS_H_
