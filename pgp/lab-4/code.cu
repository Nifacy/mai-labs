#include <iostream>
#include <iomanip>
#include <cstdint>

#include <thrust/extrema.h>
#include <thrust/device_vector.h>

const size_t BLOCKS = 1024;
const size_t THREADS = 1024;

const size_t BLOCKS_2[2] = {32, 32};
const size_t THREADS_2[2] = {32, 32};

#define EXIT_WITH_ERROR(message)                                   \
{                                                                  \
    fprintf(stderr, "ERROR: [line %d] %s\n", __LINE__, (message)); \
    exit(0);                                                       \
}

#define SAVE_CUDA(call)                                            \
{                                                                  \
    cudaError_t result = call;                                     \
    if (result != cudaSuccess) {                                   \
        EXIT_WITH_ERROR(cudaGetErrorString(result));               \
    }                                                              \
}                                                                  \

__global__ void swapRows(double* m, int32_t n, int32_t i, int32_t j) {
    size_t offset = gridDim.x * blockDim.x;
    size_t index = blockDim.x * blockIdx.x + threadIdx.x;
    double tmp;

    while (index < n) {
        tmp = m[index * n + i];
        m[index * n + i] = m[index * n + j];
        m[index * n + j] = tmp;
        index += offset;
    }
}

__global__ void initCoefs(double* a, double* coefs, int32_t n, int32_t i) {
    size_t offset = gridDim.x * blockDim.x;
    size_t start = blockDim.x * blockIdx.x + threadIdx.x;

    for (int32_t j = start + i + 1; j < n; j += offset) {
        coefs[j] = - a[i * n + j] / a[i * n + i];
    }
}

__global__ void updateRows(double* a, double* b, double* coefs, int32_t n, int32_t i) {
    int32_t idx = blockDim.x * blockIdx.x + threadIdx.x;
    int32_t offsetX = gridDim.x * blockDim.x;

    int32_t idy = blockDim.y * blockIdx.y + threadIdx.y;
    int32_t offsetY = gridDim.y * blockDim.y;

    double k;

    for (int32_t t = idy; t < n; t += offsetY) {
        for (int32_t j = i + 1 + idx; j < n; j += offsetX) {
            k = coefs[j];
            a[t * n + j] += k * a[t * n + i];
            b[t * n + j] += k * b[t * n + i];
        }
    }
}

__global__ void initCoefsReversed(double* a, double* coefs, int32_t n, int32_t i) {
    size_t offset = gridDim.x * blockDim.x;
    size_t start = blockDim.x * blockIdx.x + threadIdx.x;
    double a_ii = a[i * n + i];

    for (int32_t j = i - 1 - start; j >= 0; j -= offset) {
        coefs[j] = - a[i * n + j] / a_ii;
    }
}

__global__ void updateRowsReversed(double* a, double* b, double* coefs, int32_t n, int32_t i) {
    int32_t idx = blockDim.x * blockIdx.x + threadIdx.x;
    int32_t offsetX = gridDim.x * blockDim.x;

    int32_t idy = blockDim.y * blockIdx.y + threadIdx.y;
    int32_t offsetY = gridDim.y * blockDim.y;

    for (int32_t t = idy; t < n; t += offsetY) {
        for (int32_t j = i - 1 - idx; j >= 0; j -= offsetX) {
            b[t * n + j] += coefs[j] * b[t * n + i];
        }
    }
}

__global__ void mult(double* a, double* b, int32_t n) {
    int32_t idx = blockDim.x * blockIdx.x + threadIdx.x;
    int32_t offsetX = gridDim.x * blockDim.x;

    size_t offsetY = gridDim.y * blockDim.y;
    size_t idy = blockDim.y * blockIdx.y + threadIdx.y;

    for (int32_t j = idx; j < n; j += offsetX) {
        for (int32_t i = idy; i < n; i += offsetY) {
            b[j * n + i] /= a[i * n + i];
        }
    }
}

__global__ void initEyeMatrix(double* m, int32_t n) {
    int32_t idx = blockDim.x * blockIdx.x + threadIdx.x;
    int32_t offsetX = gridDim.x * blockDim.x;

    size_t offsetY = gridDim.y * blockDim.y;
    size_t idy = blockDim.y * blockIdx.y + threadIdx.y;

    for (uint32_t j = idx; j < n; j += offsetX) {
        for (uint32_t i = idy; i < n; i += offsetY) {
            m[j * n + i] = double(i == j);
        }
    }
}

struct Comparator {
    __host__ __device__ bool operator()(double a, double b) {
        return std::fabs(a) < std::fabs(b);
    }
};

int main() {
    std::ios::sync_with_stdio(false);
    std::cout.tie(nullptr);

    Comparator comp;
 
    int32_t n;
    double *a;

    double *deviceA, *deviceB;
    double *coefs;
    
    std::cin >> n;

    a = new double[n * n];
    for (int32_t i = 0; i < n; ++i) {
        for (int32_t j = 0; j < n; ++j) {
            std::cin >> a[j * n + i];
        }
    }

    SAVE_CUDA(cudaMalloc(&deviceA, sizeof(double) * n * n));
    SAVE_CUDA(cudaMemcpy(deviceA, a, sizeof(double) * n * n, cudaMemcpyHostToDevice));

    SAVE_CUDA(cudaMalloc(&deviceB, sizeof(double) * n * n));
    SAVE_CUDA(cudaMalloc(&coefs, sizeof(double) * n));

    initEyeMatrix<<<
        dim3(BLOCKS_2[0], BLOCKS_2[1]),
        dim3(THREADS_2[0], THREADS_2[1])
    >>>(deviceB, n);
    cudaDeviceSynchronize();
    SAVE_CUDA(cudaGetLastError());

    // Forward
    for (int32_t i = 0; i < n; ++i) {
        int32_t maxIndex;
        thrust::device_ptr<double> leftPtr;
        thrust::device_ptr<double> maxPtr;

        leftPtr = thrust::device_pointer_cast(deviceA + i * n);
        maxPtr = thrust::max_element(leftPtr + i, leftPtr + n, comp);
        maxIndex = maxPtr - leftPtr;

        swapRows<<<BLOCKS, THREADS>>>(deviceA, n, i, maxIndex);
        swapRows<<<BLOCKS, THREADS>>>(deviceB, n, i, maxIndex);
        initCoefs<<<BLOCKS, THREADS>>>(deviceA, coefs, n, i);

        updateRows<<<
            dim3(BLOCKS_2[0], BLOCKS_2[1]),
            dim3(THREADS_2[0], THREADS_2[1])
        >>>(deviceA, deviceB, coefs, n, i);
        cudaDeviceSynchronize();
        SAVE_CUDA(cudaGetLastError());
    }

    // Backward
    for (int32_t i = n - 1; i >= 0; --i) {
        initCoefsReversed<<<BLOCKS, THREADS>>>(deviceA, coefs, n, i);
        updateRowsReversed<<<
            dim3(BLOCKS_2[0], BLOCKS_2[1]),
            dim3(THREADS_2[0], THREADS_2[1])
        >>>(deviceA, deviceB, coefs, n, i);
        cudaDeviceSynchronize();
        SAVE_CUDA(cudaGetLastError());
    }

    // Normalize
    mult<<<
        dim3(BLOCKS_2[0], BLOCKS_2[1]),
        dim3(THREADS_2[0], THREADS_2[1])
    >>>(deviceA, deviceB, n);
    cudaDeviceSynchronize();
    SAVE_CUDA(cudaGetLastError());

    // Print results
    SAVE_CUDA(cudaMemcpy(a, deviceB, sizeof(double) * n * n, cudaMemcpyDeviceToHost));

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            fprintf(stdout, "%.10e ", a[j * n + i]);
        }
        fprintf(stdout, "\n");
    }

    SAVE_CUDA(cudaFree(deviceA));
    SAVE_CUDA(cudaFree(deviceB));
    SAVE_CUDA(cudaFree(coefs));

    delete[] a;

    return 0;
}
