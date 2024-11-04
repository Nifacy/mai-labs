#include <iostream>
#include <iomanip>
#include <cstdint>

#include <thrust/extrema.h>
#include <thrust/device_vector.h>

const size_t BLOCKS = 1024;
const size_t THREADS = 1024;

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

__global__ void updateRow(double* a, double* b, int32_t n, int32_t i, int32_t j) {
    double k = - a[i * n + j] / a[i * n + i];
    size_t offset = gridDim.x * blockDim.x;
    size_t index = blockDim.x * blockIdx.x + threadIdx.x;

    while (index < n) {
        a[index * n + j] += k * a[index * n + i];
        b[index * n + j] += k * b[index * n + i];
        index += offset;
    }
}

__global__ void mult(double* a, double* b, int32_t n, int32_t i) {
    size_t offset = gridDim.x * blockDim.x;
    size_t index = blockDim.x * blockIdx.x + threadIdx.x;
    double k = a[i * n + i];

    while (index < n) {
        b[index * n + i] /= k;
        index += offset;
    }
}

double* createEyeMatrix(int32_t n) {
    double* m = new double[n * n];

    for (uint32_t i = 0; i < n; ++i) {
        for (uint32_t j = 0; j < n; ++j) {
            m[i * n + j] = double(i == j);
        }
    }

    return m;
}

struct Comparator {
    __host__ __device__ bool operator()(double a, double b) {
        return std::fabs(a) < std::fabs(b);
    }
};

int main() {
    Comparator comp;
 
    int32_t n;
    double *a, *b;
    int32_t* p;

    double *deviceA, *deviceB;
    
    std::cin >> n;

    a = new double[n * n];
    for (int32_t i = 0; i < n; ++i) {
        for (int32_t j = 0; j < n; ++j) {
            std::cin >> a[j * n + i];
        }
    }

    p = new int32_t[n];
    for (int32_t i = 0; i < n; ++i) {
        p[i] = i;
    }

    b = createEyeMatrix(n);

    SAVE_CUDA(cudaMalloc(&deviceA, sizeof(double) * n * n));
    SAVE_CUDA(cudaMemcpy(deviceA, a, sizeof(double) * n * n, cudaMemcpyHostToDevice));

    SAVE_CUDA(cudaMalloc(&deviceB, sizeof(double) * n * n));
    SAVE_CUDA(cudaMemcpy(deviceB, b, sizeof(double) * n * n, cudaMemcpyHostToDevice));

    // Forward
    for (int32_t i = 0; i < n; ++i) {
        int32_t maxIndex;
        thrust::device_ptr<double> leftPtr;
        thrust::device_ptr<double> maxPtr;

        leftPtr = thrust::device_pointer_cast(deviceA + i * n);
        maxPtr = thrust::max_element(leftPtr + i, leftPtr + n, comp);
        maxIndex = maxPtr - leftPtr;
        
        std::swap(p[i], p[maxIndex]);
        swapRows<<<BLOCKS, THREADS>>>(deviceA, n, i, maxIndex);
        swapRows<<<BLOCKS, THREADS>>>(deviceB, n, i, maxIndex);

        for (int32_t j = i + 1; j < n; ++j) {
            updateRow<<<BLOCKS, THREADS>>>(deviceA, deviceB, n, i, j);
        }
    }

    // Backward
    for (int32_t i = n - 1; i >= 0; --i) {
        for (int32_t j = i - 1; j >= 0; --j) {
            updateRow<<<BLOCKS, THREADS>>>(deviceA, deviceB, n, i, j);
        }
    }

    // Normalize
    for (int32_t i = 0; i < n; ++i) {
        mult<<<BLOCKS, THREADS>>>(deviceA, deviceB, n, i);
    }

    // Print results
    SAVE_CUDA(cudaMemcpy(b, deviceB, sizeof(double) * n * n, cudaMemcpyDeviceToHost));

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            std::cout << std::scientific << std::setprecision(10) << b[j * n + i] << " ";
        }
        std::cout << std::endl;
    }

    SAVE_CUDA(cudaFree(deviceA));
    SAVE_CUDA(cudaFree(deviceB));

    delete[] b;
    delete[] p;
    delete[] a;

    return 0;
}
