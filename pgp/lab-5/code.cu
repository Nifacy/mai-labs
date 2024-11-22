#include <chrono>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

using TElement = uint32_t;

const size_t BLOCKS = 1024;
const size_t THREADS = 1024;
const size_t HIST_SIZE = 16777216;

typedef struct _TArray {
    uint32_t* data;
    uint32_t size;
} TArray;

#define EXIT_WITH_ERROR(message)                                       \
    {                                                                  \
        fprintf(stderr, "ERROR: [line %d] %s\n", __LINE__, (message)); \
        exit(0);                                                       \
    }

#define SAVE_CUDA(call)                                  \
    {                                                    \
        cudaError_t result = call;                       \
        if (result != cudaSuccess) {                     \
            EXIT_WITH_ERROR(cudaGetErrorString(result)); \
        }                                                \
    }

void checkRunResult() {
    cudaDeviceSynchronize();
    SAVE_CUDA(cudaGetLastError());
}

/* build histogram */

__global__ void hist(TElement* arr, uint32_t* h, uint32_t n, uint32_t m) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t offsetx = gridDim.x * blockDim.x;

    for (uint32_t i = idx; i < n; i += offsetx) {
        atomicAdd(&h[arr[i]], 1);
    }

#ifdef DEBUG_LONG
    for (uint32_t i = idx; i < m; ++i) {
        if (h[i] != 0) {
            printf("hist : h[%u] = %u\n", i, h[i]);
        }
    }
#endif
}

/* scan algorithm */

__global__ void upPhase(uint32_t* h, uint32_t n) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t offsetx = gridDim.x * blockDim.x;
    uint32_t d, indexA, indexB;

    for (uint32_t m = 2; m <= n; m <<= 1) {
        d = n / m;
        for (uint32_t i = idx; i < d; i += offsetx) {
            indexA = i * m + m - 1;
            indexB = i * m + m / 2 - 1;

#ifdef DEBUG
            if (indexA >= n) {
                printf("upPhase : error : indexA = %u >= %u\n", indexA, n);
            }

            if (indexB >= n) {
                printf("upPhase : error : indexB = %u >= %u\n", indexB, n);
            }
#endif

            h[indexA] += h[indexB];
        }
        __syncthreads();
    }
}

__global__ void downPhase(uint32_t* h, uint32_t n) {
    __shared__ uint32_t lastElement;

    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t offsetx = gridDim.x * blockDim.x;
    uint32_t d, tmp, indexA, indexB;

    if (idx == 0) {
        lastElement = h[n - 1];
        h[n - 1] = 0;
    }
    __syncthreads();

#ifdef DEBUG
    if (idx == 0) {
        printf("[log] downPhase : lastElement set\n");
    }
#endif

    for (uint32_t m = n; m > 1; m >>= 1) {
        d = n / m;
        for (uint32_t i = idx; i < d; i += offsetx) {
            indexA = i * m + m - 1;
            indexB = i * m + m / 2 - 1;

#ifdef DEBUG
            if (indexA >= n) {
                printf("upPhase : error : indexA = %u >= %u\n", indexA, n);
            }

            if (indexB >= n) {
                printf("upPhase : error : indexB = %u >= %u\n", indexB, n);
            }
#endif

            tmp = h[indexA];
            h[indexA] += h[indexB];
            h[indexB] = tmp;
        }
        __syncthreads();
    }

    if (idx == 0) {
        h[0] = lastElement;
    }
}

void blellochScan(uint32_t* h, uint32_t n) {
    // use only 1 block cause blelloch scan algorithm
    // requires sync between blocks in that way

#ifdef TIME
    auto start1 = std::chrono::high_resolution_clock::now();
#endif

    upPhase<<<1, THREADS>>>(h, n);
    checkRunResult();

#ifdef TIME
    auto end1 = std::chrono::high_resolution_clock::now();
#endif

#ifdef TIME
    auto start2 = std::chrono::high_resolution_clock::now();
#endif

    downPhase<<<1, THREADS>>>(h, n);
    checkRunResult();

#ifdef TIME
    auto end2 = std::chrono::high_resolution_clock::now();
#endif

#ifdef TIME
    auto duration1 =
        std::chrono::duration_cast<std::chrono::milliseconds>(end1 - start1);
    auto duration2 =
        std::chrono::duration_cast<std::chrono::milliseconds>(end2 - start2);

    std::cerr << "[log] blellochScan.upPhase : " << duration1.count()
              << std::endl;
    std::cerr << "[log] blellochScan.downPhase : " << duration2.count()
              << std::endl;
#endif
}

__global__ void buildResult(TElement* src, uint32_t* h, TElement* dst,
                            uint32_t n, uint32_t m) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t offsetx = gridDim.x * blockDim.x;
    uint32_t pos, insertIndex;
    TElement element;

#ifdef DEBUG
    if (idx == 0) {
        printf("buildResult : h: [");
        for (uint32_t i = 0; i < 5; ++i) {
            printf("(%u, %u), ", i, h[i]);
        }

        printf("..., ");

        for (uint32_t i = m - 20; i < m; ++i) {
            printf("(%u, %u), ", i, h[i]);
        }

        printf("]\n");
    }
#endif

    for (uint32_t i = idx; i < n; i += offsetx) {
        element = src[i];
        insertIndex = (element + 1 == HIST_SIZE) ? 0 : element + 1;

#ifdef DEBUG
        if (insertIndex >= m) {
            printf("buildResult : error : insertIndex = %u >= %u\n",
                   insertIndex, m);
        }
#endif

        pos = atomicSub(&h[insertIndex], 1) - 1;

#ifdef DEBUG
        if (pos >= n) {
            printf("buildResult : error : pos = %u >= %u\n", pos, n);
            printf("element: %u\n", element);
            printf("insertIndex: %u\n", insertIndex);
        }
#endif

        dst[pos] = element;
    }
}

void countSort(TElement* arr, uint32_t* h, TElement* dst, uint32_t n,
               uint32_t m) {
    // create histogram

#ifdef TIME
    auto start1 = std::chrono::high_resolution_clock::now();
#endif

    hist<<<BLOCKS, THREADS>>>(arr, h, n, m);
    checkRunResult();

#ifdef TIME
    auto end1 = std::chrono::high_resolution_clock::now();
#endif

// build prefix sum of histogram
#ifdef TIME
    auto start2 = std::chrono::high_resolution_clock::now();
#endif

    blellochScan(h, m);

#ifdef TIME
    auto end2 = std::chrono::high_resolution_clock::now();
#endif

// build result array on prefix sum
#ifdef TIME
    auto start3 = std::chrono::high_resolution_clock::now();
#endif

    buildResult<<<BLOCKS, THREADS>>>(arr, h, dst, n, m);
    checkRunResult();

#ifdef TIME
    auto end3 = std::chrono::high_resolution_clock::now();
#endif

#ifdef TIME
    auto duration1 =
        std::chrono::duration_cast<std::chrono::milliseconds>(end1 - start1);
    auto duration2 =
        std::chrono::duration_cast<std::chrono::milliseconds>(end2 - start2);
    auto duration3 =
        std::chrono::duration_cast<std::chrono::milliseconds>(end3 - start3);

    std::cerr << "[log] countSort.hist : " << duration1.count() << std::endl;
    std::cerr << "[log] countSort.scan : " << duration2.count() << std::endl;
    std::cerr << "[log] countSort.build : " << duration3.count() << std::endl;
#endif
}

TArray readArray() {
    TArray arr = {.data = nullptr, .size = 0};

    if (fread(&(arr.size), sizeof(uint32_t), 1, stdin) != 1) {
        EXIT_WITH_ERROR("[ERROR] readArray : unable to read array size");
    }

    arr.data = new TElement[arr.size];
    if (!arr.data) {
        EXIT_WITH_ERROR("[ERROR] readArray : memory allocation failed");
    }

    if (fread(arr.data, sizeof(TElement), arr.size, stdin) != arr.size) {
        EXIT_WITH_ERROR("[ERROR] readArray : unable to read array data");
    }

    return arr;
}

void writeArray(TArray arr) {
    fwrite(arr.data, sizeof(uint32_t), arr.size, stdout);
}

TArray initDeviceArray(uint32_t size) {
    TArray arr = {.data = nullptr, .size = size};
    SAVE_CUDA(cudaMalloc(&arr.data, arr.size * sizeof(uint32_t)));
    return arr;
}

TArray initDeviceArray(uint32_t size, uint32_t* src) {
    TArray arr = initDeviceArray(size);
    SAVE_CUDA(cudaMemcpy(arr.data, src, sizeof(uint32_t) * size,
                         cudaMemcpyHostToDevice));
    return arr;
}

TArray initHist(uint32_t m) {
    TArray arr = initDeviceArray(m);
    cudaMemset(arr.data, 0, arr.size * sizeof(uint32_t));
    return arr;
}

int main() {
#ifdef TIME
    auto start1 = std::chrono::high_resolution_clock::now();
#endif

    TArray hostSrc = readArray();
    TArray hostResult = {.data = new TElement[hostSrc.size],
                         .size = hostSrc.size};

#ifdef DEBUG
    std::cerr << "[log] hostSrc : [";

    for (uint32_t i = 0; i < hostSrc.size; ++i) {
        std::cerr << hostSrc.data[i] << ", ";
    }

    std::cerr << "]" << std::endl;
#endif

    // init device memory
    TArray h = initHist(HIST_SIZE);
    TArray arr = initDeviceArray(hostSrc.size, hostSrc.data);
    TArray dst = initDeviceArray(hostSrc.size);

#ifdef TIME
    auto start2 = std::chrono::high_resolution_clock::now();
#endif

    // main part
    countSort(arr.data, h.data, dst.data, arr.size, h.size);

#ifdef TIME
    auto end2 = std::chrono::high_resolution_clock::now();
#endif

    cudaMemcpy(hostResult.data, dst.data, sizeof(TElement) * hostResult.size,
               cudaMemcpyDeviceToHost);
    writeArray(hostResult);

#ifdef TIME
    auto end1 = std::chrono::high_resolution_clock::now();
#endif

#ifdef TIME
    auto duration1 =
        std::chrono::duration_cast<std::chrono::milliseconds>(end1 - start1);
    auto duration2 =
        std::chrono::duration_cast<std::chrono::milliseconds>(end2 - start2);

    std::cerr << "[log][time] total: " << duration1.count() << std::endl;
    std::cerr << "[log][time] kernel: " << duration2.count() << std::endl;
#endif

    SAVE_CUDA(cudaFree(arr.data));
    SAVE_CUDA(cudaFree(dst.data));
    SAVE_CUDA(cudaFree(h.data));

    delete[] hostResult.data;
    delete[] hostSrc.data;

    return 0;
}
