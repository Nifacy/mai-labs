#ifndef _ARRAY_H_
#define _ARRAY_H_

#include "utils.cuh"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum DeviceType {
    CPU,
    GPU
};

template <typename T>
struct TArray {
    T* data;
    int* size;
    int capacity;
    DeviceType device;
};

template <typename T>
__host__ void Init(TArray<T> *array, int capacity, DeviceType device) {
    array->capacity = capacity;
    array->device = device;

    if (device == GPU) {
        SAVE_CUDA(cudaMalloc(&array->data, capacity * sizeof(T)));
        SAVE_CUDA(cudaMalloc(&array->size, sizeof(int)));
        SAVE_CUDA(cudaMemset(array->size, 0, sizeof(int)));
    } else {
        array->data = (T*) std::malloc(capacity * sizeof(T));
        array->size = (int*) std::malloc(sizeof(int));
        memset(array->size, 0, sizeof(int));
    }
}

template <typename T>
__host__ void Destroy(TArray<T> *array) {
    if (array->device == GPU) {
        cudaFree(array->data);
        cudaFree(array->size);
    } else {
        free(array->data);
        free(array->size);
    }
}

template <typename T>
__host__ __device__ void Append(TArray<T> *array, T value) {
    if (array->deviceType == GPU) {
        int index = atomicAdd(array->size, 1);
        if (index < array->capacity) {
            array->data[index] = value;
        } else {
            atomicSub(array->size, 1);
        }
    } else {
        int index = (*array.size)++;
        if (index < array.capacity) {
            array.data[index] = value;
        } else {
            (*array.size)--;
        }
    }
}

template <typename T>
__host__ __device__ int Size(TArray<T> *array) {
    return *array.size;
}

#endif
