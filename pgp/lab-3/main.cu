#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include <chrono>
#include <iostream>
#include <string>


const uint32_t DEVICE_BLOCKS = 512;
const uint32_t DEVICE_THREADS = 512;


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
}                                                    \


/* Vector module */


typedef double Vector[3];


void vectorAdd(Vector v, Vector w) {
    for (size_t i = 0; i < 3; ++i) {
        v[i] += w[i];
    }
}


void pixelToVector(uchar4 pixel, Vector v) {
    v[0] = double(pixel.x);
    v[1] = double(pixel.y);
    v[2] = double(pixel.z);
}


void vectorMult(Vector v, double coef) {
    for (size_t i = 0; i < 3; ++i) {
        v[i] *= coef;
    }
}


double vectorLength(Vector v) {
    double s = 0;
    for (size_t i = 0; i < 3; ++i) {
        s += v[i] * v[i];
    }
    return sqrt(s);
}


/* Image module */


typedef struct _Image {
    int32_t width;
    int32_t height;
    uchar4* pixels;
} Image;


Image loadImage(const char* filepath) {
    Image img;
    FILE* fp = fopen(filepath, "rb");

    fread(&img.width, sizeof(int32_t), 1, fp);
    fread(&img.height, sizeof(int32_t), 1, fp);

    img.pixels = (uchar4*)malloc(sizeof(uchar4) * img.width * img.height);
    fread(img.pixels, sizeof(uchar4), img.width * img.height, fp);

    fclose(fp);
    return img;
}


void saveImage(const Image* img, const char* filepath) {
    FILE* fp = fopen(filepath, "wb");

    fwrite(&img->width, sizeof(int32_t), 1, fp);
    fwrite(&img->height, sizeof(int32_t), 1, fp);
    fwrite(img->pixels, sizeof(uchar4), img->width * img->height, fp);

    fclose(fp);
}


/* Device help functions */


__device__ void devicePixelToVector(uchar4 pixel, Vector v) {
    v[0] = double(pixel.x);
    v[1] = double(pixel.y);
    v[2] = double(pixel.z);
}


__device__ void deviceVectorDot(Vector v, Vector w, double* dot) {
    *dot = 0.0;
    for (size_t i = 0; i < 3; ++i) {
        *dot += v[i] * w[i];
    }
}


/* Kernel */


__constant__ Vector constVectors[255];
__constant__ int actualSize;


__global__ void kernel(uchar4* pixels, int w, int h) {
    size_t offset = gridDim.x * blockDim.x;
    size_t index = blockDim.x * blockIdx.x + threadIdx.x;
    uchar4 pixel;
    int maxIndex = -1;
    double maxDot = 0.0;
    Vector p;
    double dot;

    while (index < w * h) {
        pixel = pixels[index];
        devicePixelToVector(pixel, p);

        maxIndex = -1;
        maxDot = 0.0;

        for (int i = 0; i < actualSize; ++i) {
            deviceVectorDot(p, constVectors[i], &dot);
            if (maxIndex == -1 || maxDot < dot) {
                maxIndex = i;
                maxDot = dot;
            }
        }

        pixels[index].w = maxIndex;
        index += offset;
    }
}


/* Main */


void getAverageClassVector(Image* img, Vector v) {
    size_t np;
    size_t i, j;
    Vector w;
    uchar4 pixel;

    for (size_t t = 0; t < 3; t++) {
        v[t] = 0.0;
    }

    std::cin >> np;
    fprintf(stderr, "%lu, ", np);
    for (size_t t = 0; t < np; ++t) {
        std::cin >> i >> j;
        pixel = img->pixels[i * img->width + j];
        pixelToVector(pixel, w);
        vectorAdd(v, w);
    }

    vectorMult(v, 1.0 / (double(np)));
    vectorMult(v, 1.0 / vectorLength(v));
}


int main() {
    std::string inputPath, outputPath;
    int n;
    Image img;
    Vector* classVectors;
    uchar4* deviceOutput;

    std::getline(std::cin, inputPath);
    std::getline(std::cin, outputPath);
    img = loadImage(inputPath.c_str());

    std::cin >> n;
    fprintf(stderr, "n: %d\n", n);
    classVectors = new Vector[n];
    for (size_t i = 0; i < n; ++i) {
        getAverageClassVector(&img, classVectors[i]);
        fprintf(stderr, "\n");
    }

    // Prepair device input data

    SAVE_CUDA(
        cudaMalloc(
            &deviceOutput,
            sizeof(uchar4) * img.width * img.height
        )
    );

    SAVE_CUDA(
        cudaMemcpy(
            deviceOutput,
            img.pixels,
            sizeof(uchar4) * img.width * img.height,
            cudaMemcpyHostToDevice
        )
    );

    SAVE_CUDA(
        cudaMemcpyToSymbol(
            constVectors,
            classVectors,
            sizeof(Vector) * n
        )
    );

    SAVE_CUDA(cudaMemcpyToSymbol(actualSize, &n, sizeof(int)));

    kernel<<<DEVICE_BLOCKS, DEVICE_THREADS>>>(
        deviceOutput,
        img.width,
        img.height
    );

    cudaDeviceSynchronize();
    SAVE_CUDA(cudaGetLastError());

    SAVE_CUDA(
        cudaMemcpy(
            img.pixels,
            deviceOutput,
            sizeof(uchar4) * img.width * img.height,
            cudaMemcpyDeviceToHost
        )
    );

    saveImage(&img, outputPath.c_str());

    SAVE_CUDA(cudaFree(deviceOutput));
    free(img.pixels);
    delete[] classVectors;

    return 0;
}
