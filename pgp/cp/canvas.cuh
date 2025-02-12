#ifndef _CANVAS_H_
#define _CANVAS_H_

#include <cstdio>
#include <stdio.h>
#include <stdexcept>

#include "utils.cuh"

namespace Canvas {

    /* Types */

    typedef struct {
        unsigned char r;
        unsigned char g;
        unsigned char b;
        unsigned char a;
    } TColor;

    typedef struct {
        unsigned int width;
        unsigned int height;
        DeviceType deviceType;
        TColor *data;
        int *lock;
    } TCanvas;

    typedef struct {
        unsigned int x;
        unsigned int y;
    } TPosition;

    /* Methods */

    __host__ void Init(TCanvas *canvas, unsigned int width, unsigned int height, DeviceType device) {
        canvas->width = width;
        canvas->height = height;
        canvas->deviceType = device;

        if (device == DeviceType::GPU) {
            SAVE_CUDA(cudaMalloc(&canvas->data, sizeof(TColor) * width * height));
            SAVE_CUDA(cudaMalloc(&canvas->lock, sizeof(int)));
            SAVE_CUDA(cudaMemset(canvas->lock, 0, sizeof(int)));
        } else {
            canvas->data = (TColor*) std::malloc(sizeof(TColor) * width * height);
            if (!canvas->data) {
                throw std::runtime_error("Failed to allocate CPU memory for canvas data\n");
            }
        }
    }

    __host__ void Destroy(TCanvas *canvas) {
        if (canvas->deviceType == DeviceType::GPU) {
            cudaFree(canvas->data);
            cudaFree(canvas->lock);
        } else {
            std::free(canvas->data);
        }
    }

    __host__ void Dump(const TCanvas *canvas, const char *filename) {
        FILE *out = std::fopen(filename, "wb");
        if (!out) {
            throw std::runtime_error("Failed to open file");
        }

        std::fwrite(&canvas->width, sizeof(unsigned int), 1, out);
        std::fwrite(&canvas->height, sizeof(unsigned int), 1, out);

        if (canvas->deviceType == DeviceType::GPU) {
            TColor *data = (TColor*) std::malloc(sizeof(TColor) * canvas->width * canvas->height);
            if (!data) {
                throw std::runtime_error("Failed to allocate CPU memory for dumping GPU data");
            }

            SAVE_CUDA(cudaMemcpy(
                data,
                canvas->data,
                sizeof(TColor) * canvas->width * canvas->height,
                cudaMemcpyDeviceToHost
            ));

            std::fwrite(data, sizeof(TColor), canvas->width * canvas->height, out);
            std::free(data);
        } else {
            std::fwrite(canvas->data, sizeof(TColor), canvas->width * canvas->height, out);
        }

        std::fclose(out);
    }

    __host__ __device__ void PutPixel(TCanvas *canvas, TPosition pos, TColor color) {
        #ifdef __CUDA_ARCH__
            atomicAdd(canvas->lock, 1);
        #endif
        canvas->data[pos.y * canvas->width + pos.x] = color;
        #ifdef __CUDA_ARCH__
            atomicSub(canvas->lock, 1);
        #endif
    }

    __host__ __device__ TColor GetPixel(TCanvas *canvas, TPosition pos) {
        return canvas->data[pos.y * canvas->width + pos.x];
    }

    __host__ __device__ void AddPixel(TCanvas *canvas, TPosition pos, TColor color) {
        if (pos.x >= canvas->width || pos.y >= canvas->height) {
            return;
        }

        TColor sourceColor = GetPixel(canvas, pos);
        Canvas::TColor resultColor = {
            .r = (unsigned char) Min(255, int(color.r) + int(sourceColor.r)),
            .g = (unsigned char) Min(255, int(color.g) + int(sourceColor.g)),
            .b = (unsigned char) Min(255, int(color.b) + int(sourceColor.b)),
            .a = (unsigned char) Min(255, int(color.a) + int(sourceColor.a))
        };

        PutPixel(canvas, pos, resultColor);
    }

}

#endif // _CANVAS_H_
