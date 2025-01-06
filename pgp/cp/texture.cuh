#ifndef _TEXTURE_CUH_
#define _TEXTURE_CUH_

#include "utils.cuh"
#include "vector.cuh"

#include <cstdio>
#include <stdexcept>

namespace Texture {

    /* Types */

    typedef unsigned char TPixel[4];

    struct TPosition {
        double x;
        double y;
    };

    struct TTexture {
        TPixel *data;
        unsigned int width;
        unsigned int height;
        DeviceType deviceType;
    };

    /* Methods */

    void Load(TTexture *out, const char *filepath, DeviceType deviceType) {
        FILE *in = fopen(filepath, "rb");
        if (!in) {
            throw std::runtime_error("Failed to open file '" + std::string(filepath) + "'");
        }

        fread(&out->width, sizeof(unsigned int), 1, in);
        fread(&out->height, sizeof(unsigned int), 1, in);
        out->deviceType = deviceType;

        size_t size = sizeof(TPixel) * out->width * out->height;
        TPixel *cpuData = (TPixel*) malloc(size);
        if (!cpuData) {
            fclose(in);
            throw std::runtime_error("Failed to allocate temporary CPU memory for texture");
        }

        fread(cpuData, sizeof(TPixel), out->width * out->height, in);
        fclose(in);

        if (deviceType == DeviceType::CPU) {
            out->data = cpuData;
        } else {
            SAVE_CUDA(cudaMalloc(&out->data, size));
            SAVE_CUDA(cudaMemcpy(out->data, cpuData, size, cudaMemcpyHostToDevice));
            free(cpuData);
        }
    }

    void Destroy(TTexture *texture) {
        if (texture->deviceType == DeviceType::CPU) {
            free(texture->data);
        } else {
            cudaFree(texture->data);
        }

        texture->data = nullptr;
        texture->width = 0;
        texture->height = 0;
    }

    __host__ __device__ Vector::TVector3 GetPixel(TTexture *texture, TPosition pos) {
        unsigned int x = (unsigned int) (Max(0.0, pos.x) * (texture->width - 1));
        x = Min(texture->width - 1, x);

        unsigned int y = (unsigned int) (Max(0.0, pos.y) * (texture->height - 1));
        y = Min(texture->height - 1, y);

        unsigned char *pixel = texture->data[(texture->height - y - 1) * texture->width + x];
        return { pixel[0] / 255.0, pixel[1] / 255.0, pixel[2] / 255.0 };
    }

}

#endif
