#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include <iostream>
#include <string>


const uint32_t DEVICE_BLOCKS[2] = {16, 16};
const uint32_t DEVICE_THREADS[2] = {32, 32};


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


typedef struct _Image {
    int32_t width;
    int32_t height;
    uchar4* pixels;
} Image;


typedef struct _Texture {
    cudaArray* sourceArray;
    struct cudaResourceDesc resourceDesc;
    struct cudaTextureDesc textureDesc;
    cudaTextureObject_t texture;
} Texture;


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


Texture createTextureFromImage(const Image* image) {
    Texture texture;

    cudaChannelFormatDesc channelFormat = cudaCreateChannelDesc<uchar4>();
    SAVE_CUDA(cudaMallocArray(
        &texture.sourceArray,
        &channelFormat,
        image->width,
        image->height
    ));
    SAVE_CUDA(cudaMemcpy2DToArray(
        texture.sourceArray,
        0,
        0,
        image->pixels,
        image->width * sizeof(uchar4),
        image->width * sizeof(uchar4),
        image->height,
        cudaMemcpyHostToDevice
    ));

    memset(&texture.resourceDesc, 0, sizeof(texture.resourceDesc));
    texture.resourceDesc.resType = cudaResourceTypeArray;
    texture.resourceDesc.res.array.array = texture.sourceArray;

    memset(&texture.textureDesc, 0, sizeof(texture.textureDesc));
    texture.textureDesc.addressMode[0] = cudaAddressModeClamp;
    texture.textureDesc.addressMode[1] = cudaAddressModeClamp;
    texture.textureDesc.filterMode = cudaFilterModePoint;
    texture.textureDesc.normalizedCoords = false;

    texture.texture = 0;
    SAVE_CUDA(cudaCreateTextureObject(
        &texture.texture,
        &texture.resourceDesc,
        &texture.textureDesc,
        NULL
    ));

    return texture;
}

void free_texture(const Texture* texture) {
    SAVE_CUDA(cudaDestroyTextureObject(texture->texture));
    SAVE_CUDA(cudaFreeArray(texture->sourceArray));
}

__global__ void kernel(cudaTextureObject_t tex, uchar4* out, int w, int h) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idy = blockDim.y * blockIdx.y + threadIdx.y;
    int offsetx = blockDim.x * gridDim.x;
    int offsety = blockDim.y * gridDim.y;

    int x, y, dx, dy, result;
    double Y, GX, GY, G;
    uchar4 pixel;

    double sobelX[3][3] = {
        {-1.0, 0.0, 1.0},
        {-2.0, 0.0, 2.0},
        {-1.0, 0.0, 1.0}
    };

    double sobelY[3][3] = {
        {-1.0, -2.0, -1.0},
        {0.0, 0.0, 0.0},
        {1.0, 2.0, 1.0}
    };

    for (y = idy; y < h; y += offsety) {
        for (x = idx; x < w; x += offsetx) {
            GX = 0.0;
            GY = 0.0;
            G = 0.0;

            for (dx = -1; dx <= 1; ++dx) {
                for (dy = -1; dy <= 1; ++dy) {
                    pixel = tex2D<uchar4>(tex, x + dx, y + dy);
                    Y = 0.299 * pixel.x + 0.587 * pixel.y + 0.114 * pixel.z;
                    GX += sobelX[dy + 1][dx + 1] * Y;
                    GY += sobelY[dy + 1][dx + 1] * Y;
                }
            }

            G = sqrt(GX * GX + GY * GY);
            result = max(min(int(G), 255), 0);

            pixel = tex2D<uchar4>(tex, x, y);
            out[y * w + x] = make_uchar4(result, result, result, result);
        }
    }
}

int main() {
    uchar4* deviceOutput;
    Image img;
    Texture texture;

    std::string inputPath = "";
    std::string outputPath = "";

    std::getline(std::cin, inputPath);
    std::getline(std::cin, outputPath);

    img = loadImage(inputPath.c_str());
    texture = createTextureFromImage(&img);
    SAVE_CUDA(cudaMalloc(
        &deviceOutput,
        sizeof(uchar4) * img.width * img.height
    ));

    kernel<<<
        dim3(DEVICE_BLOCKS[0], DEVICE_BLOCKS[1]),
        dim3(DEVICE_THREADS[0], DEVICE_THREADS[1])
    >>>(
        texture.texture,
        deviceOutput,
        img.width,
        img.height
    );

    cudaDeviceSynchronize();
    SAVE_CUDA(cudaGetLastError());

    SAVE_CUDA(cudaMemcpy(
        img.pixels,
        deviceOutput,
        sizeof(uchar4) * img.width * img.height,
        cudaMemcpyDeviceToHost
    ));

    saveImage(&img, outputPath.c_str());

    SAVE_CUDA(cudaFree(deviceOutput));
    free_texture(&texture);
    free(img.pixels);

    return 0;
}
