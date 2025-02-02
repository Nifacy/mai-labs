#ifndef _GPU_RENDER_H_
#define _GPU_RENDER_H_

#include <vector>

#include "utils.cuh"
#include "canvas.cuh"
#include "vector.cuh"
#include "render.cuh"

namespace GpuRenderer {

    /* Constants */

    const unsigned int DEVICE_BLOCKS = 512;
    const unsigned int DEVICE_THREADS = 512;

    /* Methods */

    __global__ void Ssaa(Canvas::TCanvas src, Canvas::TCanvas dst, unsigned int coef) {
        int startX = blockDim.x * blockIdx.x + threadIdx.x;
        int startY = blockDim.y * blockIdx.y + threadIdx.y;
        int offsetX = blockDim.x * gridDim.x;
        int offsetY = blockDim.y * gridDim.y;

        double k = 1.0 / coef / coef;

        for (unsigned int x = startX; x < dst.width; x += offsetX) {
            for (unsigned int y = startY; y < dst.height; y += offsetY) {
                Vector::TVector3 color = { 0.0, 0.0, 0.0 };

                for (unsigned int dx = 0; dx < coef; ++dx) {
                    for (unsigned int dy = 0; dy < coef; ++dy) {
                        Canvas::TColor srcColor = Canvas::GetPixel(&src, { x * coef + dx, y * coef + dy });
                        color = Vector::Add(color, Renderer::ColorToVector(srcColor));
                    }
                }

                color = Vector::Mult(k, color);
                Canvas::PutPixel(&dst, { x, y }, Renderer::VectorToColor(color));
            }
        }
    }

    __global__ void _Kernel(
        Renderer::TRayTraceContext *current, int currentCount,
        Renderer::TRayTraceContext *next, int *cursor,
        Polygon::TPolygon *polygons, size_t polygonsAmount,
        Renderer::TLight *lights, size_t lightsAmount,
        Canvas::TCanvas canvas,
        int maxRecursionDepth
    ) {
        size_t offset = gridDim.x * blockDim.x;
        size_t start = blockDim.x * blockIdx.x + threadIdx.x;

        for (int j = start; j < currentCount; j += offset) {
            Renderer::TRayTraceContext el = current[j];
            Canvas::TColor color = Renderer::VectorToColor(Ray(el, polygons, polygonsAmount, lights, lightsAmount, next, cursor, maxRecursionDepth));
            Canvas::AddPixel(&canvas, el.pixelPos, color);
        }
    }

    __global__ void _InitRays(
        Canvas::TCanvas canvas,
        Vector::TVector3 pc, Vector::TVector3 pv, double angle,
        Renderer::TRayTraceContext *rays
    ) {
        double dw = 2.0 / (canvas.width - 1.0);
        double dh = 2.0 / (canvas.height - 1.0);
        double z = 1.0 / tan(angle * M_PI / 360.0);

        Vector::TVector3 bz = Vector::Normalize(Vector::Sub(pv, pc));
        Vector::TVector3 bx = Vector::Normalize(Vector::Prod(bz, {0.0, 0.0, 1.0}));
        Vector::TVector3 by = Vector::Normalize(Vector::Prod(bx, bz));

        int startX = blockDim.x * blockIdx.x + threadIdx.x;
        int startY = blockDim.y * blockIdx.y + threadIdx.y;
        int offsetX = blockDim.x * gridDim.x;
        int offsetY = blockDim.y * gridDim.y;

        for(unsigned int i = startX; i < canvas.width; i += offsetX) {
            for(unsigned int j = startY; j < canvas.height; j += offsetY) {
                Vector::TVector3 v = {-1.0 + dw * i, (-1.0 + dh * j) * canvas.height / canvas.width, z};
                Vector::TVector3 dir = Vector::Mult(bx, by, bz, v);
                Canvas::TPosition pixelPos = { i, canvas.height - 1 - j };

                rays[j * canvas.width + i] = {
                    .ray = {
                        .pos = pc,
                        .dir = Vector::Normalize(dir)
                    },
                    .color = { 1.0, 1.0, 1.0 },
                    .pixelPos = pixelPos,
                    .depth = 0
                };

                Canvas::PutPixel(&canvas, pixelPos, { 0, 0, 0, 255 });
            }
        }
    }

    void Render(
        Vector::TVector3 pc, Vector::TVector3 pv, double angle,
        Canvas::TCanvas *canvas,
        std::vector<Polygon::TPolygon> &polygons,
        std::vector<Renderer::TLight> &lights,
        size_t *raysCount,
        int maxRecursionDepth
    ) {
        size_t initialRayCount = canvas->width * canvas->height;
        int currentSize;

        Renderer::TRayTraceContext *rays1;
        SAVE_CUDA(cudaMalloc((void**) &rays1, 8 * initialRayCount * sizeof(Renderer::TRayTraceContext)));

        _InitRays<<<DEVICE_BLOCKS, DEVICE_THREADS>>>(*canvas, pc, pv, angle, rays1);
        cudaDeviceSynchronize();
        SAVE_CUDA(cudaGetLastError());

        int *rays1Count;
        SAVE_CUDA(cudaMalloc((void **) &rays1Count, sizeof(int)));
        SAVE_CUDA(cudaMemcpy(rays1Count, &initialRayCount, sizeof(int), cudaMemcpyHostToDevice));

        Renderer::TRayTraceContext *rays2;
        SAVE_CUDA(cudaMalloc((void**) &rays2, 8 * initialRayCount * sizeof(Renderer::TRayTraceContext)));

        int *rays2Count;
        SAVE_CUDA(cudaMalloc((void **) &rays2Count, sizeof(int)));
        SAVE_CUDA(cudaMemset(rays2Count, 0, sizeof(int)));

        Polygon::TPolygon *devicePolygons;
        SAVE_CUDA(cudaMalloc((void**) &devicePolygons, polygons.size() * sizeof(Polygon::TPolygon)));
        SAVE_CUDA(cudaMemcpy(devicePolygons, polygons.data(), polygons.size() * sizeof(Polygon::TPolygon), cudaMemcpyHostToDevice));

        Renderer::TLight *deviceLights;
        SAVE_CUDA(cudaMalloc((void**)& deviceLights, lights.size() * sizeof(Renderer::TLight)));
        SAVE_CUDA(cudaMemcpy(deviceLights, lights.data(), lights.size() * sizeof(Renderer::TLight), cudaMemcpyHostToDevice));

        for (int i = 0;; i = (i + 1) % 2) {
            Renderer::TRayTraceContext *current = (i % 2 == 0) ? rays1 : rays2;
            int *currentCount = (i % 2 == 0) ? rays1Count : rays2Count;

            Renderer::TRayTraceContext *next = (i % 2 == 0) ? rays2 : rays1;
            int *nextCount = (i % 2 == 0) ? rays2Count : rays1Count;

            cudaMemcpy(&currentSize, currentCount, sizeof(int), cudaMemcpyDeviceToHost);
            SAVE_CUDA(cudaMemset(nextCount, 0, sizeof(int)));
            *raysCount += currentSize;

            if (currentSize == 0) {
                break;
            }

            _Kernel<<<DEVICE_BLOCKS, DEVICE_THREADS>>>(
                current, currentSize,
                next, nextCount,
                devicePolygons, polygons.size(),
                deviceLights, lights.size(),
                *canvas,
                maxRecursionDepth
            );
            cudaDeviceSynchronize();
            SAVE_CUDA(cudaGetLastError());
        }

        cudaFree(devicePolygons);
        cudaFree(deviceLights);
        cudaFree(rays1);
        cudaFree(rays1Count);
        cudaFree(rays2);
        cudaFree(rays2Count);
    }
}

#endif // _GPU_RENDER_H_
