#include <iostream>
#include <vector>
#include <string>

#include "texture.cuh"
#include "texture_projection.cuh"
#include "utils.cuh"
#include "canvas.cuh"
#include "polygon.cuh"
#include "debug_render.cuh"
#include "cpu_render.cuh"
#include "render.cuh"
#include "scene.cuh"


__device__ Canvas::TColor VectorToColor2(Vector::TVector3 v) {
    return {
        (unsigned char) Max(0, Min(int(v.x * 255.0), 255)),
        (unsigned char) Max(0, Min(int(v.y * 255.0), 255)),
        (unsigned char) Max(0, Min(int(v.z * 255.0), 255)),
        255
    };
}

__global__ void kernel(
    TRay *current, int currentCount,
    TRay *next, int *cursor,
    Polygon::TPolygon *polygons, size_t polygonsAmount,
    TLight *lights, size_t lightsAmount,
    Canvas::TCanvas canvas,
    int *lock
) {
    size_t offset = gridDim.x * blockDim.x;
    size_t start = blockDim.x * blockIdx.x + threadIdx.x;

    for (int j = start; j < currentCount; j += offset) {
        TRay el = current[j];
        Canvas::TColor color = VectorToColor2(Ray(el, polygons, polygonsAmount, lights, lightsAmount, next, cursor, true));
        atomicAdd(lock, 1);
        Canvas::TColor canvasColor = Canvas::GetPixel(&canvas, { .x = el.pixelPos.x, .y = el.pixelPos.y });
        Canvas::TColor resultColor = {
            .r = (unsigned char) Min(255, int(color.r) + int(canvasColor.r)),
            .g = (unsigned char) Min(255, int(color.g) + int(canvasColor.g)),
            .b = (unsigned char) Min(255, int(color.b) + int(canvasColor.b)),
            .a = (unsigned char) Min(255, int(color.a) + int(canvasColor.a))
        };

        Canvas::PutPixel(&canvas, { .x = el.pixelPos.x, .y = el.pixelPos.y }, resultColor);
        atomicSub(lock, 1);
    }
}

__global__ void initRays(Canvas::TCanvas canvas, Vector::TVector3 pc, Vector::TVector3 pv, double angle, TRay *rays) {
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
            rays[j * canvas.width + i] = {
                .pos = pc,
                .dir = Vector::Normalize(dir),
                .color = { 1.0, 1.0, 1.0 },
                .pixelPos = { i, canvas.height - 1 - j },
                .depth = 0
            };

            Canvas::PutPixel(&canvas, { i, canvas.height - 1 - j }, { 0, 0, 0, 255 });
		}
	}
}

void GpuRender2(Vector::TVector3 pc, Vector::TVector3 pv, double angle, Canvas::TCanvas *canvas, std::vector<Polygon::TPolygon> &polygons, std::vector<TLight> &lights) {
    size_t initialRayCount = canvas->width * canvas->height;

    TRay *rays1;
    SAVE_CUDA(cudaMalloc((void**) &rays1, 8 * initialRayCount * sizeof(TRay)));
    initRays<<<100, 100>>>(*canvas, pc, pv, angle, rays1);
    cudaDeviceSynchronize();
    SAVE_CUDA(cudaGetLastError());

    int *rays1Count;
    SAVE_CUDA(cudaMalloc((void **) &rays1Count, sizeof(int)));
    SAVE_CUDA(cudaMemcpy(rays1Count, &initialRayCount, sizeof(int), cudaMemcpyHostToDevice));

    int *lock;
    SAVE_CUDA(cudaMalloc((void **) &lock, sizeof(int)));

    TRay *rays2;
    SAVE_CUDA(cudaMalloc((void**) &rays2, 8 * initialRayCount * sizeof(TRay)));

    int *rays2Count;
    SAVE_CUDA(cudaMalloc((void **) &rays2Count, sizeof(int)));
    SAVE_CUDA(cudaMemset(rays2Count, 0, sizeof(int)));

    Polygon::TPolygon *devicePolygons;
    SAVE_CUDA(cudaMalloc((void**) &devicePolygons, polygons.size() * sizeof(Polygon::TPolygon)));
    SAVE_CUDA(cudaMemcpy(devicePolygons, polygons.data(), polygons.size() * sizeof(Polygon::TPolygon), cudaMemcpyHostToDevice));

    TLight *deviceLights;
    SAVE_CUDA(cudaMalloc((void**)& deviceLights, lights.size() * sizeof(TLight)));
    SAVE_CUDA(cudaMemcpy(deviceLights, lights.data(), lights.size() * sizeof(TLight), cudaMemcpyHostToDevice));

    int it = 0;

    for (int i = 0;; i = (i + 1) % 2) {
        TRay *current = (i % 2 == 0) ? rays1 : rays2;
        int *currentCount = (i % 2 == 0) ? rays1Count : rays2Count;
        TRay *next = (i % 2 == 0) ? rays2 : rays1;
        int *nextCount = (i % 2 == 0) ? rays2Count : rays1Count;

        int tmp;
        cudaMemcpy(&tmp, currentCount, sizeof(int), cudaMemcpyDeviceToHost);
        SAVE_CUDA(cudaMemset(nextCount, 0, sizeof(int)));

        std::cout << "iteration: " << it << ", rays: " << tmp << std::endl;
        
        if (tmp == 0) {
            break;
        }

        kernel<<<200, 200>>>(
            current, tmp,
            next, nextCount,
            devicePolygons, polygons.size(),
            deviceLights, lights.size(),
            *canvas,
            lock
        );
        cudaDeviceSynchronize();
        SAVE_CUDA(cudaGetLastError());
        std::cout << "iteration: " << it << ", rays: " << tmp << " [ok]" << std::endl;

        it++;
    }

    cudaFree(devicePolygons);
    cudaFree(deviceLights);
    cudaFree(rays1);
    cudaFree(rays1Count);
    cudaFree(rays2);
    cudaFree(rays2Count);
    cudaFree(lock);
}


__global__ void GpuDraw(Canvas::TCanvas canvas) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idy = blockDim.y * blockIdx.y + threadIdx.y;
    int offsetx = blockDim.x * gridDim.x;
    int offsety = blockDim.y * gridDim.y;

    for (unsigned int x = idx; x < canvas.width; x += offsetx) {
        for (unsigned int y = idy; y < canvas.height; y += offsety) {
            Canvas::PutPixel(&canvas, Canvas::TPosition { x, y }, Canvas::TColor { 255, 0, 0, 255 });
        }
    }
}

/* Cpu render */

void CpuDraw(Canvas::TCanvas canvas) {
    for (unsigned int x = 0; x < canvas.width; ++x) {
        for (unsigned int y = 0; y < canvas.height; ++y) {
            Canvas::PutPixel(&canvas, Canvas::TPosition { x, y }, Canvas::TColor { 255, 0, 0, 255 });
        }
    }
}


int main(int argc, char *argv[]) {
    std::string deviceTypeArg = std::string(argv[1]);    
    Canvas::TCanvas canvas;

    DeviceType deviceType = (deviceTypeArg == "gpu") ? DeviceType::GPU : DeviceType::CPU;

    Canvas::Init(&canvas, 400, 400, deviceType);

    std::vector<Polygon::TPolygon> polygons;
    std::vector<TLight> lights = {
        { .position = { 5.0, 5.0, 5.0 }, .color = { 1.0, 1.0, 1.0 } }
    };

    build_space(polygons, deviceType);

    if (deviceTypeArg == "gpu") {
        std::cerr << "[log] using GPU render ..." << std::endl;
        GpuRender2(
            { 0.0, 6.0, 4.0 },
            { 0.0, -3.0, -1.0 },
            120.0,
            &canvas,
            polygons, lights
        );
    } else if (deviceTypeArg == "cpu") {
        std::cerr << "[log] using CPU render ..." << std::endl;
        render(
            { 0.0, 6.0, 4.0 },
            { 0.0, -3.0, -1.0 },
            120.0,
            &canvas,
            polygons, lights);
    } else {
        std::cerr << "[log] using debug render ..." << std::endl;
        DebugRenderer::Render(canvas, polygons.data(), polygons.size());
    }

    Canvas::Dump(&canvas, "build/0.data");
    Canvas::Destroy(&canvas);

    return 0;
}
