#include <iostream>
#include <vector>
#include <string>

#include "utils.cuh"
#include "canvas.cuh"
#include "polygon.cuh"
#include "debug_render.cuh"


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


void CpuDraw(Canvas::TCanvas canvas) {
    for (unsigned int x = 0; x < canvas.width; ++x) {
        for (unsigned int y = 0; y < canvas.height; ++y) {
            Canvas::PutPixel(&canvas, Canvas::TPosition { x, y }, Canvas::TColor { 255, 0, 0, 255 });
        }
    }
}


int main(int argc, char *argv[]) {
    std::string deviceType = std::string(argv[1]);    
    Canvas::TCanvas canvas;

    Canvas::Init(&canvas, 200, 200, (deviceType == "gpu") ? Canvas::DeviceType::GPU : Canvas::DeviceType::CPU);

    std::vector<Polygon::TPolygon> polygons = {
        {
            .verticles = { { 0.0, 0.0, 0.0 }, { 0.0, 0.0, 3.0 }, { 3.0, 0.0, 3.0 } }
        }
    };

    if (deviceType == "gpu") {
        std::cerr << "[log] using GPU render ..." << std::endl;
        GpuDraw<<<100, 100>>>(canvas);
        cudaDeviceSynchronize();
        SAVE_CUDA(cudaGetLastError());    
    } else if (deviceType == "debug") {
        std::cerr << "[log] using debug render ..." << std::endl;
        DebugRenderer::Render(canvas, polygons.data(), polygons.size());
    } else {
        std::cerr << "[log] using CPU render ..." << std::endl;
        CpuDraw(canvas);
    }

    Canvas::Dump(&canvas, "build/0.data");
    Canvas::Destroy(&canvas);

    return 0;
}
