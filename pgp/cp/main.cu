#include "utils.cuh"
#include "canvas.cuh"


__global__ void Draw(Canvas::TCanvas canvas) {
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


int main() {
    Canvas::TCanvas canvas;

    Canvas::Init(&canvas, 200, 200, Canvas::DeviceType::GPU);

    Draw<<<100, 100>>>(canvas);

    cudaDeviceSynchronize();
    SAVE_CUDA(cudaGetLastError());

    Canvas::Dump(&canvas, "build/0.data");
    Canvas::Destroy(&canvas);
}
