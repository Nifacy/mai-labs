#include "canvas.cuh"


int main() {
    Canvas::TCanvas canvas;

    Canvas::Init(&canvas, 200, 200);

    for (unsigned int x = 0; x < 100; ++x) {
        for (unsigned int y = 0; y < 100; ++y) {
            Canvas::PutPixel(&canvas, {x, y}, Canvas::TColor { 255, 0, 0, 255 });
        }
    }

    Canvas::Dump(&canvas, "build/0.data");
    Canvas::Destroy(&canvas);
}
