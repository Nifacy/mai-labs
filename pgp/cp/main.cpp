#include "canvas/canvas.h"

#include <iostream>
#include <string>


int main(void) {
    std::string inputFile;
    Canvas::TCanvas canvas;

    std::cin >> inputFile;

    Canvas::Init(&canvas, 400, 400);
    Canvas::Dump(&canvas, inputFile);
    Canvas::Destroy(&canvas);

    return 0;
}
