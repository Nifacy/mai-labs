#include "canvas.h"

#include <stdexcept>


namespace Canvas {

    void Init(TCanvas *canvas, unsigned int width, unsigned int height) {
        canvas->width = width;
        canvas->height = height;
        canvas->data = (TColor*) malloc(sizeof(TColor) * width * height);

        if (!canvas->data) {
            throw std::runtime_error("Failed to allocate memory for canvas data\n");
        }
    }


    void Destroy(TCanvas *canvas) {
        free(canvas->data);
    }


    void Dump(const TCanvas *canvas, const std::string &filename) {
        FILE *out = fopen(filename.c_str(), "wb");
        if (!out) {
            throw std::runtime_error("Failed to open file '" + filename + "'");
        }

        fwrite(&canvas->width, sizeof(unsigned int), 1, out);
        fwrite(&canvas->height, sizeof(unsigned int), 1, out);
        fwrite(canvas->data, sizeof(TColor), canvas->width * canvas->height, out);

        fclose(out);
    }

}
