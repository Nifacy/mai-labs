#ifndef _CANVAS_H_
#define _CANVAS_H_


#include <cstdlib>
#include <cstdio>


namespace Canvas {

    typedef struct {
        unsigned char r;
        unsigned char g;
        unsigned char b;
        unsigned char a;
    } TColor;

    typedef struct {
        unsigned int width;
        unsigned int height;
        TColor *data;
    } TCanvas;

    typedef struct {
        unsigned int x;
        unsigned int y;
    } TPosition;

    /* Methods */

    void Init(TCanvas *canvas, unsigned int width, unsigned int height) {
        canvas->width = width;
        canvas->height = height;
        canvas->data = (TColor*) std::malloc(sizeof(TColor) * width * height);

        if (!canvas->data) {
            std::exit(1);
        }
    }

    void Destroy(TCanvas *canvas) {
        free(canvas->data);
    }

    void Dump(const TCanvas *canvas, const char *filename) {
        FILE *out = std::fopen(filename, "wb");
        if (!out) {
            std::exit(1);
        }

        std::fwrite(&canvas->width, sizeof(unsigned int), 1, out);
        std::fwrite(&canvas->height, sizeof(unsigned int), 1, out);
        std::fwrite(canvas->data, sizeof(TColor), canvas->width * canvas->height, out);

        std::fclose(out);
    }

    void PutPixel(TCanvas *canvas, TPosition pos, TColor color) {
        canvas->data[pos.y * canvas->width + pos.x] = color;
    }

    TColor GetPixel(TCanvas *canvas, TPosition pos) {
        return canvas->data[pos.y * canvas->width + pos.x];
    }

}

#endif  // _CANVAS_H_
