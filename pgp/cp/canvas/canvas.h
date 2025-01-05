#ifndef _CANVAS_H_
#define _CANVAS_H_


#include <string>


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

    // methods

    void Init(TCanvas *canvas, unsigned int width, unsigned int height);
    void Destroy(TCanvas *canvas);

    void Dump(const TCanvas *canvas, const std::string &filename);
    void PutPixel(TCanvas *canvas, const std::tuple<unsigned int, unsigned int> &pos, const TColor &color);
    TColor GetPixel(TCanvas *canvas, const std::tuple<unsigned int, unsigned int> &pos);
}

#endif  // _CANVAS_H_
