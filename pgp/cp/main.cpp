// ffmpeg -framerate 30 -pattern_type glob -i '*.png' -c:v libx264 -pix_fmt yuv420p out.mp4

#include <stdlib.h>
#include <stdio.h> 
#include <cmath>

#include "canvas/canvas.h"
#include "vector/vector.h"


unsigned int SCREEN_WIDTH = 640;
unsigned int SCREEN_HEIGHT = 480;


struct TPolygon {
    Vector::TVector3 a;
    Vector::TVector3 b;
    Vector::TVector3 c;
    Canvas::TColor color;
};

TPolygon polygons[6];

void build_space() {
    polygons[0] = {{-5, -5, 0}, {5, -5, 0}, {-5, 5, 0}, {0, 0, 255, 255}};
    polygons[1] = {{5, 5, 0}, {5, -5, 0}, {-5, 5, 0}, {0, 0, 255, 255}};
    polygons[2] = {{-2,-2, 4}, {2, -2, 4}, {0, 2, 4}, {128, 0, 128, 255}};
    polygons[3] = {{-2, -2, 4}, {2, -2, 4}, {0, 0, 7}, {255, 0, 0, 255}};
    polygons[4] = {{-2,-2, 4}, {0, 0, 7}, {0, 2, 4}, {255, 255, 0, 255}};
    polygons[5] = {{0, 0, 7}, {2, -2, 4}, {0, 2, 4}, {0, 255, 0, 255}};
}

Canvas::TColor ray(Vector::TVector3 pos, Vector::TVector3 dir) { 
    int k, k_min = -1;
    double ts_min;

    for(k = 0; k < 6; k++) {
        Vector::TVector3 e1 = Vector::Sub(polygons[k].b, polygons[k].a);
        Vector::TVector3 e2 = Vector::Sub(polygons[k].c, polygons[k].a);
        Vector::TVector3 p = Vector::Prod(dir, e2);

        double div = Vector::Dot(p, e1);
        if (fabs(div) < 1e-10) continue;

        Vector::TVector3 t = Vector::Sub(pos, polygons[k].a);
        double u = Vector::Dot(p, t) / div;
        if (u < 0.0 || u > 1.0) continue;

        Vector::TVector3 q = Vector::Prod(t, e1);
        double v = Vector::Dot(q, dir) / div;
        if (v < 0.0 || v + u > 1.0) continue;

        double ts = Vector::Dot(q, e2) / div; 
        if (ts < 0.0) continue;

        if (k_min == -1 || ts < ts_min) {
            k_min = k;
            ts_min = ts;
        }    
    }

    if (k_min == -1) {
        return {0, 0, 0, 255};
	}
    
    return polygons[k_min].color;
}


/*
in[w * h]
out[2 * w * h]

biff[20 * w * h]
*/
void render(Vector::TVector3 pc, Vector::TVector3 pv, double angle, Canvas::TCanvas *canvas) {
    double dw = 2.0 / (canvas->width - 1.0);
    double dh = 2.0 / (canvas->height - 1.0);
    double z = 1.0 / tan(angle * M_PI / 360.0);

    Vector::TVector3 bz = Vector::Normalize(Vector::Sub(pv, pc));
    Vector::TVector3 bx = Vector::Normalize(Vector::Prod(bz, {0.0, 0.0, 1.0}));
    Vector::TVector3 by = Vector::Normalize(Vector::Prod(bx, bz));

    for(unsigned int i = 0; i < canvas->width; i++) {
        for(unsigned int j = 0; j < canvas->height; j++) {
            Vector::TVector3 v = {-1.0 + dw * i, (-1.0 + dh * j) * canvas->height / canvas->width, z};
            Vector::TVector3 dir = Vector::Mult(bx, by, bz, v);
			Canvas::TColor color = ray(pc, Vector::Normalize(dir));

            Canvas::PutPixel(canvas, { i, canvas->height - 1 - j }, color);
		}
	}
}


int main() {
    build_space();
    char buff[256];

    Vector::TVector3 cameraPos, pv;

    Canvas::TCanvas canvas;
    Canvas::Init(&canvas, SCREEN_WIDTH, SCREEN_HEIGHT);

    for(unsigned int k = 0; k < 5; k++) { 
        cameraPos = (Vector::TVector3) {
			6.0 * sin(0.05 * k),
			6.0 * cos(0.05 * k),
			5.0 + 2.0 * sin(0.1 * k)
		}; // in scalar coords

        pv = (Vector::TVector3) {
			3.0 * sin(0.05 * k + M_PI),
			3.0 * cos(0.05 * k + M_PI),
			0.0
		};

        render(cameraPos, pv, 120.0, &canvas);
    
        sprintf(buff, "build/%03d.data", k);
        printf("%d: %s\n", k, buff);    

        Canvas::Dump(&canvas, buff);
    }

    Canvas::Destroy(&canvas);
    return 0;
}
