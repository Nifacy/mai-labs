#ifndef _CPU_RENDER_H_
#define _CPU_RENDER_H_

#include "vector.cuh"
#include "canvas.cuh"
#include "polygon.cuh"
#include "render.cuh"

Canvas::TColor VectorToColor(Vector::TVector3 v) {
    return {
        (unsigned char) std::max(0, std::min(int(v.x * 255.0), 255)),
        (unsigned char) std::max(0, std::min(int(v.y * 255.0), 255)),
        (unsigned char) std::max(0, std::min(int(v.z * 255.0), 255)),
        255
    };
}

void render(Vector::TVector3 pc, Vector::TVector3 pv, double angle, Canvas::TCanvas *canvas, std::vector<Polygon::TPolygon> &polygons, std::vector<TLight> &lights) {
    double dw = 2.0 / (canvas->width - 1.0);
    double dh = 2.0 / (canvas->height - 1.0);
    double z = 1.0 / tan(angle * M_PI / 360.0);

    Vector::TVector3 bz = Vector::Normalize(Vector::Sub(pv, pc));
    Vector::TVector3 bx = Vector::Normalize(Vector::Prod(bz, {0.0, 0.0, 1.0}));
    Vector::TVector3 by = Vector::Normalize(Vector::Prod(bx, bz));

    size_t initialRayCount = canvas->width * canvas->height;
    TRay *rays1 = (TRay *) malloc(8 * initialRayCount * sizeof(TRay));
    TRay *rays2 = (TRay *) malloc(8 * initialRayCount * sizeof(TRay)); // Предполагаем, что потребуется в 2 раза больше
    int rays1Count = 0;
    int rays2Count = 0;

    // initialize rays
    for(unsigned int i = 0; i < canvas->width; i++) {
        for(unsigned int j = 0; j < canvas->height; j++) {
            Vector::TVector3 v = {-1.0 + dw * i, (-1.0 + dh * j) * canvas->height / canvas->width, z};
            Vector::TVector3 dir = Vector::Mult(bx, by, bz, v);

            TRay ray = {
                .pos = pc,
                .dir = Vector::Normalize(dir),
                .color = { 1.0, 1.0, 1.0 },
                .pixelPos = { i, canvas->height - 1 - j },
                .depth = 0
            };

            rays1[rays1Count++] = ray;
            Canvas::PutPixel(canvas, { i, canvas->height - 1 - j }, { 0, 0, 0 });
		}
	}

    int it = 0;

    for (int i = 0;; i = (i + 1) % 2) {
        TRay *current = (i % 2 == 0) ? rays1 : rays2;
        int &currentCount = (i % 2 == 0) ? rays1Count : rays2Count;
        TRay *next = (i % 2 == 0) ? rays2 : rays1;
        int &nextCount = (i % 2 == 0) ? rays2Count : rays1Count;

        if (currentCount == 0) {
            break;
        }

        nextCount = 0;
        std::cout << "iteration: " << it << ", rays: " << currentCount << std::endl;

        for (int j = 0; j < currentCount; ++j) {
            TRay el = current[j];
            Canvas::TColor color = VectorToColor(Ray(el, polygons.data(), polygons.size(), lights.data(), lights.size(), next, &nextCount, false));
            Canvas::TColor canvasColor = Canvas::GetPixel(canvas, { .x = el.pixelPos.x, .y = el.pixelPos.y });
            Canvas::TColor resultColor = {
                .r = (unsigned char) std::min(255, int(color.r) + int(canvasColor.r)),
                .g = (unsigned char) std::min(255, int(color.g) + int(canvasColor.g)),
                .b = (unsigned char) std::min(255, int(color.b) + int(canvasColor.b)),
                .a = (unsigned char) std::min(255, int(color.a) + int(canvasColor.a))
            };

            Canvas::PutPixel(canvas, { .x = el.pixelPos.x, .y = el.pixelPos.y }, resultColor);
        }

        currentCount = 0;
        it++;
    }

    free(rays1);
    free(rays2);
}

#endif // _CPU_RENDER_H_
