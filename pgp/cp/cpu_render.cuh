#ifndef _CPU_RENDER_H_
#define _CPU_RENDER_H_

#include "vector.cuh"
#include "canvas.cuh"
#include "polygon.cuh"
#include "render.cuh"

namespace CpuRenderer {

    void Ssaa(Canvas::TCanvas *src, Canvas::TCanvas *dst, unsigned int coef) {
        double k = 1.0 / coef / coef;

        for (unsigned int x = 0; x < dst->width; ++x) {
            for (unsigned int y = 0; y < dst->height; ++y) {
                Vector::TVector3 color = { 0.0, 0.0, 0.0 };

                for (unsigned int dx = 0; dx < coef; ++dx) {
                    for (unsigned int dy = 0; dy < coef; ++dy) {
                        Canvas::TColor srcColor = Canvas::GetPixel(src, { x * coef + dx, y * coef + dy });
                        color = Vector::Add(color, Renderer::ColorToVector(srcColor));
                    }
                }

                color = Vector::Mult(k, color);
                Canvas::PutPixel(dst, { x, y }, Renderer::VectorToColor(color));
            }
        }
    }

    void Render(
        Vector::TVector3 pc, Vector::TVector3 pv, double angle,
        Canvas::TCanvas *canvas,
        std::vector<Polygon::TPolygon> &polygons,
        std::vector<Renderer::TLight> &lights,
        size_t *raysCount,
        int maxRecusionDepth
    ) {
        double dw = 2.0 / (canvas->width - 1.0);
        double dh = 2.0 / (canvas->height - 1.0);
        double z = 1.0 / tan(angle * M_PI / 360.0);

        Vector::TVector3 bz = Vector::Normalize(Vector::Sub(pv, pc));
        Vector::TVector3 bx = Vector::Normalize(Vector::Prod(bz, {0.0, 0.0, 1.0}));
        Vector::TVector3 by = Vector::Normalize(Vector::Prod(bx, bz));

        size_t initialRayCount = canvas->width * canvas->height;
        Renderer::TRayTraceContext *rays1 = (Renderer::TRayTraceContext*) std::malloc(8 * initialRayCount * sizeof(Renderer::TRayTraceContext));
        int rays1Count = 0;

        Renderer::TRayTraceContext *rays2 = (Renderer::TRayTraceContext*) std::malloc(8 * initialRayCount * sizeof(Renderer::TRayTraceContext));
        int rays2Count = 0;

        // initialize rays
        for(unsigned int i = 0; i < canvas->width; i++) {
            for(unsigned int j = 0; j < canvas->height; j++) {
                Vector::TVector3 v = {-1.0 + dw * i, (-1.0 + dh * j) * canvas->height / canvas->width, z};
                Canvas::TPosition pixelPos = { i, canvas->height - 1 - j };

                rays1[rays1Count++] = {
                    .ray = {
                        .pos = pc,
                        .dir = Vector::Normalize(Vector::Mult(bx, by, bz, v))
                    },
                    .color = { 1.0, 1.0, 1.0 },
                    .pixelPos = pixelPos,
                    .depth = 0
                };

                Canvas::PutPixel(canvas, pixelPos, { 0, 0, 0, 255 });
            }
        }

        for (int i = 0;; i = (i + 1) % 2) {
            Renderer::TRayTraceContext *current = (i % 2 == 0) ? rays1 : rays2;
            int &currentCount = (i % 2 == 0) ? rays1Count : rays2Count;

            Renderer::TRayTraceContext *next = (i % 2 == 0) ? rays2 : rays1;
            int &nextCount = (i % 2 == 0) ? rays2Count : rays1Count;
            *raysCount += currentCount;

            if (currentCount == 0) {
                break;
            }

            nextCount = 0;

            for (int j = 0; j < currentCount; ++j) {
                Renderer::TRayTraceContext el = current[j];
                Canvas::TColor color = Renderer::VectorToColor(Ray(el, polygons.data(), polygons.size(), lights.data(), lights.size(), next, &nextCount, maxRecusionDepth));
                Canvas::AddPixel(canvas, el.pixelPos, color);
            }

            currentCount = 0;
        }

        std::free(rays1);
        std::free(rays2);
    }

}

#endif // _CPU_RENDER_H_
