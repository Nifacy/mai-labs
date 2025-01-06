#ifndef _DEBUG_RENDER_H_
#define _DEBUG_RENDER_H_

#include "canvas.cuh"
#include "polygon.cuh"
#include "vector.cuh"

namespace DebugRenderer {

    /* Types */

    struct _TPoint {
        int x, y;
    };

    /* Methods */

    void _DrawLine(Canvas::TCanvas *canvas, _TPoint p1, _TPoint p2, Canvas::TColor color) {
        int x1 = p1.x;
        int y1 = p1.y;
        int x2 = p2.x;
        int y2 = p2.y;

        int dx = abs(x2 - x1);
        int dy = abs(y2 - y1);

        int sx = (x1 < x2) ? 1 : -1;
        int sy = (y1 < y2) ? 1 : -1;

        int err = dx - dy;

        while (true) {
            if (!(x1 < 0 || x1 >= canvas->width || y1 < 0 || y1 >= canvas->height)) {
                Canvas::PutPixel(canvas, {(unsigned int) x1, (unsigned int) y1}, color);
                Canvas::PutPixel(canvas, {(unsigned int) (x1 + 1), (unsigned int) y1}, color);
                Canvas::PutPixel(canvas, {(unsigned int) x1, (unsigned int) (y1 + 1)}, color);
                Canvas::PutPixel(canvas, {(unsigned int) (x1 + 1), (unsigned int) (y1 + 1)}, color);
            }

            if (x1 == x2 && y1 == y2) {
                break;
            }

            int e2 = 2 * err;
            if (e2 > -dy) {
                err -= dy;
                x1 += sx;
            }
            if (e2 < dx) {
                err += dx;
                y1 += sy;
            }
        }
    }


    void _DrawDot(Canvas::TCanvas *canvas, _TPoint p1) {
        for (int x = p1.x - 1; x <= p1.x + 1; x++) {
            for (int y = p1.y - 1; y <= p1.y + 1; ++y) {
                if (x < 0 || x >= canvas->width || y < 0 || y >= canvas->height) {
                    continue;
                }

                Canvas::PutPixel(canvas, { (unsigned int) x, (unsigned int) y }, { 0, 255, 0 });
            }
        }
    }


    _TPoint _WorldToCanvas(Vector::TVector3 v, Vector::TVector3 pc, Vector::TVector3 pv, double angle, Canvas::TCanvas *canvas) {
        // Расчет направления камеры
        Vector::TVector3 bz = Vector::Normalize(Vector::Sub(pv, pc)); // Вектор вперед
        Vector::TVector3 bx = Vector::Normalize(Vector::Prod(bz, {0.0, 0.0, 1.0})); // Вектор вправо
        Vector::TVector3 by = Vector::Normalize(Vector::Prod(bx, bz)); // Вектор вверх

        // Перемещение точки в систему координат камеры
        Vector::TVector3 relative = Vector::Sub(v, pc); // Вектор от камеры к точке
        double xCam = Vector::Dot(relative, bx);
        double yCam = Vector::Dot(relative, by);
        double zCam = Vector::Dot(relative, bz);

        // Если точка за камерой, она невидима
        if (zCam <= 0) {
            return { -1, -1 }; // Условный признак невидимой точки
        }

        // Угол обзора в радианах
        double z = 1.0 / tan(angle * M_PI / 360.0);
        double aspectRatio = static_cast<double>(canvas->width) / canvas->height;

        // Преобразование в экранные координаты
        double xProj = xCam / (zCam * aspectRatio) * z;
        double yProj = yCam / zCam * z;

        int xScreen = static_cast<int>((xProj + 1.0) * (canvas->width - 1) / 2.0);
        int yScreen = static_cast<int>((1.0 - yProj) * (canvas->height - 1) / 2.0);

        return { xScreen, yScreen };
    }


    void Render(Canvas::TCanvas canvas, Polygon::TPolygon *polygons, size_t polygonsAmount) {
        // TODO: remove when camera will be added
        double angle = 120.0;
        Vector::TVector3 pc = { 0.4, 6.0, 6.0 };
        Vector::TVector3 pv = { 0.0, -3.0, -2.0 };

        for (unsigned int x = 0; x < canvas.width; ++x) {
            for (unsigned int y = 0; y < canvas.height; ++y) {
                Canvas::PutPixel(&canvas, { x, y }, {0, 0, 0, 255});
            }
        }

        for (size_t i = 0; i < polygonsAmount; ++i) {
            Polygon::TPolygon polygon = polygons[i];

            size_t vertexCount = 3;
            Vector::TVector3 center = { 0.0, 0.0, 0.0 };

            for (size_t i = 0; i < vertexCount; i++) {
                // Определяем текущую вершину и следующую вершину (замыкаем полигон на последней грани)
                const Vector::TVector3 &v1 = polygon.verticles[i];
                const Vector::TVector3 &v2 = polygon.verticles[(i + 1) % vertexCount];

                // Преобразуем координаты из мира в экранные
                _TPoint p1 = _WorldToCanvas(v1, pc, pv, angle, &canvas);
                _TPoint p2 = _WorldToCanvas(v2, pc, pv, angle, &canvas);

                // Рисуем линию между текущей и следующей вершинами
                _DrawLine(&canvas, p1, p2, Canvas::TColor{255, 0, 0, 255}); // Белый цвет для линий
                // _DrawDot(canvas, p1);
                // _DrawDot(canvas, p2);
                center = Vector::Add(center, Vector::Mult(0.33, v1));
            }

            const Vector::TVector3 n = Vector::Add(center, Polygon::GetNormal(polygon));
            _TPoint p1 = _WorldToCanvas(center, pc, pv, angle, &canvas);
            _TPoint p2 = _WorldToCanvas(n, pc, pv, angle, &canvas);
            _DrawLine(&canvas, p1, p2, Canvas::TColor{0, 255, 0, 255});
        }
    }

}

#endif // _DEBUG_RENDER_H_
