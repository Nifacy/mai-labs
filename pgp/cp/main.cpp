// ffmpeg -framerate 30 -pattern_type glob -i '*.png' -c:v libx264 -pix_fmt yuv420p out.mp4

#include <stdlib.h>
#include <stdio.h> 
#include <cmath>
#include <vector>
#include <iostream>

#include "canvas/canvas.h"
#include "vector/vector.h"


unsigned int SCREEN_WIDTH = 640;
unsigned int SCREEN_HEIGHT = 480;


struct TPolygon {
    Vector::TVector3 verticles[3];
    Vector::TVector3 color;
    double reflection = 0.0;
};


const double EPS = 1e-3;


void buildCube(const Vector::TVector3 &pos, const Vector::TVector3 &color, double c, double reflection, std::vector<TPolygon> &out) {
    double x = pos.x;
    double y = pos.y;
    double z = pos.z;

    out.push_back({
        .verticles = {{x - c, y - c, z + c}, {x + c, y - c, z + c}, {x - c, y + c, z + c}},
        .color = color,
        .reflection = reflection
    });

    out.push_back({
        .verticles = {{x + c, y + c, z + c}, {x + c, y - c, z + c}, {x - c, y + c, z + c}},
        .color = color,
        .reflection = reflection
    });

    out.push_back({
        .verticles = {{x + c, y - c, z - c}, {x + c, y + c, z - c}, {x + c, y + c, z + c}},
        .color = color,
        .reflection = reflection
    });

    out.push_back({
        .verticles = {{x + c, y - c, z - c}, {x + c, y - c, z + c}, {x + c, y + c, z + c}},
        .color = color,
        .reflection = reflection
    });

    out.push_back({
        .verticles = {{x - c, y - c, z - c}, {x - c, y + c, z - c}, {x - c, y + c, z + c}},
        .color = color,
        .reflection = reflection
    });

    out.push_back({
        .verticles = {{x - c, y - c, z - c}, {x - c, y - c, z + c}, {x - c, y + c, z + c}},
        .color = color,
        .reflection = reflection
    });

    out.push_back({
        .verticles = {{x - c, y - c, z - c}, {x + c, y - c, z - c}, {x - c, y + c, z - c}},
        .color = color,
        .reflection = reflection
    });

    out.push_back({
        .verticles = {{x + c, y + c, z - c}, {x + c, y - c, z - c}, {x - c, y + c, z - c}},
        .color = color,
        .reflection = reflection
    });

    out.push_back({
        .verticles = {{x - c, y - c, z - c}, {x - c, y - c, z + c}, {x + c, y - c, z + c}},
        .color = color,
        .reflection = reflection
    });

    out.push_back({
        .verticles = {{x - c, y - c, z - c}, {x + c, y - c, z - c}, {x + c, y - c, z + c}},
        .color = color,
        .reflection = reflection
    });
}


void build_space(std::vector<TPolygon> &out) {
    buildCube({ 0.0, -3.0, 3.0 }, { 1.0, 0.0, 0.0 }, 2.0, 0.5, out);
    buildCube({ 0.0, 5.0, 0.0 }, { 0.0, 1.0, 0.0 }, 2.0, 0.5, out);
}


Vector::TVector3 GetPolygonNormal(const TPolygon &polygon) {
    Vector::TVector3 v1 = Vector::Sub(polygon.verticles[1], polygon.verticles[0]);
    Vector::TVector3 v2 = Vector::Sub(polygon.verticles[2], polygon.verticles[0]);
    return Vector::Normalize(Vector::Prod(v1, v2));
}


Vector::TVector3 Reflect(const Vector::TVector3 &v, const Vector::TVector3 &normal) {
    double k = -2.0 * Vector::Dot(v, normal);
    Vector::TVector3 temp = { k * normal.x, k * normal.y, k * normal.z };
    Vector::TVector3 rVec = Vector::Add(temp, v);
    return Vector::Normalize(rVec);
}


std::pair<Vector::TVector3, Vector::TVector3> GetReflectedRay(const Vector::TVector3 &pos, const Vector::TVector3 &dir, const TPolygon &polygon, double t) {
    Vector::TVector3 n = GetPolygonNormal(polygon);

    Vector::TVector3 hitPosition = Vector::Add(pos, Vector::Mult(t, dir));
    Vector::TVector3 nextDir = Reflect(dir, n);
    Vector::TVector3 nextPos = Vector::Add(hitPosition, Vector::Mult(EPS, nextDir));

    return { nextPos, nextDir };
}

Vector::TVector3 ray(Vector::TVector3 pos, Vector::TVector3 dir, const std::vector<TPolygon> &polygons, int depth) { 
    if (depth > 2) {
        return { 0.0, 0.0, 0.0 };
    }

    int k_min = -1;
    double ts_min;

    for(unsigned int k = 0; k < polygons.size(); k++) {
        Vector::TVector3 v0 = polygons[k].verticles[0];
        Vector::TVector3 v1 = polygons[k].verticles[1];
        Vector::TVector3 v2 = polygons[k].verticles[2];

        Vector::TVector3 E1 = Vector::Sub(v1, v0);
        Vector::TVector3 E2 = Vector::Sub(v2, v0);

        Vector::TVector3 D = dir;
        Vector::TVector3 T = Vector::Sub(pos, v0);

        Vector::TVector3 P = Vector::Prod(D, E2);
        Vector::TVector3 Q = Vector::Prod(T, E1);

        double divisor = Vector::Dot(P, E1);
        if (std::fabs(divisor) < 1e-10) continue;

        double u = Vector::Dot(P, T) / divisor;
        double v = Vector::Dot(Q, D) / divisor;
        double t = Vector::Dot(Q, E2) / divisor;

        if (u < 0.0 || u > 1.0) continue;
        if (v < 0.0 || v + u > 1.0) continue;
        if (t < 0.0) continue;

        if (k_min == -1 || t < ts_min) {
            k_min = k;
            ts_min = t;
        }    
    }

    if (k_min == -1) {
        return { 0.0, 0.0, 0.0 };
	}

    TPolygon hitPolygon = polygons[k_min];
    Vector::TVector3 hitColor = hitPolygon.color;

    std::pair<Vector::TVector3, Vector::TVector3> nextRay = GetReflectedRay(pos, dir, hitPolygon, ts_min);
    Vector::TVector3 reflectedColor = ray(nextRay.first, nextRay.second, polygons, depth + 1);

    Vector::TVector3 resultColor = Vector::Add(hitColor, Vector::Mult(hitPolygon.reflection, reflectedColor));

    return resultColor;
}


Canvas::TColor VectorToColor(const Vector::TVector3 &v) {
    return {
        (unsigned char) std::max(0, std::min(int(v.x * 255.0), 255)),
        (unsigned char) std::max(0, std::min(int(v.y * 255.0), 255)),
        (unsigned char) std::max(0, std::min(int(v.z * 255.0), 255)),
        255
    };
}


/*
in[w * h]
out[2 * w * h]

biff[20 * w * h]
*/
void render(Vector::TVector3 pc, Vector::TVector3 pv, double angle, Canvas::TCanvas *canvas, const std::vector<TPolygon> &polygons) {
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
			Vector::TVector3 colorVector = ray(pc, Vector::Normalize(dir), polygons, 0);
            Canvas::TColor color = VectorToColor(colorVector);

            Canvas::PutPixel(canvas, { i, canvas->height - 1 - j }, color);
		}
	}
}


int main() {
    char buff[256];

    std::vector<TPolygon> polygons;
    Vector::TVector3 cameraPos, pv;

    Canvas::TCanvas canvas;
    Canvas::Init(&canvas, SCREEN_WIDTH, SCREEN_HEIGHT);

    build_space(polygons);

    for(unsigned int k = 0; k < 30; k += 1) { 
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

        render(cameraPos, pv, 120.0, &canvas, polygons);
    
        sprintf(buff, "build/%03d.data", k);
        printf("%d: %s\n", k, buff);    

        Canvas::Dump(&canvas, buff);
    }

    Canvas::Destroy(&canvas);
    return 0;
}
