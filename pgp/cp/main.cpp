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
    double transparent = 0.0;
};


struct TLight {
    Vector::TVector3 position;
    Vector::TVector3 color;
};


struct THit {
    bool exists;
    double t;
};


const double EPS = 1e-3;


void buildCube(const Vector::TVector3 &pos, const Vector::TVector3 &color, double c, double reflection, double transparent, std::vector<TPolygon> &out) {
    double x = pos.x;
    double y = pos.y;
    double z = pos.z;

    // top
    out.push_back({
        .verticles = {{x - c, y - c, z + c}, {x + c, y - c, z + c}, {x - c, y + c, z + c}},
        .color = color,
        .reflection = reflection,
        .transparent = transparent
    });

    out.push_back({
        .verticles = {{x + c, y - c, z + c}, {x + c, y + c, z + c}, {x - c, y + c, z + c}},
        .color = color,
        .reflection = reflection,
        .transparent = transparent
    });

    // right
    out.push_back({
        .verticles = {{x + c, y - c, z - c}, {x + c, y + c, z - c}, {x + c, y + c, z + c}},
        .color = color,
        .reflection = reflection,
        .transparent = transparent
    });

    out.push_back({
        .verticles = {{x + c, y - c, z - c}, {x + c, y - c, z + c}, {x + c, y + c, z + c}},
        .color = color,
        .reflection = reflection,
        .transparent = transparent
    });

    // left
    out.push_back({
        .verticles = {{x - c, y - c, z - c}, {x - c, y + c, z - c}, {x - c, y + c, z + c}},
        .color = color,
        .reflection = reflection,
        .transparent = transparent
    });

    out.push_back({
        .verticles = {{x - c, y - c, z - c}, {x - c, y - c, z + c}, {x - c, y + c, z + c}},
        .color = color,
        .reflection = reflection,
        .transparent = transparent
    });

    // bottom
    out.push_back({
        .verticles = {{x - c, y - c, z - c}, {x + c, y - c, z - c}, {x - c, y + c, z - c}},
        .color = color,
        .reflection = reflection,
        .transparent = transparent
    });

    out.push_back({
        .verticles = {{x + c, y + c, z - c}, {x + c, y - c, z - c}, {x - c, y + c, z - c}},
        .color = color,
        .reflection = reflection,
        .transparent = transparent
    });

    // back
    out.push_back({
        .verticles = {{x - c, y - c, z - c}, {x - c, y - c, z + c}, {x + c, y - c, z + c}},
        .color = color,
        .reflection = reflection,
        .transparent = transparent
    });

    out.push_back({
        .verticles = {{x - c, y - c, z - c}, {x + c, y - c, z - c}, {x + c, y - c, z + c}},
        .color = color,
        .reflection = reflection,
        .transparent = transparent
    });

    // front
    out.push_back({
        .verticles = {{x - c, y + c, z - c}, {x - c, y + c, z + c}, {x + c, y + c, z + c}},
        .color = color,
        .reflection = reflection,
        .transparent = transparent
    });

    out.push_back({
        .verticles = {{x - c, y + c, z - c}, {x + c, y + c, z - c}, {x + c, y + c, z + c}},
        .color = color,
        .reflection = reflection,
        .transparent = transparent
    });
}


void build_space(std::vector<TPolygon> &out) {
    buildCube({ 0.0, -3.0, 3.0 }, { 1.0, 0.0, 0.0 }, 2.0, 0.0, 0.0, out);
    // buildCube({ 0.0, 5.0, 0.0 }, { 0.0, 1.0, 0.0 }, 2.0, 0.5, 0.0, out);
    buildCube({ 0.0, 0.0, -8.0 }, { 1.0, 1.0, 1.0 }, 8.0, 0.0, 0.0, out);
}


const double EMBIENT_COEF = 0.3;
const double SPECULAR_COEF = 0.3;
const double DIFFUSE_COEF = 0.3;


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


THit CheckHitWithPolygon(const Vector::TVector3 &pos, const Vector::TVector3 &dir, const TPolygon &polygon) {
    THit emptyHit = { .exists = false, .t = 0.0 };

    Vector::TVector3 v0 = polygon.verticles[0];
    Vector::TVector3 v1 = polygon.verticles[1];
    Vector::TVector3 v2 = polygon.verticles[2];

    Vector::TVector3 E1 = Vector::Sub(v1, v0);
    Vector::TVector3 E2 = Vector::Sub(v2, v0);

    Vector::TVector3 D = dir;
    Vector::TVector3 T = Vector::Sub(pos, v0);

    Vector::TVector3 P = Vector::Prod(D, E2);
    Vector::TVector3 Q = Vector::Prod(T, E1);

    double divisor = Vector::Dot(P, E1);
    if (std::fabs(divisor) < 1e-10) return emptyHit;

    double u = Vector::Dot(P, T) / divisor;
    double v = Vector::Dot(Q, D) / divisor;
    double t = Vector::Dot(Q, E2) / divisor;

    if (u < 0.0 || u > 1.0) return emptyHit;
    if (v < 0.0 || v + u > 1.0) return emptyHit;
    if (t < 0.0) return emptyHit;

    return { .exists = true, .t = t };
}



Vector::TVector3 GetColor(
    const Vector::TVector3 &hitPos,
    const Vector::TVector3& dir,
    size_t polygonId,
    const std::vector<TPolygon> &polygons,
    const std::vector<TLight> &lights
) {
    Vector::TVector3 totalColor = { 0.0, 0.0, 0.0 };

    for (const TLight &light : lights) {
        Vector::TVector3 lightPos = light.position;
        Vector::TVector3 lightColor = light.color;

        const TPolygon &polygon = polygons[polygonId];
        Vector::TVector3 lightDir = Vector::Normalize(Vector::Sub(hitPos, lightPos));
        Vector::TVector3 dirNormalized = Vector::Normalize(dir);
        Vector::TVector3 resultColor = Vector::Mult(polygon.color, lightColor);

        // shade coef
        double shadeCoef = 1.0;
        THit currHit = CheckHitWithPolygon(lightPos, lightDir, polygons[polygonId]);
        if (!currHit.exists) {
            std::cerr << "[error] unexpected not existent hit\n";
            exit(1);
        }

        for (size_t i = 0; i < polygons.size(); ++i) {
            if (i == polygonId) continue;
            THit hit = CheckHitWithPolygon(lightPos, lightDir, polygons[i]);

            if (hit.exists && hit.t < currHit.t) {
                shadeCoef *= polygons[i].transparent;
            }
        }

        // embient light
        Vector::TVector3 embientColor = Vector::Mult(EMBIENT_COEF, polygon.color);

        // diffuse light
        Vector::TVector3 l = Vector::Mult(-1.0, lightDir);
        Vector::TVector3 n = GetPolygonNormal(polygon);
        double diffuseAngle = std::abs(Vector::Dot(n, l));
        double diffuseCoef = DIFFUSE_COEF * diffuseAngle;

        // specular light
        Vector::TVector3 reflectedLightDirection = Vector::Normalize(Reflect(
            Vector::Sub(hitPos, lightPos),
            GetPolygonNormal(polygon)
        ));
        double specularAngle = std::abs(Vector::Dot(reflectedLightDirection, dirNormalized));
        double specularCoef = SPECULAR_COEF * std::pow(specularAngle, 12);

        // total color
        Vector::TVector3 color = { 0.0, 0.0, 0.0 };
        color = Vector::Add(color, Vector::Mult(shadeCoef * (diffuseCoef + specularCoef), resultColor));
        color = Vector::Add(color, embientColor);
        totalColor = Vector::Add(totalColor, color);
    }

    totalColor = {
        std::min(1.0, std::max(0.0, totalColor.x)),
        std::min(1.0, std::max(0.0, totalColor.y)),
        std::min(1.0, std::max(0.0, totalColor.z))
    };

    return totalColor;
}


std::pair<Vector::TVector3, Vector::TVector3> GetReflectedRay(const Vector::TVector3 &pos, const Vector::TVector3 &dir, const TPolygon &polygon, const Vector::TVector3 &hitPosition) {
    Vector::TVector3 n = GetPolygonNormal(polygon);

    Vector::TVector3 nextDir = Reflect(dir, n);
    Vector::TVector3 nextPos = Vector::Add(hitPosition, Vector::Mult(EPS, nextDir));

    return { nextPos, nextDir };
}

Vector::TVector3 ray(Vector::TVector3 pos, Vector::TVector3 dir, const std::vector<TPolygon> &polygons, const std::vector<TLight> &lights, int depth) { 
    if (depth > 2) {
        return { 0.0, 0.0, 0.0 };
    }

    int k_min = -1;
    double ts_min;

    for(unsigned int k = 0; k < polygons.size(); k++) {
        THit hit = CheckHitWithPolygon(pos, dir, polygons[k]);
        double t = hit.t;

        if (!hit.exists) {
            continue;
        }

        if (k_min == -1 || t < ts_min) {
            k_min = k;
            ts_min = t;
        }
    }

    if (k_min == -1) {
        return { 0.0, 0.0, 0.0 };
	}

    TPolygon hitPolygon = polygons[k_min];
    Vector::TVector3 hitPosition = Vector::Add(pos, Vector::Mult(ts_min, dir));
    Vector::TVector3 hitColor = GetColor(hitPosition, dir, k_min, polygons, lights);

    std::pair<Vector::TVector3, Vector::TVector3> nextRay = GetReflectedRay(pos, dir, hitPolygon, hitPosition);
    Vector::TVector3 reflectedColor = ray(nextRay.first, nextRay.second, polygons, lights, depth + 1);

    Vector::TVector3 refractedDir = dir;
    Vector::TVector3 refractedPos = Vector::Add(hitPosition, Vector::Mult(EPS, refractedDir));
    Vector::TVector3 refractedColor = ray(refractedPos, refractedDir, polygons, lights, depth + 1);

    Vector::TVector3 resultColor = Vector::Add(
        Vector::Mult(hitPolygon.transparent, refractedColor),
        Vector::Add({ 1.0, 1.0, 1.0 }, Vector::Mult(hitPolygon.reflection, reflectedColor))
    );

    return { hitColor.x * resultColor.x, hitColor.y * resultColor.y, hitColor.z * resultColor.z };
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
void render(Vector::TVector3 pc, Vector::TVector3 pv, double angle, Canvas::TCanvas *canvas, const std::vector<TPolygon> &polygons, const std::vector<TLight> &lights) {
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
			Vector::TVector3 colorVector = ray(pc, Vector::Normalize(dir), polygons, lights, 0);
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

    std::vector<TLight> lights = {
        { .position = { 0.0, -6.0, 7.0 }, .color = { 0.5, 0.5, 0.5 } },
        { .position = { 0.0, 6.0, 7.0 }, .color = { 0.5, 0.5, 0.5 } }
    };

    for(unsigned int k = 0; k < 1; k += 1) { 
        cameraPos = { -6.0, 0.0, 7.0 };
        pv = { 1.0, 0.0, -1.0 };

        // cameraPos = (Vector::TVector3) {
		// 	6.0 * sin(0.05 * k),
		// 	6.0 * cos(0.05 * k),
		// 	7.0 + 2.0 * sin(0.1 * k)
		// }; // in scalar coords

        // pv = (Vector::TVector3) {
		// 	3.0 * sin(0.05 * k + M_PI),
		// 	3.0 * cos(0.05 * k + M_PI),
		// 	-1.0
		// };

        render(cameraPos, pv, 120.0, &canvas, polygons, lights);
    
        sprintf(buff, "build/%03d.data", k);
        printf("%d: %s\n", k, buff);    

        Canvas::Dump(&canvas, buff);
    }

    Canvas::Destroy(&canvas);
    return 0;
}
