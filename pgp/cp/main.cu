#include <iostream>
#include <vector>
#include <string>

#include "utils.cuh"
#include "canvas.cuh"
#include "polygon.cuh"
#include "debug_render.cuh"


__global__ void GpuDraw(Canvas::TCanvas canvas) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idy = blockDim.y * blockIdx.y + threadIdx.y;
    int offsetx = blockDim.x * gridDim.x;
    int offsety = blockDim.y * gridDim.y;

    for (unsigned int x = idx; x < canvas.width; x += offsetx) {
        for (unsigned int y = idy; y < canvas.height; y += offsety) {
            Canvas::PutPixel(&canvas, Canvas::TPosition { x, y }, Canvas::TColor { 255, 0, 0, 255 });
        }
    }
}

/* Cpu render */

struct TLight {
    Vector::TVector3 position;
    Vector::TVector3 color;
};

struct THit {
    bool exists;
    double t;
};

struct TReflectedRay {
    Vector::TVector3 pos;
    Vector::TVector3 dir;
};

struct TPixelPosition {
    unsigned int x;
    unsigned int y;
};

struct TRay {
    Vector::TVector3 pos;
    Vector::TVector3 dir;
    Vector::TVector3 color;
    TPixelPosition pixelPos;
    int depth;
};

const double EPS = 1e-3;
const double EMBIENT_COEF = 0.1;
const double SPECULAR_COEF = 0.5;
const double DIFFUSE_COEF = 1.0;

Vector::TVector3 Reflect(Vector::TVector3 v, Vector::TVector3 normal) {
    double k = -2.0 * Vector::Dot(v, normal);
    Vector::TVector3 temp = Vector::Mult(k, normal);
    return Vector::Normalize(Vector::Add(temp, v));
}

THit CheckHitWithPolygon(Vector::TVector3 pos, Vector::TVector3 dir, Polygon::TPolygon polygon) {
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

    return THit { .exists = true, .t = t };
}

Vector::TVector3 GetPolygonPixelColor(Polygon::TPolygon polygon, Vector::TVector3 hitPos) {
    return polygon.color;
}

Vector::TVector3 GetColor(
    Vector::TVector3 hitPos,
    Vector::TVector3 dir,
    size_t polygonId,
    Polygon::TPolygon *polygons, size_t polygonsAmount,
    TLight *lights, size_t lightsAmount
) {
    Vector::TVector3 totalColor = { 0.0, 0.0, 0.0 };
    Polygon::TPolygon polygon = polygons[polygonId];

    if (polygon.isLightSource) {
        return polygon.color;
    }

    // embient light
    Vector::TVector3 embientColor = Vector::Mult(EMBIENT_COEF, polygon.color);

    for (size_t lightIndex = 0; lightIndex < lightsAmount; lightIndex++) {
        TLight light = lights[lightIndex];

        Vector::TVector3 lightPos = light.position;
        Vector::TVector3 lightColor = light.color;

        Vector::TVector3 lightDir = Vector::Normalize(Vector::Sub(hitPos, lightPos));
        Vector::TVector3 dirNormalized = Vector::Normalize(dir);
        Vector::TVector3 resultColor = lightColor;

        // shade coef
        double shadeCoef = 1.0;
        THit currHit = CheckHitWithPolygon(lightPos, lightDir, polygons[polygonId]);
        if (!currHit.exists) {
            std::cerr << "[error] unexpected not existent hit\n";
            exit(1);
        }

        for (size_t i = 0; i < polygonsAmount; ++i) {
            if (i == polygonId) continue;
            THit hit = CheckHitWithPolygon(lightPos, lightDir, polygons[i]);

            if ((!polygons[i].isLightSource) && hit.exists && (hit.t < currHit.t)) {
                shadeCoef *= polygons[i].transparent;
            }
        }

        // diffuse light
        Vector::TVector3 l = Vector::Mult(-1.0, lightDir);
        Vector::TVector3 n = Polygon::GetNormal(polygon);
        double diffuseAngle = std::max(0.0, Vector::Dot(n, l));
        double diffuseCoef = DIFFUSE_COEF * diffuseAngle;

        // specular light
        Vector::TVector3 reflectedLightDirection = Vector::Normalize(Reflect(
            Vector::Sub(hitPos, lightPos),
            Polygon::GetNormal(polygon)
        ));
        double specularAngle = std::max(0.0, Vector::Dot(reflectedLightDirection, dirNormalized));
        if (specularAngle >= 1.0) std::cout << specularAngle << std::endl;
        double specularCoef = polygon.blend * SPECULAR_COEF * std::pow(specularAngle, 9);

        // total color
        Vector::TVector3 color = Vector::Mult(shadeCoef * (diffuseCoef + specularCoef), resultColor);
        totalColor = Vector::Add(totalColor, color);
    }

    totalColor = Vector::Add(totalColor, embientColor);
    totalColor = Vector::Mult(totalColor, GetPolygonPixelColor(polygon, hitPos));

    totalColor = {
        std::min(1.0, std::max(0.0, totalColor.x)),
        std::min(1.0, std::max(0.0, totalColor.y)),
        std::min(1.0, std::max(0.0, totalColor.z))
    };

    return totalColor;
}

TReflectedRay GetReflectedRay(Vector::TVector3 pos, Vector::TVector3 dir, Polygon::TPolygon polygon, Vector::TVector3 hitPosition) {
    Vector::TVector3 n = Polygon::GetNormal(polygon);

    Vector::TVector3 nextDir = Reflect(dir, n);
    Vector::TVector3 nextPos = Vector::Add(hitPosition, Vector::Mult(EPS, nextDir));

    return { .pos = nextPos, .dir = nextDir };
}

Vector::TVector3 ray(
    TRay ray,
    Polygon::TPolygon *polygons, size_t polygonsAmount,
    TLight *lights, size_t lightsAmount,
    TRay *nextRays, size_t *cursor
) { 
    if (ray.depth > 3) {
        return { 0.0, 0.0, 0.0 };
    }

    int k_min = -1;
    double ts_min;

    for(unsigned int k = 0; k < polygonsAmount; k++) {
        THit hit = CheckHitWithPolygon(ray.pos, ray.dir, polygons[k]);
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


    Polygon::TPolygon hitPolygon = polygons[k_min];
    Vector::TVector3 hitPosition = Vector::Add(ray.pos, Vector::Mult(ts_min, ray.dir));
    Vector::TVector3 hitColor = GetColor(hitPosition, ray.dir, k_min, polygons, polygonsAmount, lights, lightsAmount);
    Vector::TVector3 resultColor = Vector::Mult(ray.color, hitColor);

    if (hitPolygon.reflection > 0.0) {
        TReflectedRay nextRay = GetReflectedRay(ray.pos, ray.dir, hitPolygon, hitPosition);

        nextRays[*cursor] = {
            .pos = nextRay.pos,
            .dir = nextRay.dir,
            .color = Vector::Mult(hitPolygon.reflection, hitColor),
            .pixelPos = ray.pixelPos,
            .depth = ray.depth + 1
        };
        *cursor = *cursor + 1;
    }

    if (hitPolygon.transparent > 0.0) {
        Vector::TVector3 refractedDir = ray.dir;
        Vector::TVector3 refractedPos = Vector::Add(hitPosition, Vector::Mult(EPS, refractedDir));

        nextRays[*cursor] = {
            .pos = refractedPos,
            .dir = refractedDir,
            .color = Vector::Mult(hitPolygon.transparent, hitColor),
            .pixelPos = ray.pixelPos,
            .depth = ray.depth + 1
        };
        *cursor = *cursor + 1;
    }

    return resultColor;
}

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
    TRay *rays1 = (TRay *) malloc(initialRayCount * sizeof(TRay));
    TRay *rays2 = (TRay *) malloc(2 * initialRayCount * sizeof(TRay)); // Предполагаем, что потребуется в 2 раза больше
    size_t rays1Count = 0;
    size_t rays2Count = 0;

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

    for (int i = 0;; i = (i + 1) % 2) {
        TRay *current = (i % 2 == 0) ? rays1 : rays2;
        size_t &currentCount = (i % 2 == 0) ? rays1Count : rays2Count;
        TRay *next = (i % 2 == 0) ? rays2 : rays1;
        size_t &nextCount = (i % 2 == 0) ? rays2Count : rays1Count;

        if (currentCount == 0) {
            break;
        }

        nextCount = 0;

        for (size_t j = 0; j < currentCount; ++j) {
            TRay el = current[j];
            Canvas::TColor color = VectorToColor(ray(el, polygons.data(), polygons.size(), lights.data(), lights.size(), next, &nextCount));
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
    }

    free(rays1);
    free(rays2);
}

void CpuDraw(Canvas::TCanvas canvas) {
    for (unsigned int x = 0; x < canvas.width; ++x) {
        for (unsigned int y = 0; y < canvas.height; ++y) {
            Canvas::PutPixel(&canvas, Canvas::TPosition { x, y }, Canvas::TColor { 255, 0, 0, 255 });
        }
    }
}


int main(int argc, char *argv[]) {
    std::string deviceType = std::string(argv[1]);    
    Canvas::TCanvas canvas;

    Canvas::Init(&canvas, 200, 200, (deviceType == "gpu") ? Canvas::DeviceType::GPU : Canvas::DeviceType::CPU);

    std::vector<Polygon::TPolygon> polygons = {
        {
            .verticles = { { 0.0, 0.0, 0.0 }, { 0.0, 0.0, 3.0 }, { 3.0, 0.0, 3.0 } },
            .color = { 1.0, 0.0, 0.0 },
            .reflection = 0.0,
            .transparent = 0.0,
            .blend = 0.0,
            .isLightSource = false
        }
    };

    std::vector<TLight> lights;

    if (deviceType == "gpu") {
        std::cerr << "[log] using GPU render ..." << std::endl;
        GpuDraw<<<100, 100>>>(canvas);
        cudaDeviceSynchronize();
        SAVE_CUDA(cudaGetLastError());    
    } else if (deviceType == "debug") {
        std::cerr << "[log] using debug render ..." << std::endl;
        render(
            { 0.0, 6.0, 4.0 },
            { 0.0, -3.0, -1.0 },
            120.0,
            &canvas,
            polygons, lights);
    } else {
        std::cerr << "[log] using CPU render ..." << std::endl;
        CpuDraw(canvas);
    }

    Canvas::Dump(&canvas, "build/0.data");
    Canvas::Destroy(&canvas);

    return 0;
}
