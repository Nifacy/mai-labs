#ifndef _RENDER_H_
#define _RENDER_H_

#include "utils.cuh"
#include "texture_projection.cuh"
#include "vector.cuh"
#include "canvas.cuh"
#include "polygon.cuh"

namespace Renderer {

    /* Types */

    struct TLight {
        Vector::TVector3 position;
        Vector::TVector3 color;
    };

    struct THit {
        bool exists;
        double t;
    };

    struct TRay {
        Vector::TVector3 pos;
        Vector::TVector3 dir;
    };

    struct TRayTraceContext {
        TRay ray;
        Vector::TVector3 color;
        Canvas::TPosition pixelPos;
        int depth;
    };

    /* Constants */

    const double EPS = 1e-3;
    const double EMBIENT_COEF = 0.1;
    const double SPECULAR_COEF = 0.5;
    const double DIFFUSE_COEF = 1.0;

    /* Methods */

    __host__ __device__ Vector::TVector3 _Reflect(Vector::TVector3 v, Vector::TVector3 normal) {
        double k = -2.0 * Vector::Dot(v, normal);
        Vector::TVector3 temp = Vector::Mult(k, normal);
        return Vector::Normalize(Vector::Add(temp, v));
    }

    __host__ __device__ THit _CheckHitWithPolygon(Vector::TVector3 pos, Vector::TVector3 dir, Polygon::TPolygon polygon) {
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

    __host__ __device__ Vector::TVector3 _GetPolygonPixelColor(Polygon::TPolygon polygon, Vector::TVector3 hitPos) {
        if (!polygon.texture.enabled) {
            return polygon.color;
        }

        Vector::TVector3 v0 = polygon.verticles[0];
        Vector::TVector3 v1 = polygon.verticles[1];
        Vector::TVector3 v2 = polygon.verticles[2];

        Vector::TVector3 E1 = Vector::Sub(v1, v0);
        Vector::TVector3 E2 = Vector::Sub(v2, v0);
        Vector::TVector3 E3 = Vector::Sub(v0, hitPos);
        Vector::TVector3 E4 = Vector::Sub(v1, hitPos);
        Vector::TVector3 E5 = Vector::Sub(v2, hitPos);

        double A = 0.5 * Vector::Length(Vector::Prod(E1, E2));
        double A1 = 0.5 * Vector::Length(Vector::Prod(E3, E4));
        double A2 = 0.5 * Vector::Length(Vector::Prod(E4, E5));

        return Vector::Mult(
            polygon.color,
            TextureProjection::GetPixel(&polygon.texture.texture, { A1 / A, A2 / A })
        );
    }

    __host__ __device__ Vector::TVector3 _GetColor(
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
            THit currHit = _CheckHitWithPolygon(lightPos, lightDir, polygons[polygonId]);

            for (size_t i = 0; i < polygonsAmount; ++i) {
                if (i == polygonId) continue;
                THit hit = _CheckHitWithPolygon(lightPos, lightDir, polygons[i]);

                if ((!polygons[i].isLightSource) && hit.exists && (hit.t < currHit.t)) {
                    shadeCoef *= polygons[i].transparent;
                }
            }

            // diffuse light
            Vector::TVector3 l = Vector::Mult(-1.0, lightDir);
            Vector::TVector3 n = Polygon::GetNormal(polygon);
            double diffuseAngle = Max(0.0, Vector::Dot(n, l));
            double diffuseCoef = DIFFUSE_COEF * diffuseAngle;

            // specular light
            Vector::TVector3 reflectedLightDirection = Vector::Normalize(_Reflect(
                Vector::Sub(hitPos, lightPos),
                Polygon::GetNormal(polygon)
            ));
            double specularAngle = Max(0.0, Vector::Dot(reflectedLightDirection, dirNormalized));
            double specularCoef = polygon.blend * SPECULAR_COEF * powf(specularAngle, 9);

            // total color
            Vector::TVector3 color = Vector::Mult(shadeCoef * (diffuseCoef + specularCoef), resultColor);
            totalColor = Vector::Add(totalColor, color);
        }

        totalColor = Vector::Add(totalColor, embientColor);
        totalColor = Vector::Mult(totalColor, _GetPolygonPixelColor(polygon, hitPos));

        totalColor = {
            Min(1.0, Max(0.0, totalColor.x)),
            Min(1.0, Max(0.0, totalColor.y)),
            Min(1.0, Max(0.0, totalColor.z))
        };

        return totalColor;
    }

    __host__ __device__ TRay _GetReflectedRay(TRay ray, Polygon::TPolygon polygon, Vector::TVector3 hitPosition) {
        Vector::TVector3 n = Polygon::GetNormal(polygon);

        Vector::TVector3 nextDir = _Reflect(ray.dir, n);
        Vector::TVector3 nextPos = Vector::Add(hitPosition, Vector::Mult(EPS, nextDir));

        return { .pos = nextPos, .dir = nextDir };
    }

    __host__ __device__ Vector::TVector3 Ray(
        TRayTraceContext ray,
        Polygon::TPolygon *polygons, size_t polygonsAmount,
        TLight *lights, size_t lightsAmount,
        TRayTraceContext *nextRays, int *cursor,
        int maxRecursionDepth
    ) { 
        if (ray.depth > maxRecursionDepth) {
            return { 0.0, 0.0, 0.0 };
        }

        int k_min = -1;
        double ts_min;

        for(unsigned int k = 0; k < polygonsAmount; k++) {
            THit hit = _CheckHitWithPolygon(ray.ray.pos, ray.ray.dir, polygons[k]);
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
        Vector::TVector3 hitPosition = Vector::Add(ray.ray.pos, Vector::Mult(ts_min, ray.ray.dir));
        Vector::TVector3 hitColor = _GetColor(hitPosition, ray.ray.dir, k_min, polygons, polygonsAmount, lights, lightsAmount);
        Vector::TVector3 resultColor = Vector::Mult(ray.color, hitColor);

        if (hitPolygon.reflection > 0.0) {
            TRay nextRay = _GetReflectedRay(ray.ray, hitPolygon, hitPosition);

            #ifdef __CUDA_ARCH__
                int index = atomicAdd(cursor, 1);
            #endif

            #ifndef __CUDA_ARCH__
                *cursor = *cursor + 1;
                int index = *cursor;
            #endif

            nextRays[index - 1] = {
                .ray = nextRay,
                .color = Vector::Mult(hitPolygon.reflection, hitColor),
                .pixelPos = ray.pixelPos,
                .depth = ray.depth + 1
            };
        }

        if (hitPolygon.transparent > 0.0) {
            Vector::TVector3 refractedDir = ray.ray.dir;
            Vector::TVector3 refractedPos = Vector::Add(hitPosition, Vector::Mult(EPS, refractedDir));

            #ifdef __CUDA_ARCH__
                int index = atomicAdd(cursor, 1);
            #endif

            #ifndef __CUDA_ARCH__
                *cursor = *cursor + 1;
                int index = *cursor;
            #endif

            nextRays[index - 1] = {
                .ray = {
                    .pos = refractedPos,
                    .dir = refractedDir
                },
                .color = Vector::Mult(hitPolygon.transparent, hitColor),
                .pixelPos = ray.pixelPos,
                .depth = ray.depth + 1
            };
        }

        return resultColor;
    }

    __host__ __device__ Canvas::TColor VectorToColor(Vector::TVector3 v) {
        return {
            (unsigned char) Max(0, Min(int(v.x * 255.0), 255)),
            (unsigned char) Max(0, Min(int(v.y * 255.0), 255)),
            (unsigned char) Max(0, Min(int(v.z * 255.0), 255)),
            255
        };
    }

    __host__ __device__ Vector::TVector3 ColorToVector(Canvas::TColor color) {
        return Vector::Mult(
            1.0 / 255.0,
            {(double) color.r, (double) color.g, (double) color.b}
        );
    }
}

#endif // _RENDER_H_
