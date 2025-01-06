#ifndef _POLYGON_H_
#define _POLYGON_H_

#include "vector.cuh"

namespace Polygon {

    /* Types */

    struct TPolygon {
        Vector::TVector3 verticles[3];
        Vector::TVector3 color;
        double reflection;
        double transparent;
        double blend;
        bool isLightSource;
    };

    /* Methods */

    Vector::TVector3 GetNormal(TPolygon polygon) {
        Vector::TVector3 v1 = Vector::Sub(polygon.verticles[1], polygon.verticles[0]);
        Vector::TVector3 v2 = Vector::Sub(polygon.verticles[2], polygon.verticles[0]);
        return Vector::Normalize(Vector::Prod(v1, v2));
    }
}

#endif // _PLYGON_H_
