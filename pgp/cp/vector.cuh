#ifndef _VECTOR_H_
#define _VECTOR_H_

#include <cmath>

namespace Vector {

    /* Types */

    struct TVector3 {
        double x;
        double y;
        double z;
    };

    /* Methods */

    __host__ __device__ double Dot(TVector3 a, TVector3 b) {
        return a.x * b.x + a.y * b.y + a.z * b.z;
    }

    __host__ __device__ double Length(TVector3 v) {
        return std::sqrt(Dot(v, v));
    }

    __host__ __device__ TVector3 Prod(TVector3 a, TVector3 b) {
        return {
            a.y * b.z - a.z * b.y,
            a.z * b.x - a.x * b.z,
            a.x * b.y - a.y * b.x
        };
    }
    
    __host__ __device__ TVector3 Normalize(TVector3 v) {
        double l = Length(v);
        return {v.x / l, v.y / l, v.z / l};
    }
    
    __host__ __device__ TVector3 Sub(TVector3 a, TVector3 b) {
        return {a.x - b.x, a.y - b.y, a.z - b.z};
    }
    
    __host__ __device__ TVector3 Add(TVector3 a, TVector3 b) {
        return {a.x + b.x, a.y + b.y, a.z + b.z};
    }
    
    __host__ __device__ TVector3 Mult(TVector3 a, TVector3 b, TVector3 c, TVector3 v) {
        return {
            a.x * v.x + b.x * v.y + c.x * v.z,
            a.y * v.x + b.y * v.y + c.y * v.z,
            a.z * v.x + b.z * v.y + c.z * v.z
        };
    }
    
    __host__ __device__ TVector3 Mult(TVector3 a, TVector3 b) {
        return { a.x * b.x, a.y * b.y, a.z * b.z };
    }
    
    __host__ __device__ TVector3 Mult(double coef, TVector3 v) {
        return { coef * v.x, coef * v.y, coef * v.z };
    }

}    

#endif // _VECTOR_H_
