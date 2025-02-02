#ifndef _TEXTURE_PROJECTION_H_
#define _TEXTURE_PROJECTION_H_

#include "texture.cuh"
#include "vector.cuh"

namespace TextureProjection {

    /* Types */

    struct TTextureProjection {
        Texture::TTexture src;
        Vector::TVector3 verticles[3];
    };

    /* Methods */

    __host__ __device__ Vector::TVector3 GetPixel(TTextureProjection *projection, Texture::TPosition pos) {
        double a1 = pos.x, a2 = pos.y;
        double a3 = 1.0 - a1 - a2;

        Vector::TVector3 p = {0.0, 0.0, 0.0};
        p = Vector::Add(p, Vector::Mult(a1, projection->verticles[0]));
        p = Vector::Add(p, Vector::Mult(a2, projection->verticles[1]));
        p = Vector::Add(p, Vector::Mult(a3, projection->verticles[2]));

        return Texture::GetPixel(&projection->src, { p.y, p.x });
    }
}

#endif // _TEXTURE_PROJECTION_H_
