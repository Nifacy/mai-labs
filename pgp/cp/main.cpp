// ffmpeg -framerate 30 -pattern_type glob -i '*.png' -c:v libx264 -pix_fmt yuv420p out.mp4

#include <stdlib.h>
#include <stdio.h> 
#include <cmath>
#include <vector>
#include <iostream>
#include <cstring>

#include "canvas/canvas.h"
#include "vector/vector.h"


unsigned int SCREEN_WIDTH = 640;
unsigned int SCREEN_HEIGHT = 480;



struct TLight {
    Vector::TVector3 position;
    Vector::TVector3 color;
};


struct THit {
    bool exists;
    double t;
};



namespace Texture {
    typedef unsigned char TPixel[4];

    struct TTexture {
        TPixel *data = nullptr;
        unsigned int width = 0;
        unsigned int height = 0;
    };

    void Load(const std::string &filepath, TTexture &out) {
        FILE *in = fopen(filepath.c_str(), "rb");
        if (!in) {
            throw std::runtime_error("Failed to open file '" + filepath + "'");
        }

        fread(&out.width, sizeof(unsigned int), 1, in);
        fread(&out.height, sizeof(unsigned int), 1, in);

        out.data = (TPixel*) malloc(sizeof(TPixel) * out.width * out.height);
        if (out.data == nullptr) {
            throw std::runtime_error("Failed to allocated data for texture");
        }

        fread(out.data, sizeof(TPixel), out.width * out.height, in);
        fclose(in);
    }

    void Destroy(TTexture &texture) {
        free(texture.data);

        texture.data = nullptr;
        texture.width = 0;
        texture.height = 0;
    }

    Vector::TVector3 GetPixel(const TTexture &texture, const std::pair<double, double> &pos) {
        TPixel pixel;
        unsigned int x, y;
        
        x = (unsigned int) (std::max(0.0, pos.first) * (texture.width - 1));
        x = std::min(texture.width - 1, x);

        y = (unsigned int) (std::max(0.0, pos.second) * (texture.height - 1));
        y = std::min(texture.height - 1, y);

        std::memcpy(pixel, texture.data[(texture.height - y - 1) * texture.width + x], sizeof(TPixel));
        return { pixel[0] / 255.0, pixel[1] / 255.0, pixel[2] / 255.0 };
    }
}


namespace TextureProjection {
    struct TTextureProjection {
        const Texture::TTexture &src;
        Vector::TVector3 verticles[3];
    };

    Vector::TVector3 GetPixel(const TTextureProjection &projection, const std::pair<double, double> &pos) {
        double a1 = pos.first, a2 = pos.second;
        double a3 = 1.0 - a1 - a2;

        if (a1 < 0.0 || a2 < 0.0 || a3 < 0.0) {
            std::string dumpedCoords = "(" + std::to_string(a1) + ", " + std::to_string(a2) + ", " + std::to_string(a3) + ")";
            throw std::runtime_error("Invalid coordinates for projection: " + dumpedCoords);
        }

        Vector::TVector3 p = {0.0, 0.0, 0.0};
        p = Vector::Add(p, Vector::Mult(a1, projection.verticles[0]));
        p = Vector::Add(p, Vector::Mult(a2, projection.verticles[1]));
        p = Vector::Add(p, Vector::Mult(a3, projection.verticles[2]));

        return Texture::GetPixel(projection.src, { p.y, p.x });
    }
}


struct TPolygon {
    Vector::TVector3 verticles[3];
    Vector::TVector3 color;
    double reflection = 0.0;
    double transparent = 0.0;
    const TextureProjection::TTextureProjection *texture = nullptr;
};


struct TCubeConfig {
    Vector::TVector3 pos;
    Vector::TVector3 color;
    double size;
    double reflection = 0.0;
    double transparent = 0.0;
    const Texture::TTexture *texture = nullptr;
};


const double EPS = 1e-3;


void buildCube(
    const TCubeConfig &config,
    std::vector<TPolygon> &out
) {
    double x = config.pos.x;
    double y = config.pos.y;
    double z = config.pos.z;

    TextureProjection::TTextureProjection *projs[2] = { nullptr, nullptr };

    if (config.texture != nullptr) {
        projs[0] = new TextureProjection::TTextureProjection {
            .src = *config.texture,
            .verticles = { { 0.0, 0.0, 0.0 }, { 0.0, 1.0, 0.0 }, { 1.0, 1.0, 0.0 } }
        };

        projs[1] = new TextureProjection::TTextureProjection {
            .src = *config.texture,
            .verticles = { { 0.0, 1.0, 0.0 }, { 1.0, 0.0, 0.0 }, { 1.0, 1.0, 0.0 } }
        };
    }

    // top
    double c = config.size;

    out.push_back({
        .verticles = {{x - c, y - c, z + c}, {x + c, y - c, z + c}, {x - c, y + c, z + c}},
        .color = config.color,
        .reflection = config.reflection,
        .transparent= config.transparent,
        .texture = projs[0]
    });

    out.push_back({
        .verticles = {{x + c, y - c, z + c}, {x + c, y + c, z + c}, {x - c, y + c, z + c}},
        .color = config.color,
        .reflection = config.reflection,
        .transparent= config.transparent,
        .texture = projs[1]
    });

    // right
    out.push_back({
        .verticles = {{x + c, y - c, z - c}, {x + c, y + c, z - c}, {x + c, y + c, z + c}},
        .color = config.color,
        .reflection = config.reflection,
        .transparent= config.transparent,
        .texture = projs[0]
    });

    out.push_back({
        .verticles = {{x + c, y - c, z - c}, {x + c, y - c, z + c}, {x + c, y + c, z + c}},
        .color = config.color,
        .reflection = config.reflection,
        .transparent= config.transparent,
        .texture = projs[1]
    });

    // left
    out.push_back({
        .verticles = {{x - c, y - c, z - c}, {x - c, y + c, z - c}, {x - c, y + c, z + c}},
        .color = config.color,
        .reflection = config.reflection,
        .transparent= config.transparent,
        .texture = projs[0]
    });

    out.push_back({
        .verticles = {{x - c, y - c, z - c}, {x - c, y - c, z + c}, {x - c, y + c, z + c}},
        .color = config.color,
        .reflection = config.reflection,
        .transparent= config.transparent,
        .texture = projs[1]
    });

    // bottom
    out.push_back({
        .verticles = {{x - c, y - c, z - c}, {x + c, y - c, z - c}, {x - c, y + c, z - c}},
        .color = config.color,
        .reflection = config.reflection,
        .transparent= config.transparent,
        .texture = projs[0]
    });

    out.push_back({
        .verticles = {{x + c, y + c, z - c}, {x + c, y - c, z - c}, {x - c, y + c, z - c}},
        .color = config.color,
        .reflection = config.reflection,
        .transparent= config.transparent,
        .texture = projs[1]
    });

    // back
    out.push_back({
        .verticles = {{x - c, y - c, z - c}, {x - c, y - c, z + c}, {x + c, y - c, z + c}},
        .color = config.color,
        .reflection = config.reflection,
        .transparent= config.transparent,
        .texture = projs[0]
    });

    out.push_back({
        .verticles = {{x - c, y - c, z - c}, {x + c, y - c, z - c}, {x + c, y - c, z + c}},
        .color = config.color,
        .reflection = config.reflection,
        .transparent= config.transparent,
        .texture = projs[1]
    });

    // front
    out.push_back({
        .verticles = {{x - c, y + c, z - c}, {x - c, y + c, z + c}, {x + c, y + c, z + c}},
        .color = config.color,
        .reflection = config.reflection,
        .transparent= config.transparent,
        .texture = projs[0]
    });

    out.push_back({
        .verticles = {{x - c, y + c, z - c}, {x + c, y + c, z - c}, {x + c, y + c, z + c}},
        .color = config.color,
        .reflection = config.reflection,
        .transparent= config.transparent,
        .texture = projs[1]
    });
}


void build_space(std::vector<TPolygon> &out, const Texture::TTexture *texture) {
    // buildCube({ 0.0, -3.0, 3.0 }, { 1.0, 0.0, 0.0 }, 2.0, 0.0, 0.0, out);
    TextureProjection::TTextureProjection *proj = nullptr;
    TextureProjection::TTextureProjection *proj2 = nullptr;
    
    if (texture != nullptr) {
        proj = new TextureProjection::TTextureProjection {
            .src = *texture,
            .verticles = { { 1.0, 0.0, 0.0 }, { 0.0, 1.0, 0.0 }, { 1.0, 1.0, 0.0 } }
        };

        proj2 = new TextureProjection::TTextureProjection {
            .src = *texture,
            .verticles = { { 1.0, 0.0, 0.0 }, { 0.0, 1.0, 0.0 }, { 0.0, 0.0, 0.0 } }
        };
    }

    out.push_back({
        .verticles = { { 0.0, 0.0, 0.0 }, { 0.0, 0.0, 3.0 }, { 3.0, 0.0, 3.0 } },
        .color = { 1.0, 0.0, 0.0 },
        .texture = proj
    });

    out.push_back({
        .verticles = { { 0.0, 0.0, 0.0 }, { 3.0, 0.0, 0.0 }, { 3.0, 0.0, 3.0 } },
        .color = { 1.0, 0.0, 0.0 },
        .texture = proj2
    });

    // buildCube(
    //     {
    //         .pos = { 0.0, -5.0, 3.0 },
    //         .color = { 0.6, 0.6, 0.6 },
    //         .size = 2.0,
    //         .reflection = 1.0,
    //         .transparent = 0.0
    //     },
    //     out
    // );

    // buildCube(
    //     {
    //         .pos = { 0.0, 5.0, 3.0 },
    //         .color = { 0.0, 1.0, 0.0 },
    //         .size = 2.0,
    //         .reflection = 0.0,
    //         .transparent = 0.0,
    //         .texture = texture
    //     },
    //     out
    // );

    // buildCube({ 0.0, 5.0, 0.0 }, { 0.0, 1.0, 0.0 }, 2.0, 0.5, 0.0, out);
    // buildCube(
    //     {
    //         .pos = { 0.0, 0.0, -8.0 },
    //         .color = { 1.0, 1.0, 1.0 },
    //         .size = 8.0,
    //         .reflection = 0.0,
    //         .transparent = 0.0,
    //         .texture = texture
    //     },
    //     out
    // );
}


const double EMBIENT_COEF = 0.2;
const double SPECULAR_COEF = 1.0;
const double DIFFUSE_COEF = 0.4;


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



Vector::TVector3 GetPolygonPixelColor(const TPolygon &polygon, const Vector::TVector3 &hitPos) {
    if (polygon.texture == nullptr) {
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

    double a1 = A1 / A;
    double a2 = A2 / A;

    return TextureProjection::GetPixel(*polygon.texture, { a1, a2 });
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
        Vector::TVector3 resultColor = Vector::Mult(GetPolygonPixelColor(polygon, hitPos), lightColor);

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
        double specularCoef = SPECULAR_COEF * std::pow(specularAngle, 9);

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

    Vector::TVector3 resultColor = { 1.0, 1.0, 1.0 };

    TPolygon hitPolygon = polygons[k_min];
    Vector::TVector3 hitPosition = Vector::Add(pos, Vector::Mult(ts_min, dir));
    Vector::TVector3 hitColor = GetColor(hitPosition, dir, k_min, polygons, lights);

    if (hitPolygon.reflection > 0.0) {
        std::pair<Vector::TVector3, Vector::TVector3> nextRay = GetReflectedRay(pos, dir, hitPolygon, hitPosition);
        Vector::TVector3 reflectedColor = ray(nextRay.first, nextRay.second, polygons, lights, depth + 1);

        resultColor = Vector::Add(resultColor, Vector::Mult(hitPolygon.reflection, reflectedColor));
    }

    if (hitPolygon.transparent > 0.0) {
        Vector::TVector3 refractedDir = dir;
        Vector::TVector3 refractedPos = Vector::Add(hitPosition, Vector::Mult(EPS, refractedDir));
        Vector::TVector3 refractedColor = ray(refractedPos, refractedDir, polygons, lights, depth + 1);

        resultColor = Vector::Add(resultColor, Vector::Mult(hitPolygon.transparent, refractedColor));
    }

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

    Texture::TTexture texture;
    Texture::Load("textures/zamay.data", texture);
    build_space(polygons, &texture);

    std::cout << "Loaded texture: (" << texture.width << ", " << texture.height << ")\n";

    std::vector<TLight> lights = {
        { .position = { 0.0, -6.0, 7.0 }, .color = { 0.5, 0.5, 0.5 } },
        { .position = { 0.0, 6.0, 7.0 }, .color = { 0.5, 0.5, 0.5 } }
    };

    for(unsigned int k = 0; k < 100; k += 10) { 
        // cameraPos = { -6.0, 0.0, 7.0 };
        // pv = { 1.0, 0.0, -1.0 };

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

        cameraPos = (Vector::TVector3) {
			6.0 * sin(0.05 * k),
			6.0 * cos(0.05 * k),
			0.0
		}; // in scalar coords

        pv = (Vector::TVector3) {
			3.0 * sin(0.05 * k + M_PI),
			3.0 * cos(0.05 * k + M_PI),
			0.0
		};

        render(cameraPos, pv, 120.0, &canvas, polygons, lights);
    
        sprintf(buff, "build/%03d.data", k);
        printf("%d: %s\n", k, buff);    

        Canvas::Dump(&canvas, buff);
    }

    Canvas::Destroy(&canvas);
    Texture::Destroy(texture);
    return 0;
}
