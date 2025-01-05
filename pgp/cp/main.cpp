// ffmpeg -framerate 30 -pattern_type glob -i '*.png' -c:v libx264 -pix_fmt yuv420p out.mp4

#include <stdlib.h>
#include <stdio.h> 
#include <cmath>
#include <vector>
#include <iostream>
#include <cstring>
#include <fstream>
#include <sstream>

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
    double blend = 0.0;
    const TextureProjection::TTextureProjection *texture = nullptr;
    bool isLightSource = false;
};


struct TCubeConfig {
    Vector::TVector3 pos;
    Vector::TVector3 color;
    double size;
    double reflection = 0.0;
    double transparent = 0.0;
    double blend = 0.0;
    const Texture::TTexture *texture = nullptr;
};


const double EPS = 1e-3;



struct TFace {
    Vector::TVector3 vertices[3];
};


std::vector<TFace> LoadCubeMesh() {
    std::vector<Vector::TVector3> vertices = {
        {-1.0, -1.0, 1.0},
        {1.0, -1.0, 1.0},
        {-1.0, 1.0, 1.0},
        {1.0, 1.0, 1.0},
        {1.0, -1.0, -1.0},
        {1.0, 1.0, -1.0},
        {-1.0, 1.0, -1.0},
        {-1.0, -1.0, -1.0}
    };

    std::vector<std::vector<int>> faces = {
        {0, 1, 2}, {1, 3, 2}, // top
        {4, 5, 3}, {1, 4, 3},
        {6, 7, 2}, {7, 0, 2},
        {4, 7, 6}, {5, 4, 6},
        {0, 7, 1}, {7, 4, 1},
        {6, 2, 3}, {5, 6, 3}
    };

    std::vector<TFace> out;

    for (const std::vector<int> &verticeIds : faces) {
        out.push_back({ .vertices = { vertices[verticeIds[0]], vertices[verticeIds[1]], vertices[verticeIds[2]] } });
    }

    return out;
}


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
        .blend = config.blend,
        .texture = projs[0]
    });

    out.push_back({
        .verticles = {{x + c, y - c, z + c}, {x + c, y + c, z + c}, {x - c, y + c, z + c}},
        .color = config.color,
        .reflection = config.reflection,
        .transparent= config.transparent,
        .blend = config.blend,
        .texture = projs[1]
    });

    // right
    out.push_back({
        .verticles = {{x + c, y - c, z - c}, {x + c, y + c, z - c}, {x + c, y + c, z + c}},
        .color = config.color,
        .reflection = config.reflection,
        .transparent= config.transparent,
        .blend = config.blend,
        .texture = projs[0]
    });

    out.push_back({
        .verticles = {{x + c, y - c, z + c}, {x + c, y - c, z - c}, {x + c, y + c, z + c}},
        .color = config.color,
        .reflection = config.reflection,
        .transparent= config.transparent,
        .blend = config.blend,
        .texture = projs[1]
    });

    // left
    out.push_back({
        .verticles = {{x - c, y + c, z - c}, {x - c, y - c, z - c}, {x - c, y + c, z + c}},
        .color = config.color,
        .reflection = config.reflection,
        .transparent= config.transparent,
        .blend = config.blend,
        .texture = projs[0]
    });

    out.push_back({
        .verticles = {{x - c, y - c, z - c}, {x - c, y - c, z + c}, {x - c, y + c, z + c}},
        .color = config.color,
        .reflection = config.reflection,
        .transparent= config.transparent,
        .blend = config.blend,
        .texture = projs[1]
    });

    // bottom
    out.push_back({
        .verticles = {{x + c, y - c, z - c}, {x - c, y - c, z - c}, {x - c, y + c, z - c}},
        .color = {1, 1, 1},
        .reflection = config.reflection,
        .transparent= config.transparent,
        .blend = config.blend,
        .texture = projs[0]
    });

    out.push_back({
        .verticles = {{x + c, y + c, z - c}, {x + c, y - c, z - c}, {x - c, y + c, z - c}},
        .color = {1, 1, 1},
        .reflection = config.reflection,
        .transparent= config.transparent,
        .blend = config.blend,
        .texture = projs[1]
    });

    // back
    out.push_back({
        .verticles = {{x - c, y - c, z + c}, {x - c, y - c, z - c}, {x + c, y - c, z + c}},
        .color = config.color,
        .reflection = config.reflection,
        .transparent= config.transparent,
        .blend = config.blend,
        .texture = projs[0]
    });

    out.push_back({
        .verticles = {{x - c, y - c, z - c}, {x + c, y - c, z - c}, {x + c, y - c, z + c}},
        .color = config.color,
        .reflection = config.reflection,
        .transparent= config.transparent,
        .blend = config.blend,
        .texture = projs[1]
    });

    // front
    out.push_back({
        .verticles = {{x - c, y + c, z - c}, {x - c, y + c, z + c}, {x + c, y + c, z + c}},
        .color = config.color,
        .reflection = config.reflection,
        .transparent= config.transparent,
        .blend = config.blend,
        .texture = projs[0]
    });

    out.push_back({
        .verticles = {{x + c, y + c, z - c}, {x - c, y + c, z - c}, {x + c, y + c, z + c}},
        .color = config.color,
        .reflection = config.reflection,
        .transparent= config.transparent,
        .blend = config.blend,
        .texture = projs[1]
    });
}


std::vector<TFace> LoadMesh(const std::string &filepath) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        throw std::runtime_error("Unable to open file: " + filepath);
    }

    std::vector<Vector::TVector3> vertices;
    std::vector<TFace> faces;

    std::string line;
    while (std::getline(file, line)) {
        std::istringstream iss(line);

        char type;
        iss >> type;

        if (type == 'v') {
            double x, y, z;
            iss >> x >> y >> z;
            vertices.push_back({x, y, z});
        } else if (type == 'f') {
            int v1, v2, v3;
            iss >> v1 >> v2 >> v3;

            if (v1 <= 0 || v2 <= 0 || v3 <= 0 ||  v1 > vertices.size() || v2 > vertices.size() || v3 > vertices.size()) {
                throw std::runtime_error("Invalid face indices in file: " + filepath);
            }

            faces.push_back({.vertices = { vertices[v1 - 1], vertices[v2 - 1], vertices[v3 - 1] }});
        } else if (type == '#') {
            continue;
        }
    }

    std::cerr << "[log] read " << vertices.size() << " vertices\n";

    file.close();
    return faces;
}


struct TModelConfig {
    Vector::TVector3 pos = { 0.0, 0.0, 0.0 };
    Vector::TVector3 color = { 1.0, 1.0, 1.0 };
    double scale = 1.0;
    double reflection = 0.0;
    double transparent = 0.0;
    double blend = 0.0;
    bool isLightSource = false;
};


void BuildModel(const TModelConfig &config, const std::vector<TFace> &mesh, std::vector<TPolygon> &out) {
    for (const TFace &face : mesh) {
        std::vector<Vector::TVector3> v;

        for (int i = 0; i < 3; ++i) {
            v.push_back(Vector::Add(config.pos, Vector::Mult(config.scale, face.vertices[i])));
        }

        out.push_back({
            .verticles = { v[0], v[1], v[2] },
            .color = config.color,
            .reflection = config.reflection,
            .transparent = config.transparent,
            .blend = config.blend,
            .isLightSource = config.isLightSource
        });
    }
}


Vector::TVector3 GetPolygonNormal(const TPolygon &polygon) {
    Vector::TVector3 v1 = Vector::Sub(polygon.verticles[1], polygon.verticles[0]);
    Vector::TVector3 v2 = Vector::Sub(polygon.verticles[2], polygon.verticles[0]);
    return Vector::Normalize(Vector::Prod(v1, v2));
}
// { .verticles = { cubeMesh[0].vertices[0], cubeMesh[0].vertices[1], cubeMesh[0].vertices[2] } },
//         { .verticles = { cubeMesh[5].vertices[0], cubeMesh[5].vertices[1], cubeMesh[5].vertices[2] } },

void BuildLampLine(const TModelConfig &modelConfig, const std::vector<TFace> &mesh, int n, std::vector<TPolygon> &out) {
    for (int t1 = 0; t1 < mesh.size(); ++t1) {
        for (int t2 = t1 + 1; t2 < mesh.size(); ++t2) {
            TPolygon a = { .verticles = { mesh[t1].vertices[0], mesh[t1].vertices[1], mesh[t1].vertices[2] } };
            TPolygon b = { .verticles = { mesh[t2].vertices[0], mesh[t2].vertices[1], mesh[t2].vertices[2] } };

            std::vector<Vector::TVector3> points;
            for (int i = 0; i < 3; ++i) {
                for (int j = 0; j < 3; ++j) {
                    if (a.verticles[i].x == b.verticles[j].x && a.verticles[i].y == b.verticles[j].y && a.verticles[i].z == b.verticles[j].z) {
                        points.push_back(a.verticles[i]);
                    }
                }
            }

            if (points.size() < 2) {
                continue;
            }

            Vector::TVector3 na = Vector::Mult(-1.0, GetPolygonNormal(a));
            Vector::TVector3 nb = Vector::Mult(-1.0, GetPolygonNormal(b));

            if (std::abs(na.x - nb.x) < 0.001 && std::abs(na.y - nb.y) < 0.001 && std::abs(na.z - nb.z) < 0.001) {
                continue;
            }

            Vector::TVector3 v = Vector::Sub(points.at(1), points.at(0));
            Vector::TVector3 d = Vector::Normalize(Vector::Add(na, nb));
            Vector::TVector3 s = Vector::Normalize(Vector::Prod(d, v));

            Vector::TVector3 diff1 = Vector::Mult(0.1, Vector::Normalize(Vector::Add(d, s)));
            Vector::TVector3 diff2 = Vector::Mult(0.1, Vector::Normalize(Vector::Sub(d, s)));

            std::vector<Vector::TVector3> vertices = {
                Vector::Add(points[0], diff1),
                Vector::Add(points[0], diff2),
                Vector::Add(points[1], diff1),
                Vector::Add(points[1], diff2)
            };

            std::vector<TFace> faces = {
                { { vertices[0], vertices[1], vertices[2] } },
                { { vertices[3], vertices[2], vertices[1] } }
            };

            BuildModel({ .pos = modelConfig.pos, .color = { 0.1, 0.1, 0.1 }, .scale = modelConfig.scale, .isLightSource = true }, faces, out);

            Vector::TVector3 e1 = Vector::Mult(0.05, Vector::Normalize(v));
            Vector::TVector3 e2 = Vector::Mult(0.05, Vector::Normalize(s));
            Vector::TVector3 e3 = Vector::Mult(0.1, Vector::Normalize(d));

            for(int k = 0; k < n; k++) {
                double t = 1.0 / (n + 1) * (k + 1);
                Vector::TVector3 lampPos = Vector::Add(points[0], Vector::Mult(t, v));

                std::vector<Vector::TVector3> lampVertices = {
                    Vector::Add(Vector::Add(e2, Vector::Add(e1, e3)), lampPos),
                    Vector::Add(Vector::Sub(Vector::Add(e1, e3), e2), lampPos),
                    Vector::Add(Vector::Add(e2, Vector::Sub(e3, e1)), lampPos),
                    Vector::Add(Vector::Sub(Vector::Sub(e3, e1), e2), lampPos)
                };

                std::vector<TFace> lampFaces = {
                    { { lampVertices[1], lampVertices[0], lampVertices[2] } },
                    { { lampVertices[2], lampVertices[3], lampVertices[1] } }
                };

                BuildModel({ .pos = modelConfig.pos, .color = { 1.0, 1.0, 1.0 }, .scale = modelConfig.scale, .transparent = 1.0, .isLightSource = true }, lampFaces, out);
            }

        }
    }
}


void build_space(std::vector<TPolygon> &out) {
    // floor
    // Texture::TTexture *floorTexture = new Texture::TTexture {};
    // Texture::Load("textures/floor.data", *floorTexture);

    // TextureProjection::TTextureProjection *floorProj1 = new TextureProjection::TTextureProjection {
    //     .src = *floorTexture,
    //     .verticles = { { 0.0, 0.0, 0.0 }, { 0.0, 1.0, 0.0 }, { 1.0, 0.0, 0.0 } }
    // };

    // TextureProjection::TTextureProjection *floorProj2 = new TextureProjection::TTextureProjection {
    //     .src = *floorTexture,
    //     .verticles = { { 1.0, 1.0, 0.0 }, { 0.0, 1.0, 0.0 }, { 1.0, 0.0, 0.0 } }
    // };

    // out.push_back({
    //     .verticles = { { -5.0, -5.0, 0.0 }, { 5.0, -5.0, 0.0 }, { -5.0, 5.0, 0.0 } },
    //     .color = { 1.0, 1.0, 1.0 },
    //     .texture = floorProj1
    // });

    // out.push_back({
    //     .verticles = { { -5.0, 5.0, 0.0 }, { 5.0, -5.0, 0.0 }, { 5.0, 5.0, 0.0 } },
    //     .color = { 1.0, 1.0, 1.0 },
    //     .texture = floorProj2
    // });

    // cube
    std::vector<TFace> cubeMesh = LoadCubeMesh();
    TModelConfig cubeConfig =         {
            .pos = { 0.0, 0.0, 0.5 + 2.0 * EPS },
            .color = { 0.2, 1.0, 0.2 },
            .scale = 3.0,
            .reflection = 1.0,
            .transparent = 1.0,
            .blend = 1.0
        };

    BuildLampLine(
        cubeConfig,
        cubeMesh,
        2,
        out
    );

    // buildCube(
    //     {
    //         .pos = { 0.0, 0.0, 1.0 + 2.0 * EPS },
    //         .color = { 1.0, 0.5, 0.5 },
    //         .size = 1.0,
    //         .reflection = 1.0,
    //         .transparent = 0.0,
    //         .blend = 0.0
    //     },
    //     out
    // );

    BuildModel(
        cubeConfig,
        cubeMesh,
        out
    );

    // std::vector<TFace> mesh = LoadMesh("objects/model2.obj");

    // BuildLampLine(
    //     cubeConfig,
    //     mesh,
    //     2,
    //     out
    // );

    // BuildModel(cubeConfig, mesh, out);
}


const double EMBIENT_COEF = 0.1;
const double SPECULAR_COEF = 0.5;
const double DIFFUSE_COEF = 1.0;


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
    const TPolygon &polygon = polygons[polygonId];

    if (polygon.isLightSource) {
        return polygon.color;
    }

    // embient light
    Vector::TVector3 embientColor = Vector::Mult(EMBIENT_COEF, polygon.color);

    for (const TLight &light : lights) {
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

        for (size_t i = 0; i < polygons.size(); ++i) {
            if (i == polygonId) continue;
            THit hit = CheckHitWithPolygon(lightPos, lightDir, polygons[i]);

            if ((!polygons[i].isLightSource) && hit.exists && (hit.t < currHit.t)) {
                shadeCoef *= polygons[i].transparent;
            }
        }

        // diffuse light
        Vector::TVector3 l = Vector::Mult(-1.0, lightDir);
        Vector::TVector3 n = GetPolygonNormal(polygon);
        double diffuseAngle = std::max(0.0, Vector::Dot(n, l));
        double diffuseCoef = DIFFUSE_COEF * diffuseAngle;

        // specular light
        Vector::TVector3 reflectedLightDirection = Vector::Normalize(Reflect(
            Vector::Sub(hitPos, lightPos),
            GetPolygonNormal(polygon)
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


std::pair<Vector::TVector3, Vector::TVector3> GetReflectedRay(const Vector::TVector3 &pos, const Vector::TVector3 &dir, const TPolygon &polygon, const Vector::TVector3 &hitPosition) {
    Vector::TVector3 n = GetPolygonNormal(polygon);

    Vector::TVector3 nextDir = Reflect(dir, n);
    Vector::TVector3 nextPos = Vector::Add(hitPosition, Vector::Mult(EPS, nextDir));

    return { nextPos, nextDir };
}

struct TRay {
    Vector::TVector3 pos;
    Vector::TVector3 dir;
    Vector::TVector3 color;
    std::pair<unsigned int, unsigned int> pixelPos;
    int depth;
};


Vector::TVector3 ray(
    TRay ray,
    const std::vector<TPolygon> &polygons,
    const std::vector<TLight> &lights,
    std::vector<TRay> &nextRays
) { 
    if (ray.depth > 3) {
        return { 0.0, 0.0, 0.0 };
    }

    int k_min = -1;
    double ts_min;

    for(unsigned int k = 0; k < polygons.size(); k++) {
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


    TPolygon hitPolygon = polygons[k_min];
    Vector::TVector3 hitPosition = Vector::Add(ray.pos, Vector::Mult(ts_min, ray.dir));
    Vector::TVector3 hitColor = GetColor(hitPosition, ray.dir, k_min, polygons, lights);
    Vector::TVector3 resultColor = Vector::Mult(ray.color, hitColor);

    if (hitPolygon.reflection > 0.0) {
        std::pair<Vector::TVector3, Vector::TVector3> nextRay = GetReflectedRay(ray.pos, ray.dir, hitPolygon, hitPosition);
        // Vector::TVector3 reflectedColor = ray(nextRay.first, nextRay.second, polygons, lights, depth + 1);

        TRay r2 = {
            .pos = nextRay.first,
            .dir = nextRay.second,
            .color = Vector::Mult(hitPolygon.reflection, hitColor),
            .pixelPos = ray.pixelPos,
            .depth = ray.depth + 1
        };
        nextRays.push_back(r2);

        // resultColor = Vector::Add(resultColor, Vector::Mult(hitColor, Vector::Mult(hitPolygon.reflection, reflectedColor)));
    }

    if (hitPolygon.transparent > 0.0) {
        Vector::TVector3 refractedDir = ray.dir;
        Vector::TVector3 refractedPos = Vector::Add(hitPosition, Vector::Mult(EPS, refractedDir));
        // Vector::TVector3 refractedColor = ray(refractedPos, refractedDir, polygons, lights, depth + 1);

        TRay r2 = {
            .pos = refractedPos,
            .dir = refractedDir,
            .color = Vector::Mult(hitPolygon.transparent, hitColor),
            .pixelPos = ray.pixelPos,
            .depth = ray.depth + 1
        };
        nextRays.push_back(r2);

        // resultColor = Vector::Add(resultColor, Vector::Mult(hitColor, Vector::Mult(hitPolygon.transparent, refractedColor)));
    }

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
void render(Vector::TVector3 pc, Vector::TVector3 pv, double angle, Canvas::TCanvas *canvas, const std::vector<TPolygon> &polygons, const std::vector<TLight> &lights) {
    double dw = 2.0 / (canvas->width - 1.0);
    double dh = 2.0 / (canvas->height - 1.0);
    double z = 1.0 / tan(angle * M_PI / 360.0);

    Vector::TVector3 bz = Vector::Normalize(Vector::Sub(pv, pc));
    Vector::TVector3 bx = Vector::Normalize(Vector::Prod(bz, {0.0, 0.0, 1.0}));
    Vector::TVector3 by = Vector::Normalize(Vector::Prod(bx, bz));

    std::vector<TRay> rays1, rays2;

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

            rays1.push_back(ray);
            Canvas::PutPixel(canvas, { i, canvas->height - 1 - j }, { 0, 0, 0 });
		}
	}

    for (int i = 0;; i = (i + 1) % 2) {
        std::vector<TRay> &first = (i % 2 == 0) ? rays1 : rays2;
        std::vector<TRay> &second = (i % 2 == 0) ? rays2 : rays1;

        if (first.size() == 0) {
            break;
        }

        for (TRay &el : first) {
            Canvas::TColor color = VectorToColor(ray(el, polygons, lights, second));
            Canvas::TColor canvasColor = Canvas::GetPixel(canvas, el.pixelPos);
            Canvas::TColor resultColor = {
                .r = (unsigned char) std::min(255, int(color.r) + int(canvasColor.r)),
                .g = (unsigned char) std::min(255, int(color.g) + int(canvasColor.g)),
                .b = (unsigned char) std::min(255, int(color.b) + int(canvasColor.b)),
                .a = (unsigned char) std::min(255, int(color.a) + int(canvasColor.a))
            };

            Canvas::PutPixel(canvas, el.pixelPos, resultColor);
        }

        first.clear();
    }

    // for(unsigned int i = 0; i < canvas->width; i++) {
    //     for(unsigned int j = 0; j < canvas->height; j++) {
    //         Vector::TVector3 v = {-1.0 + dw * i, (-1.0 + dh * j) * canvas->height / canvas->width, z};
    //         Vector::TVector3 dir = Vector::Mult(bx, by, bz, v);
	// 		Vector::TVector3 colorVector = ray(pc, Vector::Normalize(dir), polygons, lights, 0);
    //         Canvas::TColor color = VectorToColor(colorVector);

    //         Canvas::PutPixel(canvas, { i, canvas->height - 1 - j }, color);
	// 	}
	// }
}


/*  Debug Render */

struct TPoint {
    int x, y;
};

void DrawLine(Canvas::TCanvas *canvas, TPoint p1, TPoint p2, Canvas::TColor color) {
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
        if (x1 < 0 || x1 >= canvas->width || y1 < 0 || y1 >= canvas->height) {
        } else {
        // Рисуем текущую точку
        Canvas::PutPixel(canvas, {x1, y1}, color);
        Canvas::PutPixel(canvas, {x1 + 1, y1}, color);
        Canvas::PutPixel(canvas, {x1, y1 + 1}, color);
        Canvas::PutPixel(canvas, {x1 + 1, y1 + 1}, color);
        }

        // Если достигли конца линии, выходим из цикла
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


void DrawDot(Canvas::TCanvas *canvas, TPoint p1) {
    for (int x = p1.x - 1; x <= p1.x + 1; x++) {
        for (int y = p1.y - 1; y <= p1.y + 1; ++y) {
            if (x < 0 || x >= canvas->width || y < 0 || y >= canvas->height) {
                continue;
            }

            Canvas::PutPixel(canvas, { x, y }, { 0, 255, 0 });
        }
    }
}


TPoint WorldToCanvas(const Vector::TVector3 &v, 
                             const Vector::TVector3 &pc, 
                             const Vector::TVector3 &pv, 
                             double angle, 
                             Canvas::TCanvas *canvas) {
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


void DebugRender(Vector::TVector3 pc, Vector::TVector3 pv, double angle, Canvas::TCanvas *canvas, const std::vector<TPolygon> &polygons, const std::vector<TLight> &lights) {
    for (int x = 0; x < canvas->width; ++x) {
        for (int y = 0; y < canvas->height; ++y) {
            Canvas::PutPixel(canvas, { x, y }, {0, 0, 0});
        }
    }

    for (const auto &polygon : polygons) {
        size_t vertexCount = 3;
        Vector::TVector3 center = { 0.0, 0.0, 0.0 };

        for (size_t i = 0; i < vertexCount; i++) {
            // Определяем текущую вершину и следующую вершину (замыкаем полигон на последней грани)
            const Vector::TVector3 &v1 = polygon.verticles[i];
            const Vector::TVector3 &v2 = polygon.verticles[(i + 1) % vertexCount];

            // Преобразуем координаты из мира в экранные
            TPoint p1 = WorldToCanvas(v1, pc, pv, angle, canvas);
            TPoint p2 = WorldToCanvas(v2, pc, pv, angle, canvas);

            // Рисуем линию между текущей и следующей вершинами
            DrawLine(canvas, p1, p2, Canvas::TColor{255, 0, 0}); // Белый цвет для линий
            // DrawDot(canvas, p1);
            // DrawDot(canvas, p2);
            center = Vector::Add(center, Vector::Mult(0.33, v1));
        }

        const Vector::TVector3 n = Vector::Add(center, GetPolygonNormal(polygon));
        TPoint p1 = WorldToCanvas(center, pc, pv, angle, canvas);
        TPoint p2 = WorldToCanvas(n, pc, pv, angle, canvas);
        DrawLine(canvas, p1, p2, Canvas::TColor{0, 255, 0});
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
        // { .position = { 10.0, 0.0, 10.0 }, .color = { 1.0, 1.0, 1.0 } },
        // { .position = { -10.0, 0.0, 10.0 }, .color = { 1.0, 1.0, 1.0 } }
        { .position = { 5.0, 5.0, 5.0 }, .color = { 1.0, 1.0, 1.0 } }
    };

    for(unsigned int k = 0; k < 150; k += 10) { 
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
			4.0
		}; // in scalar coords

        pv = (Vector::TVector3) {
			3.0 * sin(0.05 * k + M_PI),
			3.0 * cos(0.05 * k + M_PI),
			-1.0
		};

        render(cameraPos, pv, 120.0, &canvas, polygons, lights);
    
        sprintf(buff, "build/%03d.data", k);
        printf("%d: %s\n", k, buff);    

        Canvas::Dump(&canvas, buff);
    }

    Canvas::Destroy(&canvas);
    return 0;
}
