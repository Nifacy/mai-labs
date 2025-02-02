#ifndef _SCENE_H_
#define _SCENE_H_

#include <vector>
#include <fstream>
#include <sstream>

#include "texture.cuh"
#include "texture_projection.cuh"
#include "polygon.cuh"
#include "vector.cuh"


struct TCubeConfig {
    Vector::TVector3 pos;
    Vector::TVector3 color;
    double size;
    double reflection = 0.0;
    double transparent = 0.0;
    double blend = 0.0;
};


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
        out.push_back(TFace { .vertices = { vertices[verticeIds[0]], vertices[verticeIds[1]], vertices[verticeIds[2]] } });
    }

    return out;
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

            if (v1 <= 0 || v2 <= 0 || v3 <= 0 ||  v1 > (int) vertices.size() || v2 > (int) vertices.size() || v3 > (int) vertices.size()) {
                throw std::runtime_error("Invalid face indices in file: " + filepath);
            }

            faces.push_back(TFace {.vertices = { vertices[v1 - 1], vertices[v2 - 1], vertices[v3 - 1] }});
        } else if (type == '#') {
            continue;
        }
    }

    file.close();
    return faces;
}


struct TModelConfig {
    Vector::TVector3 pos;
    Vector::TVector3 color;
    double scale;
    double reflection;
    double transparent;
    double blend;
    bool isLightSource;
};

struct TObjectConfig {
    Vector::TVector3 pos;
    Vector::TVector3 color;
    double r;
    double reflection;
    double transparent;
    int lightsAmount;
};

struct TFloorConfig {
    Vector::TVector3 vertices[4];
    std::string texturePath;
    Vector::TVector3 color;
    double reflection;
};

void BuildModel(const TModelConfig &config, const std::vector<TFace> &mesh, std::vector<Polygon::TPolygon> &out) {
    for (const TFace &face : mesh) {
        std::vector<Vector::TVector3> v;

        for (int i = 0; i < 3; ++i) {
            v.push_back(Vector::Add(config.pos, Vector::Mult(config.scale, face.vertices[i])));
        }

        out.push_back(Polygon::TPolygon {
            .verticles = { v[0], v[1], v[2] },
            .color = config.color,
            .reflection = config.reflection,
            .transparent = config.transparent,
            .blend = config.blend,
            .isLightSource = config.isLightSource,
            .texture = {
                .enabled = false,
                .texture = TextureProjection::TTextureProjection {}
            }
        });
    }
}


struct TLampLineConfig {
    double frameOffset;
    double heightScale;
    double widthScale;
};


void BuildLampLine(const TModelConfig &modelConfig, const std::vector<TFace> &mesh, int n, TLampLineConfig config, std::vector<Polygon::TPolygon> &out) {
    for (int t1 = 0; t1 < (int) mesh.size(); ++t1) {
        for (int t2 = t1 + 1; t2 < (int) mesh.size(); ++t2) {
            Polygon::TPolygon a = { .verticles = { mesh[t1].vertices[0], mesh[t1].vertices[1], mesh[t1].vertices[2] } };
            Polygon::TPolygon b = { .verticles = { mesh[t2].vertices[0], mesh[t2].vertices[1], mesh[t2].vertices[2] } };

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

            Vector::TVector3 na = Vector::Mult(-1.0, Polygon::GetNormal(a));
            Vector::TVector3 nb = Vector::Mult(-1.0, Polygon::GetNormal(b));

            if (std::abs(na.x - nb.x) < 0.001 && std::abs(na.y - nb.y) < 0.001 && std::abs(na.z - nb.z) < 0.001) {
                continue;
            }

            Vector::TVector3 v = Vector::Sub(points.at(1), points.at(0));
            Vector::TVector3 d = Vector::Mult(config.frameOffset, Vector::Normalize(Vector::Add(na, nb)));
            Vector::TVector3 s = Vector::Mult(config.heightScale, Vector::Normalize(Vector::Prod(d, v)));

            Vector::TVector3 diff1 = Vector::Mult(0.1, Vector::Normalize(Vector::Add(d, s)));
            Vector::TVector3 diff2 = Vector::Mult(0.1, Vector::Normalize(Vector::Sub(d, s)));

            std::vector<Vector::TVector3> vertices = {
                Vector::Add(Vector::Add(points[0], Vector::Mult((1.0 - config.widthScale) * 0.5, v)), diff1),
                Vector::Add(Vector::Add(points[0], Vector::Mult((1.0 - config.widthScale) * 0.5, v)), diff2),
                Vector::Add(Vector::Add(points[0], Vector::Mult((1.0 - config.widthScale) * 0.5 + config.widthScale, v)), diff1),
                Vector::Add(Vector::Add(points[0], Vector::Mult((1.0 - config.widthScale) * 0.5 + config.widthScale, v)), diff2)
            };

            std::vector<TFace> faces = {
                { { vertices[0], vertices[1], vertices[2] } },
                { { vertices[3], vertices[2], vertices[1] } }
            };

            BuildModel(TModelConfig {
                .pos = modelConfig.pos,
                .color = { 0.1, 0.1, 0.1 },
                .scale = modelConfig.scale,
                .reflection = 0.0,
                .transparent = 0.0,
                .blend = 0.0,
                .isLightSource = true
            }, faces, out);

            Vector::TVector3 e1 = Vector::Mult(0.05 * config.heightScale, Vector::Normalize(v));
            Vector::TVector3 e2 = Vector::Mult(0.05 * config.heightScale, Vector::Normalize(s));
            Vector::TVector3 e3 = Vector::Mult(0.1 * config.frameOffset, Vector::Normalize(d));

            for(int k = 0; k < n; k++) {
                double t = 1.0 / (n + 1) * (k + 1);
                Vector::TVector3 lampPos = Vector::Add(points[0], Vector::Mult((1.0 - config.widthScale) * 0.5 + t * config.widthScale, v));

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

                BuildModel(TModelConfig {
                    .pos = modelConfig.pos,
                    .color = { 1.0, 1.0, 1.0 },
                    .scale = modelConfig.scale,
                    .reflection = 0.0,
                    .transparent = 1.0,
                    .blend = 0.0,
                    .isLightSource = true
                }, lampFaces, out);
            }

        }
    }
}


void BuildFloor(std::vector<Polygon::TPolygon> &out, DeviceType deviceType, TFloorConfig config) {
    Texture::TTexture floorTexture;
    Texture::Load(&floorTexture, config.texturePath.c_str(), deviceType);

    out.push_back(Polygon::TPolygon {
        .verticles = { config.vertices[0], config.vertices[1], config.vertices[2] },
        .color = config.color,
        .reflection = config.reflection,
        .transparent = 0.0,
        .blend = 0.0,
        .isLightSource = false,
        .texture = {
            .enabled = true,
            .texture = TextureProjection::TTextureProjection {
                .src = floorTexture,
                .verticles = { { 0.0, 1.0, 0.0 }, { 1.0, 1.0, 0.0 }, { 1.0, 0.0, 0.0 }  }
            }
        }
    });

    out.push_back(Polygon::TPolygon {
        .verticles = { config.vertices[2], config.vertices[1], config.vertices[3] },
        .color = config.color,
        .reflection = config.reflection,
        .transparent = 0.0,
        .blend = 0.0,
        .isLightSource = false,
        .texture = {
            .enabled = true,
            .texture = TextureProjection::TTextureProjection {
                .src = floorTexture,
                .verticles = { { 0.0, 0.0, 0.0 }, { 0.0, 1.0, 0.0 }, { 1.0, 0.0, 0.0 }}
            }
        }
    });
}


void BuildTetrahedron(std::vector<Polygon::TPolygon> &out, TObjectConfig config) {
    // Vector::TVector3 pos; { -3.0, -3.0, 1.0 }
    // Vector::TVector3 color; { 0.6, 0.0, 0.0 }
    // double r; 3.0
    // double reflection; 1.0
    // double transparent; 1.0
    // int lightsAmount; 5


    std::vector<TFace> mesh = LoadMesh("objects/tetrahedron.obj");
    TModelConfig modelConfig = {
        .pos = config.pos,
        .color = config.color,
        .scale = config.r,
        .reflection = config.reflection,
        .transparent = config.transparent,
        .blend = 1.0,
        .isLightSource = false
    };

    BuildLampLine(modelConfig, mesh, config.lightsAmount, { .frameOffset = 1.0, .heightScale = 0.5, .widthScale = 0.8 }, out);
    BuildModel(modelConfig, mesh, out);
}

void BuildDodecahedron(std::vector<Polygon::TPolygon> &out, TObjectConfig config) {
    // Vector::TVector3 pos;  { 3.0, 3.0, 2.5 }
    // Vector::TVector3 color; { 0.0, 0.6, 0.0 }
    // double r; 1.0
    // double reflection; 1.0
    // double transparent; 0.6
    // int lightsAmount; 5


    std::vector<TFace> mesh = LoadMesh("objects/dodecahedron.obj");
    TModelConfig modelConfig = {
        .pos = config.pos,
        .color = config.color,
        .scale = 0.2 * config.r,
        .reflection = config.reflection,
        .transparent = config.transparent,
        .blend = 1.0,
        .isLightSource = false
    };

    BuildLampLine(modelConfig, mesh, config.lightsAmount, { .frameOffset = 2.5, .heightScale = 2.5, .widthScale = 1.0 }, out);
    BuildModel(modelConfig, mesh, out);
}


void BuildIcosahedron(std::vector<Polygon::TPolygon> &out, TObjectConfig config) {
    // Vector::TVector3 pos;  { -3.0, 3.0, 2.5 }
    // Vector::TVector3 color; { 0.0, 0.6, 0.6 }
    // double r; 1.0
    // double reflection; 1.0
    // double transparent; 1.0
    // int lightsAmount; 5

    std::vector<TFace> mesh = LoadMesh("objects/icosahedron.obj");
    TModelConfig modelConfig = {
        .pos = config.pos,
        .color = config.color,
        .scale = config.r,
        .reflection = config.reflection,
        .transparent = config.transparent,
        .blend = 1.0,
        .isLightSource = false
    };

    BuildLampLine(modelConfig, mesh, config.lightsAmount, { .frameOffset = 1.1, .heightScale = 0.3, .widthScale = 1.0 }, out);
    BuildModel(modelConfig, mesh, out);
}


void build_space(std::vector<Polygon::TPolygon> &out, DeviceType deviceType, TFloorConfig floorConfig, std::vector<TObjectConfig> objectConfigs) {
    BuildFloor(out, deviceType, floorConfig);
    BuildIcosahedron(out, objectConfigs[0]);
    BuildTetrahedron(out, objectConfigs[1]);
    BuildDodecahedron(out, objectConfigs[2]);
}

#endif
