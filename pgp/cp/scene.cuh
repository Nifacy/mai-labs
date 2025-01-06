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

            if (v1 <= 0 || v2 <= 0 || v3 <= 0 ||  v1 > vertices.size() || v2 > vertices.size() || v3 > vertices.size()) {
                throw std::runtime_error("Invalid face indices in file: " + filepath);
            }

            faces.push_back(TFace {.vertices = { vertices[v1 - 1], vertices[v2 - 1], vertices[v3 - 1] }});
        } else if (type == '#') {
            continue;
        }
    }

    std::cerr << "[log] read " << vertices.size() << " vertices\n";

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
            .isLightSource = config.isLightSource
        });
    }
}


void BuildLampLine(const TModelConfig &modelConfig, const std::vector<TFace> &mesh, int n, std::vector<Polygon::TPolygon> &out) {
    for (int t1 = 0; t1 < mesh.size(); ++t1) {
        for (int t2 = t1 + 1; t2 < mesh.size(); ++t2) {
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

            BuildModel(TModelConfig {
                .pos = modelConfig.pos,
                .color = { 0.1, 0.1, 0.1 },
                .scale = modelConfig.scale,
                .reflection = 0.0,
                .transparent = 0.0,
                .blend = 0.0,
                .isLightSource = true
            }, faces, out);

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


void BuildFloor(std::vector<Polygon::TPolygon> &out, DeviceType deviceType) {
    Texture::TTexture floorTexture;
    Texture::Load(&floorTexture, "textures/floor.data", deviceType);

    out.push_back(Polygon::TPolygon {
        .verticles = { { -5.0, -5.0, 0.0 }, { 5.0, -5.0, 0.0 }, { -5.0, 5.0, 0.0 } },
        .color = { 1.0, 1.0, 1.0 },
        .reflection = 0.0,
        .transparent = 0.0,
        .blend = 0.0,
        .isLightSource = false,
        .texture = {
            .enabled = true,
            .texture = TextureProjection::TTextureProjection {
                .src = floorTexture,
                .verticles = { { 0.0, 0.0, 0.0 }, { 0.0, 1.0, 0.0 }, { 1.0, 0.0, 0.0 } }
            }
        }
    });

    out.push_back(Polygon::TPolygon {
        .verticles = { { -5.0, 5.0, 0.0 }, { 5.0, -5.0, 0.0 }, { 5.0, 5.0, 0.0 } },
        .color = { 1.0, 1.0, 1.0 },
        .reflection = 0.0,
        .transparent = 0.0,
        .blend = 0.0,
        .isLightSource = false,
        .texture = {
            .enabled = true,
            .texture = TextureProjection::TTextureProjection {
                .src = floorTexture,
                .verticles = { { 1.0, 1.0, 0.0 }, { 0.0, 1.0, 0.0 }, { 1.0, 0.0, 0.0 } }
            }
        }
    });
}


void build_space(std::vector<Polygon::TPolygon> &out, DeviceType deviceType) {
    BuildFloor(out, deviceType);
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
    // std::vector<TFace> cubeMesh = LoadCubeMesh();
    // TModelConfig cubeConfig =         {
    //         .pos = { 0.0, 0.0, 0.5 + 2.0 * EPS },
    //         .color = { 0.2, 1.0, 0.2 },
    //         .scale = 3.0,
    //         .reflection = 1.0,
    //         .transparent = 0.0,
    //         .blend = 1.0
    //     };

    // BuildLampLine(
    //     cubeConfig,
    //     cubeMesh,
    //     2,
    //     out
    // );

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

    // BuildModel(
    //     cubeConfig,
    //     cubeMesh,
    //     out
    // );

    // std::vector<TFace> mesh = LoadMesh("objects/model2.obj");

    // BuildLampLine(
    //     cubeConfig,
    //     mesh,
    //     2,
    //     out
    // );

    // BuildModel(cubeConfig, mesh, out);
}

#endif
