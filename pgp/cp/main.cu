#include <iostream>
#include <vector>
#include <string>
#include <cassert>

#include "vector.cuh"
#include "scene.cuh"
#include "canvas.cuh"
#include "timer.cuh"

#include "cpu_render.cuh"
#include "gpu_render.cuh"

typedef enum {
    ON_CPU,
    ON_GPU
} TRenderMode;

typedef enum {
    RENDER,
    PRINT_DEFAULT_CONFIG
} TAction;

struct TCameraMotionConfig {
    double r_0_c;
    double z_0_c;
    double phi_0_c;
    double a_r_c;
    double a_z_c;
    double omega_r_c;
    double omega_z_c;
    double omega_phi_c;
    double p_r_c;
    double p_z_c;
    double r_0_n;
    double z_0_n;
    double phi_0_n;
    double a_r_n;
    double a_z_n;
    double omega_r_n;
    double omega_z_n;
    double omega_phi_n;
    double p_r_n;
    double p_z_n;
};

struct TLightConfig {
    Vector::TVector3 pos;
    Vector::TVector3 color;
};

struct TConfig {
    unsigned int frames;
    std::string outputFileTemplate;
    unsigned int canvasWidth;
    unsigned int canvasHeight;
    double angle;
    TCameraMotionConfig motionConfig;
    std::vector<TObjectConfig> objectsConfig;
    TFloorConfig floorConfig;
    std::vector<TLightConfig> lightsConfig;
    int recursionDepth;
    int ssaaCoef;
};

TRenderMode GetRenderMode(const std::vector<std::string> &args) {
    for (const std::string &arg : args) {
        if (arg == "--cpu") return TRenderMode::ON_CPU;
        if (arg == "--gpu") return TRenderMode::ON_GPU;
    }
    return TRenderMode::ON_GPU;
}

TAction GetAction(const std::vector<std::string> &args) {
    for (const std::string &arg : args) {
        if (arg == "--default") return TAction::PRINT_DEFAULT_CONFIG;
    }
    return TAction::RENDER;
}

TConfig ReadRenderConfig() {
    TConfig config;

    std::cin >> config.frames;
    std::cin >> config.outputFileTemplate;
    std::cin >> config.canvasWidth >> config.canvasHeight >> config.angle;

    std::cin >> config.motionConfig.r_0_c;
    std::cin >> config.motionConfig.z_0_c;
    std::cin >> config.motionConfig.phi_0_c;
    std::cin >> config.motionConfig.a_r_c;
    std::cin >> config.motionConfig.a_z_c;
    std::cin >> config.motionConfig.omega_r_c;
    std::cin >> config.motionConfig.omega_z_c;
    std::cin >> config.motionConfig.omega_phi_c;
    std::cin >> config.motionConfig.p_r_c;
    std::cin >> config.motionConfig.p_z_c;
    std::cin >> config.motionConfig.r_0_n;
    std::cin >> config.motionConfig.z_0_n;
    std::cin >> config.motionConfig.phi_0_n;
    std::cin >> config.motionConfig.a_r_n;
    std::cin >> config.motionConfig.a_z_n;
    std::cin >> config.motionConfig.omega_r_n;
    std::cin >> config.motionConfig.omega_z_n;
    std::cin >> config.motionConfig.omega_phi_n;
    std::cin >> config.motionConfig.p_r_n;
    std::cin >> config.motionConfig.p_z_n;

    for (int i = 0; i < 3; ++i) {
        TObjectConfig objectConfig;
        std::cin >> objectConfig.pos.x >> objectConfig.pos.y >> objectConfig.pos.z;
        std::cin >> objectConfig.color.x >> objectConfig.color.y >> objectConfig.color.z;
        std::cin >> objectConfig.r;
        std::cin >> objectConfig.reflection;
        std::cin >> objectConfig.transparent;
        std::cin >> objectConfig.lightsAmount;

        config.objectsConfig.push_back(objectConfig);
    }

    for (int i = 0; i < 4; ++i) {
        std::cin >> config.floorConfig.vertices[i].x;
        std::cin >> config.floorConfig.vertices[i].y;
        std::cin >> config.floorConfig.vertices[i].z;
    }

    std::cin >> config.floorConfig.texturePath;
    std::cin >> config.floorConfig.color.x;
    std::cin >> config.floorConfig.color.y;
    std::cin >> config.floorConfig.color.z;
    std::cin >> config.floorConfig.reflection;

    int lightsAmount;
    std::cin >> lightsAmount;

    for (int i = 0; i < lightsAmount; ++i) {
        TLightConfig lightConfig;
        std::cin >> lightConfig.pos.x >> lightConfig.pos.y >> lightConfig.pos.z;
        std::cin >> lightConfig.color.x >> lightConfig.color.y >> lightConfig.color.z;
        config.lightsConfig.push_back(lightConfig);
    }

    std::cin >> config.recursionDepth;
    std::cin >> config.ssaaCoef;

    return config;
}

int main(int argc, char *argv[]) {
    assert(argc <= 2);

    std::vector<std::string> args;
    char buff[256];

    for (int i = 0; i < argc; ++i) {
        args.push_back(std::string(argv[i]));
    }

    TRenderMode renderMode = GetRenderMode(args);
    TAction action = GetAction(args);

    if (action == TAction::PRINT_DEFAULT_CONFIG) {
        std::ifstream inputFile("example.in");
        std::string line;

        if (!inputFile.is_open()) {
            return 1;
        }

        while (std::getline(inputFile, line)) {
            std::cout << line << std::endl;
        }

        inputFile.close();
        return 0;
    }

    TConfig config = ReadRenderConfig();
    DeviceType deviceType = (renderMode == TRenderMode::ON_CPU) ? DeviceType::CPU : DeviceType::GPU;    
    Canvas::TCanvas canvas, extendedCanvas;

    Canvas::Init(&canvas, config.canvasWidth, config.canvasHeight, deviceType);
    Canvas::Init(&extendedCanvas, config.ssaaCoef * config.canvasWidth, config.ssaaCoef * config.canvasHeight, deviceType);

    std::vector<Polygon::TPolygon> polygons;
    build_space(polygons, deviceType, config.floorConfig, config.objectsConfig);

    std::vector<Renderer::TLight> lights;
    for (const TLightConfig lightConfig : config.lightsConfig) {
        lights.push_back(Renderer::TLight {
            .position = lightConfig.pos,
            .color = lightConfig.color
        });
    }

    double tau = 2.0 * M_PI / config.frames;

    for (int k = 0; k < (int) config.frames; ++k) {
        double t = k * tau;
        Timer::TTimer timer;
        size_t raysCount = 0;

        Vector::TVector3 pc = Vector::FromCylindric({
            config.motionConfig.r_0_c + config.motionConfig.a_r_c * sin(config.motionConfig.omega_r_c * t + config.motionConfig.p_r_c),
            config.motionConfig.z_0_c + config.motionConfig.a_z_c * sin(config.motionConfig.omega_z_c * t + config.motionConfig.p_z_c),
            config.motionConfig.phi_0_c + config.motionConfig.omega_phi_c * t
        });

        Vector::TVector3 pv = Vector::FromCylindric({
            config.motionConfig.r_0_n + config.motionConfig.a_r_n * sin(config.motionConfig.omega_r_n * t + config.motionConfig.p_r_n),
            config.motionConfig.z_0_n + config.motionConfig.a_z_n * sin(config.motionConfig.omega_z_n * t + config.motionConfig.p_z_n),
            config.motionConfig.phi_0_n + config.motionConfig.omega_phi_n * t
        });

        if (deviceType == DeviceType::GPU) {
            Timer::Start(&timer);
            GpuRenderer::Render(
                pc, pv, config.angle,
                &extendedCanvas,
                polygons, lights,
                &raysCount,
                config.recursionDepth
            );

            GpuRenderer::Ssaa<<<GpuRenderer::DEVICE_BLOCKS, GpuRenderer::DEVICE_THREADS>>>(extendedCanvas, canvas, config.ssaaCoef);
            cudaDeviceSynchronize();
            SAVE_CUDA(cudaGetLastError());
            Timer::Stop(&timer);
        } else {
            Timer::Start(&timer);
            CpuRenderer::Render(
                pc, pv, config.angle,
                &extendedCanvas,
                polygons, lights,
                &raysCount,
                config.recursionDepth
            );

            CpuRenderer::Ssaa(&extendedCanvas, &canvas, config.ssaaCoef);
            Timer::Stop(&timer);
        }

        printf("%d\t%lu\t%lld\n", k, raysCount, Timer::GetTime(&timer));

        sprintf(buff, config.outputFileTemplate.c_str(), k);
        Canvas::Dump(&canvas, buff);
    }

    Canvas::Destroy(&canvas);
    Canvas::Destroy(&extendedCanvas);

    return 0;
}
