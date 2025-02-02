#include <GL/glut.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>

int w = 400;
int h = 400;

/* Configuration */

const float dt = 0.08f;
const float W = 0.99f; // коэффициент замедления
const float g = 10.0f; // ускорение свободного падения
const float eps = 0.0001f;
const float K = 50.0f; // коэффициент пропорциональности

/* GIZMO */

void drawCube(float x, float y, float z, float size, float r, float g, float b) {
    glColor3f(r, g, b);
    glPushMatrix();
    glTranslatef(x, y, z);

    float half = size / 2;

    // Front face
    glBegin(GL_QUADS);
    glVertex3f(-half, -half, half);
    glVertex3f(half, -half, half);
    glVertex3f(half, half, half);
    glVertex3f(-half, half, half);
    glEnd();

    // Back face
    glBegin(GL_QUADS);
    glVertex3f(-half, -half, -half);
    glVertex3f(-half, half, -half);
    glVertex3f(half, half, -half);
    glVertex3f(half, -half, -half);
    glEnd();

    // Left face
    glBegin(GL_QUADS);
    glVertex3f(-half, -half, -half);
    glVertex3f(-half, -half, half);
    glVertex3f(-half, half, half);
    glVertex3f(-half, half, -half);
    glEnd();

    // Right face
    glBegin(GL_QUADS);
    glVertex3f(half, -half, -half);
    glVertex3f(half, half, -half);
    glVertex3f(half, half, half);
    glVertex3f(half, -half, half);
    glEnd();

    // Top face
    glBegin(GL_QUADS);
    glVertex3f(-half, half, -half);
    glVertex3f(-half, half, half);
    glVertex3f(half, half, half);
    glVertex3f(half, half, -half);
    glEnd();

    // Bottom face
    glBegin(GL_QUADS);
    glVertex3f(-half, -half, -half);
    glVertex3f(half, -half, -half);
    glVertex3f(half, -half, half);
    glVertex3f(-half, -half, half);
    glEnd();

    glPopMatrix();
}

void drawSphere(float x, float y, float z, float radius, float r, float g, float b) {
    glColor3f(r, g, b);
    glPushMatrix();
    glTranslatef(x, y, z);
    glutSolidSphere(radius, 32, 32);
    glPopMatrix();
}

void displayGizmo() {
    float k = 30.0f;
    float size = 2.0f;

    // Draw squares
    drawCube(k, 0.0f, 0.0f, size, 1.0f, 0.0f, 0.0f); // Red square
    drawCube(0.0f, k, 0.0f, size, 0.0f, 1.0f, 0.0f); // Green square
    drawCube(0.0f, 0.0f, k, size, 0.0f, 0.0f, 1.0f); // Blue square

    // Draw spheres
    drawSphere(-k, 0.0f, 0.0f, 0.5 * size, 1.0f, 0.0f, 0.0f); // Red sphere
    drawSphere(0.0f, -k, 0.0f, 0.5 * size, 0.0f, 1.0f, 0.0f); // Green sphere
    drawSphere(0.0f, 0.0f, -k, 0.5 * size, 0.0f, 0.0f, 1.0f); // Blue sphere
}

/* Cube */

struct {
    // position vector
    float x = 0.0f;
    float y = 0.0f;
    float z = 15.0f;

    float size = 15.0f;
} cube = {};

void displayCube() {
    glColor3f(1.0f, 1.0f, 1.0f);

    glBegin(GL_QUADS);
        glVertex3f(cube.x - cube.size, cube.y - cube.size, cube.z - cube.size);
        glVertex3f(cube.x + cube.size, cube.y - cube.size, cube.z - cube.size);
        glVertex3f(cube.x + cube.size, cube.y + cube.size, cube.z - cube.size);
        glVertex3f(cube.x - cube.size, cube.y + cube.size, cube.z - cube.size);
    glEnd();

	glLineWidth(2);				
	glColor3f(1.0f, 1.0f, 1.0f);

	glBegin(GL_LINES);
		glVertex3f(cube.x - cube.size, cube.y - cube.size, cube.z - cube.size);
		glVertex3f(cube.x - cube.size, cube.y - cube.size, cube.z + cube.size);

		glVertex3f(cube.x + cube.size, cube.y - cube.size, cube.z - cube.size);
		glVertex3f(cube.x + cube.size, cube.y - cube.size, cube.z + cube.size);

		glVertex3f(cube.x + cube.size, cube.y + cube.size, cube.z - cube.size);
		glVertex3f(cube.x + cube.size, cube.y + cube.size, cube.z + cube.size);

		glVertex3f(cube.x - cube.size, cube.y + cube.size, cube.z - cube.size);
		glVertex3f(cube.x - cube.size, cube.y + cube.size, cube.z + cube.size);
	glEnd();

	glBegin(GL_LINE_LOOP);
		glVertex3f(cube.x - cube.size, cube.y - cube.size, cube.z - cube.size);
		glVertex3f(cube.x + cube.size, cube.y - cube.size, cube.z - cube.size);
		glVertex3f(cube.x + cube.size, cube.y + cube.size, cube.z - cube.size);
		glVertex3f(cube.x - cube.size, cube.y + cube.size, cube.z - cube.size);
	glEnd();

	glBegin(GL_LINE_LOOP);
		glVertex3f(cube.x - cube.size, cube.y - cube.size, cube.z + cube.size);
		glVertex3f(cube.x + cube.size, cube.y - cube.size, cube.z + cube.size);
		glVertex3f(cube.x + cube.size, cube.y + cube.size, cube.z + cube.size);
		glVertex3f(cube.x - cube.size, cube.y + cube.size, cube.z + cube.size);
	glEnd();

	glColor3f(1.0f, 1.0f, 1.0f);
}

/* Camera */

struct {
    // position vector
    float x = 0.0f;
    float y = 0.0f;
    float z = 60.0f;

    // velocity vector
    float vx = 0.0f;
    float vy = 0.0f;
    float vz = 0.0f;

    float yaw = 0.0f; // xy rotation angle
    float pitch = -0.5 * M_PI; // yz rotation angle

    float dyaw = 0.0f;
    float dpitch = 0.0f;

    float speed = 0.1f; // move speed

    float q = 30.0f; // заряд игрока
} camera = {};

void displayCamera() {
    glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	
	gluPerspective(90.0f, 1.0f, 0.1f, 100.0f);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	gluLookAt(
        camera.x, camera.y, camera.z,
        camera.x + cos(camera.yaw) * cos(camera.pitch),
        camera.y + sin(camera.yaw) * cos(camera.pitch),
        camera.z + sin(camera.pitch),
        0.0f, 0.0f, 1.0f
    );
}

void updateCamera() {
	float v = sqrt(
        camera.vx * camera.vx +
        camera.vy * camera.vy +
        camera.vz * camera.vz
    );

	if (v > camera.speed) {
		camera.vx *= camera.speed / v;
		camera.vy *= camera.speed / v;
		camera.vz *= camera.speed / v;
	}

	camera.x += camera.vx; camera.vx *= 0.99;
	camera.y += camera.vy; camera.vy *= 0.99;
	camera.z += camera.vz; camera.vz *= 0.99;

	// if (camera.z < 1.0) {
	// 	camera.z = 1.0;
	// 	camera.vz = 0.0;
	// }

	if (fabs(camera.dpitch) + fabs(camera.dyaw) > 0.0001) {
		camera.yaw += camera.dyaw;
		camera.pitch += camera.dpitch;

		camera.pitch = fmin(0.5f * M_PI, fmax(-0.5f * M_PI, camera.pitch));
		camera.dyaw *= 0.5;
		camera.dpitch *= 0.5;
	}
}

void keys(unsigned char key, int x, int y) {
	switch (key) {
		case 'w':
            camera.vx += cos(camera.yaw) * cos(camera.pitch) * camera.speed;
			camera.vy += sin(camera.yaw) * cos(camera.pitch) * camera.speed;
			camera.vz += sin(camera.pitch) * camera.speed;
		break;
		case 's':
			camera.vx += -cos(camera.yaw) * cos(camera.pitch) * camera.speed;
			camera.vy += -sin(camera.yaw) * cos(camera.pitch) * camera.speed;
			camera.vz += -sin(camera.pitch) * camera.speed;
		break;
		case 'a':
			camera.vx += -sin(camera.yaw) * camera.speed;
			camera.vy += cos(camera.yaw) * camera.speed;
			break;
		case 'd':
			camera.vx += sin(camera.yaw) * camera.speed;
			camera.vy += -cos(camera.yaw) * camera.speed;
		break;
		case 27:
			exit(0);
		break;
	}
}

void mouse(int x, int y) {
	static int x_prev = w / 2, y_prev = h / 2;

	float dx = 0.005f * (x - x_prev);
    float dy = 0.005f * (y - y_prev);

	camera.dyaw -= dx;
    camera.dpitch -= dy;
	x_prev = x;
	y_prev = y;

	// if ((x < 20) || (y < 20) || (x > w - 20) || (y > h - 20)) {
	// 	glutWarpPointer(w / 2, h / 2);
	// 	x_prev = w / 2;
	// 	y_prev = h / 2;
    // }
}

/* Bullet */

struct {
    // position
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;

    // velocity
    float vx = 0.5f;
    float vy = 0.5f;
    float vz = 0.5f;

    float color[3] = { 0.75f, 0.75f, 0.0f };
    float size = 0.8f;
    float speed = 1.0f;

    bool isActive = false;
    float q = 50.0f;
} bullet = {};

void updateBullet() {
    bullet.x = bullet.x + bullet.vx * dt;
    bullet.y = bullet.y + bullet.vy * dt;
    bullet.z = bullet.z + bullet.vz * dt;
}

void displayBullet() {
    if (!bullet.isActive) return;

    glPushMatrix();
        glTranslatef(bullet.x, bullet.y, bullet.z);
        glColor3f(bullet.color[0], bullet.color[1], bullet.color[2]);
        glutSolidSphere(bullet.size, 16, 16);
    glPopMatrix();    
}

/* Particle */

struct TParticle {
    float x = 10.0f;
    float y = 0.0f;
    float z = 5.0f;

    float vx = 0.0f;
    float vy = 0.0f;
    float vz = 0.0f;

    float size = 0.5f;
    float color[3] = { 1.0f, 0.5f, 0.0f };
    float q = 5.0f;
};

std::vector<TParticle> particles;

void updateParticle() {
    for (size_t i = 0; i < particles.size(); ++i) {
        TParticle& particle = particles[i];

        float vx2 = 0.0f, vy2 = 0.0f, vz2 = 0.0f;

        vx2 += W * particle.vx;
        vy2 += W * particle.vy;
        vz2 += W * particle.vz;

        // гравитация
        vz2 += - g * dt;

        // границы куба
        vx2 += powf(particle.q, 2.0) * K * (fabs(particle.x - (cube.x - cube.size))) / (powf(fabs(particle.x - (cube.x - cube.size)), 3.0) + eps) * dt;
        vx2 += powf(particle.q, 2.0) * K * (-fabs(particle.x - (cube.x + cube.size))) / (powf(fabs(particle.x - (cube.x + cube.size)), 3.0) + eps) * dt;

        vy2 += powf(particle.q, 2.0) * K * (particle.y - (cube.y - cube.size)) / (powf(fabs(particle.y - (cube.y - cube.size)), 3.0) + eps) * dt;
        vy2 += powf(particle.q, 2.0) * K * (-fabs(particle.y - (cube.y + cube.size))) / (powf(fabs(particle.y - (cube.y + cube.size)), 3.0) + eps) * dt;

        vz2 += powf(particle.q, 2.0) * K * (fabs(particle.z - (cube.z - cube.size))) / (powf(fabs(particle.z - (cube.z - cube.size)), 3.0) + eps) * dt;
        vz2 += powf(particle.q, 2.0) * K * (-fabs(particle.z - (cube.z + cube.size))) / (powf(fabs(particle.z - (cube.z + cube.size)), 3.0) + eps) * dt;

        // отталкивание от игрока
        float l_ic = sqrt(
            powf(particle.x - camera.x, 2.0) +
            powf(particle.y - camera.y, 2.0) +
            powf(particle.z - camera.z, 2.0)
        );

        vx2 += particle.q * camera.q * K * (particle.x - camera.x) / (powf(l_ic, 3.0) + eps) * dt;
        vy2 += particle.q * camera.q * K * (particle.y - camera.y) / (powf(l_ic, 3.0) + eps) * dt;
        vz2 += particle.q * camera.q * K * (particle.z - camera.z) / (powf(l_ic, 3.0) + eps) * dt;

        // отталкивание от пули
        if (bullet.isActive) {
            float l_ib = sqrt(
                powf(particle.x - bullet.x, 2.0) +
                powf(particle.y - bullet.y, 2.0) +
                powf(particle.z - bullet.z, 2.0)
            );

            vx2 += particle.q * bullet.q * K * (particle.x - bullet.x) / (powf(l_ib, 3.0) + eps) * dt;
            vy2 += particle.q * bullet.q * K * (particle.y - bullet.y) / (powf(l_ib, 3.0) + eps) * dt;
            vz2 += particle.q * bullet.q * K * (particle.z - bullet.z) / (powf(l_ib, 3.0) + eps) * dt;
        }

        // отталкивание от других частиц
        for (size_t j = 0; j < particles.size(); ++j) {
            if (i == j) continue;
            const TParticle& p2 = particles[j];

            float l_ij = sqrt(
                powf(particle.x - p2.x, 2.0) +
                powf(particle.y - p2.y, 2.0) +
                powf(particle.z - p2.z, 2.0)
            );

            vx2 += particle.q * p2.q * K * (particle.x - p2.x) / (powf(l_ij, 3.0) + eps) * dt;
            vy2 += particle.q * p2.q * K * (particle.y - p2.y) / (powf(l_ij, 3.0) + eps) * dt;
            vz2 += particle.q * p2.q * K * (particle.z - p2.z) / (powf(l_ij, 3.0) + eps) * dt;
        }

        particle.vx = vx2;
        particle.vy = vy2;
        particle.vz = vz2;

        particle.x = particle.x + particle.vx * dt;
        particle.y = particle.y + particle.vy * dt;
        particle.z = particle.z + particle.vz * dt;
    }
}

void drawParticle() {
    for (const TParticle& particle : particles) {
        glPushMatrix();
            glTranslatef(particle.x, particle.y, particle.z);
            glColor3f(particle.color[0], particle.color[1], particle.color[2]);
            glutSolidSphere(particle.size, 16, 16);
        glPopMatrix();
    }
}

void generateParticles() {
    int n = 4, m = 4;
    float pad = 3.0f;

    float x1 = cube.x - cube.size + pad;
    float x2 = cube.x + cube.size - pad;

    float y1 = cube.y - cube.size + pad;
    float y2 = cube.y + cube.size - pad;

    particles.clear();

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            TParticle particle;
            particle.x = x1 + (x2 - x1) / (n - 1) * i;
            particle.y = y1 + (y2 - y1) / (m - 1) * j;
            particle.z = cube.z;

            particles.push_back(particle);
        }
    }
}

/* Common logic */

void initialize() {
    glEnable(GL_DEPTH_TEST);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(90.0f, 1.0f, 0.1f, 100.0f);
    glMatrixMode(GL_MODELVIEW);
}

void display() {
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    displayCamera();
    displayGizmo();
    displayCube();
    drawParticle();
    displayBullet();

	glutSwapBuffers();
}

void update() {
    updateCamera();
    updateParticle();
    glutPostRedisplay();
    updateBullet();
}

void onClick(int button, int state, int x, int y) {
    if (button == GLUT_LEFT_BUTTON && state == GLUT_DOWN) {
        bullet.vx = bullet.speed * cos(camera.yaw) * cos(camera.pitch);
        bullet.vy = bullet.speed * sin(camera.yaw) * cos(camera.pitch);
        bullet.vz = bullet.speed * sin(camera.pitch);

        bullet.x = camera.x + bullet.vx;
        bullet.y = camera.y + bullet.vy;
        bullet.z = camera.z + bullet.vz;

        bullet.isActive = true;
    }
}

int main(int argc, char **argv) {
	glutInit(&argc, argv);
    generateParticles();

	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA);
	glutInitWindowSize(400, 400);
	glutCreateWindow("OpenGL");

	glutIdleFunc(update);
	glutDisplayFunc(display);
	glutPassiveMotionFunc(mouse);
	glutKeyboardFunc(keys);
    glutMouseFunc(onClick);

	// glutSetCursor(GLUT_CURSOR_NONE);

	glShadeModel(GL_SMOOTH);                             // Разрешение сглаженного закрашивания
	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);                // Черный фон
	glClearDepth(1.0f);                                  // Установка буфера глубины
	glDepthFunc(GL_LEQUAL);                              // Тип теста глубины. 
	glEnable(GL_DEPTH_TEST);                			 // Включаем тест глубины
	glEnable(GL_CULL_FACE);                 			 // Режим при котором, тектуры накладываются только с одной стороны
	
	glutMainLoop();

    return 0;
}
