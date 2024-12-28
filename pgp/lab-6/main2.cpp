#include <GL/glut.h>
#include <math.h>


/* Configuration */


const float dt = 0.08f;
const float w = 0.99f; // коэффициент замедления
const float g = 10.0f; // ускорение свободного падения
const float eps = 0.0001f;
const float K = 50.0f; // коэффициент пропорциональности


/* State */

struct {
    float x, y, z;
    float vx, vy, vz;
    float speed;
    float yaw, pitch;
} camera = {
    0.0, 1.0, -60.0,
    0.0, 0.0, 0.0,
    0.2,
    0.0 * M_PI, 0.5 * M_PI
};

struct {
    float x, y, z;
    float size;
    float color[3];
    int width;
} cube = { 0.0, 0.0, 0.0, 15.0f, { 1.0f, 1.0f, 1.0f }, 2 };


struct {
    float x, y, z;
    float vx, vy, vz;
    float size;
    float color[3];
    float q; // заряд
} particle = { 0.5, 0.0, 0.0, 0.0f, 0.0f, 0.0f, 0.5, { 1.0f, 0.5f, 0.0f }, 1.0f };


/* Update logic */


void updateParticle() {
    float vx2 = 0.0f, vy2 = 0.0f, vz2 = 0.0f;

    vx2 += w * particle.vx;
    vy2 += w * particle.vy;
    vz2 += w * particle.vz;

    // гравитация
    vz2 += - g * dt;

    // границы куба
    vx2 += powf(particle.q, 2.0) * K * (fabs(particle.x - (cube.x - cube.size))) / (powf(fabs(particle.x - (cube.x - cube.size)), 3.0) + eps) * dt;
    vx2 += powf(particle.q, 2.0) * K * (-fabs(particle.x - (cube.x + cube.size))) / (powf(fabs(particle.x - (cube.x + cube.size)), 3.0) + eps) * dt;

    vy2 += powf(particle.q, 2.0) * K * (particle.y - (cube.y - cube.size)) / (powf(fabs(particle.y - (cube.y - cube.size)), 3.0) + eps) * dt;
    vy2 += powf(particle.q, 2.0) * K * (-fabs(particle.y - (cube.y + cube.size))) / (powf(fabs(particle.y - (cube.y + cube.size)), 3.0) + eps) * dt;

    vz2 += powf(particle.q, 2.0) * K * (fabs(particle.z - (cube.z - cube.size))) / (powf(fabs(particle.z - (cube.z - cube.size)), 3.0) + eps) * dt;
    vz2 += powf(particle.q, 2.0) * K * (-fabs(particle.z - (cube.z + cube.size))) / (powf(fabs(particle.z - (cube.z + cube.size)), 3.0) + eps) * dt;

    particle.vx = vx2;
    particle.vy = vy2;
    particle.vz = vz2;

    particle.x = particle.x + particle.vx * dt;
    particle.y = particle.y + particle.vy * dt;
    particle.z = particle.z + particle.vz * dt;
}


/* Display logic */


void drawParticle() {
	glPushMatrix();
		glTranslatef(particle.x, particle.y, particle.z);
        glColor3f(particle.color[0], particle.color[1], particle.color[2]);
		glutSolidSphere(particle.size, 16, 16);
	glPopMatrix();
}


void drawCube() {
    glLineWidth(cube.width);
    glColor3f(cube.color[0], cube.color[1], cube.color[2]);

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
}


void updateCamera() {
    camera.x += camera.vx * dt;
    camera.y += camera.vy * dt;
    camera.z += camera.vz * dt;

    camera.vx *= w;
    camera.vy *= w;
    camera.vz *= w;
}


void setCamera() {
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();

    gluPerspective(90.0f, 1.0f, 0.1f, 100.0f);

	gluLookAt(
        camera.x, camera.y, camera.z,
        camera.x + cos(camera.yaw) * cos(camera.pitch),
        camera.y + sin(camera.yaw) * cos(camera.pitch),
        camera.z + sin(camera.pitch),
        0.0f, 0.0f, 1.0f
    );

    // glRotatef(90.0f, 0.0f, 0.0f, 1.0f); // Поворот вокруг оси Y
    // glRotatef(180.0f * camera.pitch, 1.0f, 0.0f, 0.0f); // Поворот вокруг оси X
    // glRotatef(180.0f * camera.yaw, 0.0f, 1.0f, 0.0f); // Поворот вокруг оси Y
}

/* Event handlers */

void keys(unsigned char key, int x, int y) {
	switch (key) {
		case 'w':
			camera.vy += 1.0 * camera.speed;
            // camera.vy +=  cos(camera.yaw) * cos(camera.pitch) * camera.speed;
			// camera.vz += cos(camera.pitch) * camera.speed;
		break;
		case 's':
			camera.vx +=  -sin(camera.yaw) * cos(camera.pitch) * camera.speed;
			camera.vy +=  -cos(camera.yaw) * cos(camera.pitch) * camera.speed;
			camera.vz +=  -sin(camera.pitch) * camera.speed;
		break;
		case 'a':
			camera.vy +=  -sin(camera.yaw) * camera.speed;
			camera.vx +=  cos(camera.yaw) * camera.speed;
			break;
		case 'd':
			camera.vy +=  sin(camera.yaw) * camera.speed;
			camera.vx +=  -cos(camera.yaw) * camera.speed;
		break;
		case 27:
			exit(0);
		break;
	}
}

void update() {
    updateCamera();
    updateParticle();
    glutPostRedisplay();
}


void display() {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); 

    setCamera();
    drawCube();
    drawParticle();

    glFlush();
    glutSwapBuffers();
}


void initWindow() {
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);
    glutInitWindowSize(400, 400);
    glutCreateWindow("Hot Map");
    glClearColor(0.0, 0.0, 0.0, 1.0);
}


int main(int argc, char** argv) {
    glutInit(&argc, argv);
    initWindow();

    glutIdleFunc(update);
    glutDisplayFunc(display);
    glutKeyboardFunc(keys);
    glutMainLoop();

    return 0;
}
