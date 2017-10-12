#include <iostream>
#include <cmath>
#include <fstream>
#include <cstring>
#include <cstdio>
using namespace std;


const double PI = 4*atan(1);
const double EPS = 0.1;
const double ALPHA = 1e-1;

const int W = 16;
const int X = 32;
const int Y = 42;
const int Z = 32;

struct vec {
    double x, y, z;
};

double images[W][X][Y];

vec norm[X][Y][Z];
vec grad[X][Y][Z];
vec *hull[X][Y];
double zbuf[X][Y];
double proj[X][Y];
double max_proj[X][Y][Z];


double dot(vec a, vec b) {
    return a.x*b.x + a.y*b.y + a.z*b.z;
}


void read_images(string path) {
    ifstream fin(path, ios::binary);
    for(int w = 0; w < W; w++) {
        for(int x = 0; x < X; x++) {
            for(int y = 0; y < Y; y++) {
                float f;
                fin.read(reinterpret_cast<char*>(&f), sizeof(float));
                images[w][x][y] = f;
            }
        }
    }
    fin.close();
}


void make_hull(vec cam) {
    double costh = cam.z, sinth = -cam.x;
    vec p = {X/2+X*sqrt(2)/2*sinth, Y/2, Z/2-Z*sqrt(2)/2*costh};

    memset(hull, 0, sizeof(hull));
    for(int x = 0; x < X; x++) {
        for(int y = 0; y < Y; y++) {
            for(int z = 0; z < Z; z++) {
                vec *v = &norm[x][y][z];
                if(dot(*v, cam) > 0) {
                    continue;
                }
                vec t{x-p.x, y-p.y, z-p.z};
                vec r{costh*t.x+sinth*t.z, t.y, -sinth*t.x+costh*t.z};
                int px = int(r.x+X/2), py = int(r.y+Y/2);
                if(px < 0 || px >= X || py < 0 || py >= Y) {
                    continue;
                }
                if(hull[px][py] == NULL || r.z < zbuf[px][py]) {
                    hull[px][py] = v;
                    zbuf[px][py] = r.z;
                }
            }
        }
    }
}


void make_proj(vec cam) {
    make_hull(cam);
    for(int x = 0; x < X; x++) {
        for(int y = 0; y < Y; y++) {
            vec *v = hull[x][y];
            if(v != NULL) {
                proj[x][y] = -dot(*v, cam);
            } else {
                proj[x][y] = 0.0;
            }
        }
    }
}


vec get_cam(int w) {
    double theta = 2.0*PI/W*w;
    return vec{-sin(theta), 0, cos(theta)};
}


void make_max_proj() {
    memset(max_proj, 0, sizeof(max_proj));
    for(int w = 0; w < W; w++) {
        vec cam = get_cam(w);
        make_hull(cam);
        for(int x = 0; x < X; x++) {
            for(int y = 0; y < Y; y++) {
                vec *v = hull[x][y];
                if(v != NULL) {
                    int i = v-**norm;
                    double &mp = max_proj[0][0][i];
                    mp = max(mp, -dot(*v, cam));
                }
            }
        }
    }
}


void update_step(vec cam, double image[X][Y]) {
    make_proj(cam);
    for(int x = 0; x < X; x++) {
        for(int y = 0; y < Y; y++) {
            vec *v = hull[x][y];
            if(v != NULL) {
                int dir = proj[x][y]-image[x][y] > 0? 1 : -1;
                v->x -= -ALPHA*cam.x*dir;
                v->y -= -ALPHA*cam.y*dir;
                v->z -= -ALPHA*cam.z*dir;
            }
        }
    }
}


void update_all() {
    for(int w = 0; w < W; w++) {
        update_step(get_cam(w), images[w]);
    }
}


void write_projs(string path) {
    ofstream fout(path, ios::binary);
    for(int w = 0; w < W; w++) {
        make_proj(get_cam(w));
        for(int x = 0; x < X; x++) {
            for(int y = 0; y < Y; y++) {
                float f = proj[x][y];
                fout.write(reinterpret_cast<char*>(&f), sizeof(float));
            }
        }
    }
    fout.close();
}


void write_max_projs(string path) {
    ofstream fout(path, ios::binary);
    for(int x = 0; x < X; x++) {
        for(int y = 0; y < Y; y++) {
            for(int z = 0; z < Z; z++) {
                float f = max_proj[x][y][z];
                fout.write(reinterpret_cast<char*>(&f), sizeof(float));
            }
        }
    }
    fout.close();
}


void init_random() {
    for(int x = 0; x < X; x++) {
        for(int y = 0; y < Y; y++) {
            for(int z = 0; z < Z; z++) {
                if(max_proj[x][y][z] > 0) {
                    continue;
                }
                vec &v = norm[x][y][z];
                v = {1.0*rand()/RAND_MAX-0.5, 1.0*rand()/RAND_MAX-0.5, 1.0*rand()/RAND_MAX-0.5};
            }
        }
    }
}


int main() {
    read_images("raw_data.bin");
    for(int x = 0; x < X; x++) {
        for(int y = 0; y < Y; y++) {
            //cout << int(images[0][x][y]*100) << " ";
             cout << (images[0][x][y] > EPS? "##" : "  ");
        }
        cout << "\n";
    }
    init_random();
    for(int i = 0; i < 100; i++) {
        make_proj(get_cam(1));
        for(int x = 0; x < X; x++) {
            for(int y = 0; y < Y; y++) {
                cout << (proj[x][y] > EPS? "##" : "  ");
            }
            cout << endl;
        }
        cout << endl;

        make_max_proj();
        for(int j = 0; j < 1000; j++) {
            update_all();
        }
        make_proj(get_cam(1));
        for(int x = 0; x < X; x++) {
            for(int y = 0; y < Y; y++) {
                cout << (proj[x][y] > EPS? "##" : "  ");
            }
            cout << endl;
        }
        cout << endl;
    }
    write_projs("projs.bin");
    write_max_projs("max_projs.bin");
}