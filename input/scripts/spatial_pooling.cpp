#include <string>
#include <fstream>
#include <cstring>
#include <cmath>
#include <cassert>
#include <algorithm>
#define rep(i, N) for(int i = 0; i < N; i++)
using namespace std;


const int A = 16,
		  N = 330,
		  M = 256,
		  K = 19,
		  L = 7;
const double PI = 4*atan(1);


pair<double, double> rotate_point(double x, double y, double angle) {
	y *= M;
	x -= M/2;
	y -= M/2;
	double rx = cos(angle)*x - sin(angle)*y, ry = sin(angle)*x + cos(angle)*y;
	x = rx + M/2;
	y = ry + M/2;
	y /= M;
	x = min(max(x, 0.0), M-1.0);
	return make_pair(x, y);
}


void rotate_hulls(float hulls[A][N][M][K], float out[A][N][M][K], int base) {
	rep(a, A) rep(i, N) rep(j, M) {
		out[a][i][j][0] = 1.0;
		for(int k = 1; k < K; k++) {
			out[a][i][j][k] = 0.0;
		}
	}
	rep(a, A) rep(i, N) rep(j, M-1) {
		if(hulls[a][i][j][0] == 1 || hulls[a][i][j+1][0] == 1) {
			continue;
		}
		if(abs(hulls[a][i][j][0] - hulls[a][i][j+1][0]) > 0.1) {
			continue;
		}
		double angle = 2*PI/A*(a-base);
		auto p1 = rotate_point(j, hulls[a][i][j][0], angle),
			 p2 = rotate_point(j+1, hulls[a][i][j+1][0], angle);
		for(int jj = p1.first; jj <= p2.first; jj++) {
			double d = p1.second + (p2.second-p1.second)*(jj-p1.first)/(p2.first-p1.first);
			if(d < out[a][i][jj][0]) {
				out[a][i][jj][0] = d;
				for(int k = 1; k < K; k++) {
					out[a][i][jj][k] = hulls[a][i][j][k];
				}
			}
		}
	}
}


float rotbuf[A][N][M][K];
float pools[N][M][K][L];
void spatial_pool_hulls(float hulls[A][N][M][K], float out[A][N][M][K]) {
	rep(a, A) rep(i, N) rep(j, M) rep(k, K) {
		out[a][i][j][k] = 0;
	}

	rep(a, A) {
		rotate_hulls(hulls, rotbuf, a);
		rep(i, N) rep(j, M) {
			if(hulls[a][i][j][0] == 1) {
				continue;
			}
			rep(l, L) rep(k, K) {
				pools[i][j][k][l] = 1;
			}
			rep(aa, A) {
				int maxd = 0;
				rep(l, L) {
					if(pools[i][j][0][maxd] < pools[i][j][0][l]) {
						maxd = l;
					}
				}
				if(rotbuf[aa][i][j][0] < pools[i][j][0][maxd]) {
					rep(k, K) {
						pools[i][j][k][maxd] = rotbuf[aa][i][j][k];
					}
				}
			}
			int num = 0;
			rep(l, L) {
				num += pools[i][j][0][l] != 1.0;
			}
			if(num == 0) {
				continue;
			}
			rep(k, K) {
				sort(pools[i][j][k], pools[i][j][k]+L);
				out[a][i][j][k] = pools[i][j][k][num/2];
			}
		}
	}
}


void read_hulls(string filename, float hulls[A][N][M][K]) {
	ifstream fin(filename, ios::binary);
	rep(a, A) rep(i, N) rep(j, M) rep(k, K) {
		fin.read(reinterpret_cast<char*>(&hulls[a][i][j][k]), sizeof(float));
	}
	fin.close();
}


void write_hulls(string filename, float hulls[A][N][M][K]) {
	ofstream fout(filename, ios::binary);
	rep(a, A) rep(i, N) rep(j, M) rep(k, K) {
		fout.write(reinterpret_cast<char*>(&hulls[a][i][j][k]), sizeof(float));
	}
	fout.close();
}


float hulls[A][N][M][K];
float out[A][N][M][K];
int main(int argc, char **argv) {
	read_hulls(argv[1], hulls);
	//rotate_hulls(hulls, out, 1);
	spatial_pool_hulls(hulls, out);
	write_hulls(argv[2], out);
}