#include <iostream>
#include <omp.h>
#include <map>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <fstream>

using namespace std;
using namespace cv;
using namespace Eigen;
Matrix3d K;
const double delta_z = 0.1;
vector<Vector3d> means;
vector<Matrix3d> covs;
Vector3d shift = Vector3d::Zero();

#define GEOHASH_X_WIDTH 7
#define GEOHASH_Y_WIDTH 7
#define GEOHASH_Z_WIDTH 4
#define GEOHASH_X_MASK ((1 << GEOHASH_X_WIDTH) - 1)
#define GEOHASH_Y_MASK ((1 << GEOHASH_Y_WIDTH) - 1)
#define GEOHASH_Z_MASK ((1 << GEOHASH_Z_WIDTH) - 1)

union voxelIdx {
    struct {
        uint32_t x;
        uint32_t y;
        uint32_t z;
    } elem;
    uint32_t list[3];
    voxelIdx() {}
    voxelIdx(uint32_t x, uint32_t y, uint32_t z) {
        elem.x = x;
        elem.y = y;
        elem.z = z;
    }
    voxelIdx(Vector3d p) {
        for (int i=0; i<3; i++)
            list[i] = p(i);
    }
    bool operator == (const voxelIdx& rhs) {
        return elem.x == rhs.elem.x && elem.y == rhs.elem.y && elem.z == rhs.elem.z;
    }
    uint32_t concat() const {
        return ((elem.x & GEOHASH_X_MASK) << (GEOHASH_Y_WIDTH + GEOHASH_Z_WIDTH)) | 
            ((elem.y & GEOHASH_Y_MASK) << GEOHASH_Z_WIDTH) | 
            (elem.z & GEOHASH_Z_MASK);
    }
    void unpack(uint32_t code) {
        elem.x = (code >> (GEOHASH_Y_WIDTH + GEOHASH_Z_WIDTH)) & GEOHASH_X_MASK;
        elem.y = (code >> GEOHASH_Z_WIDTH) & GEOHASH_Y_MASK;
        elem.z = code & GEOHASH_Z_MASK;
    }
    voxelIdx operator - (const voxelIdx& rhs) {
        voxelIdx ret;
        for (uint32_t i = 0; i < 3; i++) {
            ret.list[i] = list[i] - rhs.list[i];
        }
        return ret;
    }
};

inline bool operator < (const voxelIdx& lhs, const voxelIdx& rhs) {
    return (lhs.elem.x < rhs.elem.x) || (lhs.elem.x == rhs.elem.x && lhs.elem.y < rhs.elem.y) || (lhs.elem.x == rhs.elem.x && lhs.elem.y == rhs.elem.y && lhs.elem.z < rhs.elem.z);
}

template <int GEOHASH_WIDTH>
uint32_t coordinate2Geohash(float x, float lb, float ub) {
	uint32_t tempGeohash = 0;
    float sepPoint;
    for(uint8_t i = 0; i < GEOHASH_WIDTH; i++) {
        tempGeohash <<= 1;
        sepPoint = (lb + ub) / 2;
        if (x >= sepPoint) {
            tempGeohash |= 1;
            lb = sepPoint;
        } else {
            ub = sepPoint;
        }
    }
    return tempGeohash;
}

std::map<voxelIdx, uint32_t> idx;

Vector3d back2XYZ(double x0, double y0, double z) {
    Vector3d x;
    x(0) = z / K(0, 0) * (x0 - K(0, 2));
    x(1) = z / K(1, 1) * (y0 - K(1, 2));
    x(2) = z;
    return x;
}

Point2d project(Vector3d p) {
    if (p(2) < 0)
        return Point2d(-1, -1);
    p = K * p;
    p /= p(2);
    return Point2d(p(0), p(1));
}

double getVoxel(double x0, double y0, double z) {
    Vector3d dx = back2XYZ(x0, y0, z) - back2XYZ(x0 - 1, y0 - 1, z);
    Vector3d dx2 = back2XYZ(x0, y0, z-delta_z) - back2XYZ(x0 - 1, y0 - 1, z-delta_z);
    return abs((dx(0)*dx(1) + dx2(0)*dx2(1)) * delta_z / 2);
}

double getGauss(Vector3d x, Vector3d mean, Matrix3d cov, double weight) {
    return weight * exp(-0.5 * (x - mean).transpose() * cov * (x - mean)) / sqrt(2 * M_PI / cov.determinant());
}

void drawGuass(Mat& img, double weight, const Matrix3d cam_r, const Vector3d cam_t) {
    int cnt[img.rows] = {0,};
    #pragma omp parallel for num_threads(8)
    for (int i=0; i<img.rows; i++) {
        int cnti = 0;
        for (int j=0; j<img.cols; j++) {
            double temp = 0;
            for (double z=0; z<100; z+=delta_z) {
                Vector3d p = cam_r * back2XYZ(j, i, z) + cam_t;
                Vector3d idp = (p - shift) / 1.5;
                if (idx.count(voxelIdx(idp)) == 0)
                    continue;
                cnti++;
                uint32_t id = idx[voxelIdx(idp)];
                Vector3d mean = means[id];
                Matrix3d cov = covs[id];
                temp += getGauss(p, mean, cov, weight) * delta_z / M_PI; // * getVoxel(j, i, z);
            }
            img.at<float>(i, j) = temp * img.cols * img.rows;
        }
        cnt[i] = cnti;
    }
    int sumcnt = 0;
    for (int i=0; i<img.rows; i++)
        sumcnt += cnt[i];
    cout << "cnt: " << sumcnt << endl;
}

int main() {
    fstream f("../ndtmap.csv", ios::in);
    f.exceptions(fstream::failbit);
    if(!f.is_open()) {
        printf("Failed to open;\n");
    }
    uint32_t cnt = 0;
    while (true) {
        try {
            voxelIdx _idx;
            for (int i=0; i<3; i++)
                f >> _idx.list[i];
            Vector3d mean;
            Matrix3d cov;
            Vector3d _shift;
            for (int i=0; i<3; i++)
                f >> _shift(i);
            _shift = _shift - 1.5 * Vector3d(_idx.elem.x, _idx.elem.y, _idx.elem.z) - Vector3d(0.75, 0.75, 0.75);
            if ((_shift - shift).norm() > 0.01)
                printf("none equal shift!\n");
            shift = _shift;
            for (int i=0; i<3; i++)
                f >> mean(i);
            for (int i=0; i<3; i++)
                for (int j=0; j<3; j++)
                    f >> cov(i, j);
            means.push_back(mean);
            covs.push_back(cov);
            idx[_idx] = cnt;
            cnt++;
        } catch (fstream::failure& e) {
            if(f.eof()) {
                break;
            } else {
                printf("Broken File %s\n", e.what());
                return -1;
            }
        }
    }
    cout << "finished read file. cnt=" << cnt << endl;
    Vector3d po(0, 0, 0);
    Vector3d px(1, 0, 0);
    Vector3d py(0, 1, 0);
    Vector3d pz(0, 0, 1);
    #define IMG_SIZE 720
    K <<    IMG_SIZE, 0, IMG_SIZE/2,
            0, IMG_SIZE, IMG_SIZE/2,
            0, 0, 1;
    Matrix3d cam_r = Eigen::Quaterniond(
        Eigen::AngleAxisd(M_PI / 2, Eigen::Vector3d(0, -1, 0)) *
        Eigen::AngleAxisd(M_PI / 2, Eigen::Vector3d(0, 0, 1)) *
        Eigen::AngleAxisd(M_PI*3/4, Eigen::Vector3d(0, -1, 0))
        ).matrix();
    Vector3d cam_t = cam_r * Vector3d(10, -5, -50);
    cam_r *= Eigen::Quaterniond(Eigen::AngleAxisd(M_PI / 24, Eigen::Vector3d(0, -1, 0))).matrix();
    Mat sum = Mat(Size(IMG_SIZE, IMG_SIZE), CV_32F, 0.0);
    namedWindow("img", CV_WINDOW_NORMAL);
    resizeWindow("img", sum.cols, sum.rows);
    while (true) {
        drawGuass(sum, 1e-7, cam_r, cam_t);
        // line(sum, project(cam_r.transpose() * po - cam_t), project(cam_r.transpose() * px - cam_t), 1, 1);
        // line(sum, project(cam_r.transpose() * po - cam_t), project(cam_r.transpose() * py - cam_t), 1, 1);
        // line(sum, project(cam_r.transpose() * po - cam_t), project(cam_r.transpose() * pz - cam_t), 1, 1);
        // double min, max;
        // minMaxLoc(sum, &min, &max);
        // if (max > 0)
        //     sum /= max;
        imshow("img", sum);
        int key = waitKey(0);
        switch (key)
        {
        case 'q':
            cout << endl;
            return 0;
            break;
        case 'a':
            cam_t += cam_r * Vector3d(-1, 0, 0);
            break;
        case 'd':
            cam_t += cam_r * Vector3d(1, 0, 0);
            break;
        case 's':
            cam_t += cam_r * Vector3d(0, 0, -1);
            break;
        case 'w':
            cam_t += cam_r * Vector3d(0, 0, 1);
            break;
        case 'r':
            cam_t += cam_r * Vector3d(0, -1, 0);
            break;
        case 'f':
            cam_t += cam_r * Vector3d(0, 1, 0);
            break;
        case 'j':
            cam_r = cam_r * Eigen::Quaterniond(Eigen::AngleAxisd(M_PI / 24, Eigen::Vector3d(0, -1, 0))).matrix();
            break;
        case 'l':
            cam_r = cam_r * Eigen::Quaterniond(Eigen::AngleAxisd(M_PI / 24, Eigen::Vector3d(0, +1, 0))).matrix();
            break;
        case 'u':
            cam_r = cam_r * Eigen::Quaterniond(Eigen::AngleAxisd(M_PI / 24, Eigen::Vector3d(0, 0, -1))).matrix();
            break;
        case 'o':
            cam_r = cam_r * Eigen::Quaterniond(Eigen::AngleAxisd(M_PI / 24, Eigen::Vector3d(0, 0, 1))).matrix();
            break;
        case 'i':
            cam_r = cam_r * Eigen::Quaterniond(Eigen::AngleAxisd(M_PI / 24, Eigen::Vector3d(1, 0, 0))).matrix();
            break;
        case 'k':
            cam_r = cam_r * Eigen::Quaterniond(Eigen::AngleAxisd(M_PI / 24, Eigen::Vector3d(-1, 0, 0))).matrix();
            break;
        case 'b':
            imwrite("../open3d_output3.png", sum * 255);
            cout << "finished save image~\n";
            break;
        default:
            break;
        }
    }
    return 0;
}