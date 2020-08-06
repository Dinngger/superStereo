#include <iostream>
#include <omp.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <Eigen/Core>
#include <Eigen/Dense>

using namespace std;
using namespace cv;
using namespace Eigen;
Matrix3d K;
const double delta_z = 0.2;
Matrix<double, 8, 3> means_trans;
Matrix<double, 1, 8> weights;
Matrix<double, 8, 3> cov_trans;

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

double getGauss(Vector3d x, Vector3d mean, Matrix3d cov) {
    double res = exp(-0.5 * (x - mean).transpose() * cov.inverse() * (x - mean)) / sqrt(2 * M_PI * cov.determinant());
    if (res > 1)
        return 1;
    else if (res < 0 || res != res)
        return 0;
    else
        return res;
}

void drawGuass(Mat& img, double weight, const Matrix3d cam_r, const Vector3d cam_t) {
    #pragma omp parallel for num_threads(8)
    for (int i=0; i<img.rows; i++) {
        for (int j=0; j<img.cols; j++) {
            double temp = 0;
            for (double z=0; z<100; z+=delta_z) {
                Vector3d p = cam_r * back2XYZ(j, i, z) + cam_t;
                for (int k=0; k<8; k++) {
                    Vector3d mean = means_trans.block<1, 3>(0, k).transpose();
                    Matrix3d cov = cov_trans.block<1, 3>(0, k).transpose().asDiagonal();
                    temp += weight * getGauss(p, mean, cov) * delta_z / M_PI; // * weights(k) * getVoxel(j, i, z);
                }
            }
            img.at<float>(i, j) = temp * img.cols * img.rows;
        }
    }
}

int main() {
    Vector3d po(0, 0, 0);
    Vector3d px(1, 0, 0);
    Vector3d py(0, 1, 0);
    Vector3d pz(0, 0, 1);
    #define IMG_SIZE 360
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
    means_trans <<
        -3.5329654 , -5.9225087 ,  4.064147  ,
         0.9108581 ,  0.03187832, -5.9937406 ,
         0.89965934, -5.8209953 , -5.37284   ,
        -3.469027  , -6.086397  , -3.804792  ,
        -3.7339954 , -5.876941  ,  1.331353  ,
        -3.5435963 , -4.3732285 ,  5.7543445 ,
         0.9565428 ,  6.2580543 , -5.3496146 ,
        -3.4642053 , -6.051702  , -0.25389466;
    weights << 
        1.0990811e-03, 7.9775476e-01, 9.3191519e-02, 4.1610315e-03,
        4.6698420e-05, 9.0432900e-04, 9.8393098e-02, 4.3344432e-03;
    cov_trans << 
        2.69799097e-03, 3.58248204e-02, 8.08650076e-01,
        1.48512985e+02, 1.19271564e+01, 5.92903933e-03,
        1.52023926e+02, 5.07454714e-03, 1.02377936e-01,
        1.97830666e-02, 2.14319695e-02, 9.32703078e-01,
        4.63256811e-08, 4.81469715e-06, 1.11920929e-06,
        3.43041285e-03, 7.03896582e-01, 6.50034398e-02,
        1.57864166e+02, 5.14321169e-03, 9.96523350e-02,
        1.21598109e-03, 1.26696099e-02, 4.18638420e+00;
    namedWindow("img", CV_WINDOW_NORMAL);
    resizeWindow("img", sum.cols, sum.rows);
    while (true) {
        drawGuass(sum, 1e-7, cam_r, cam_t);
        // line(sum, project(cam_r * po + cam_t), project(cam_r * px + cam_t), 1, 1);
        // line(sum, project(cam_r * po + cam_t), project(cam_r * py + cam_t), 1, 1);
        // line(sum, project(cam_r * po + cam_t), project(cam_r * pz + cam_t), 1, 1);
        imshow("img", sum);
        cout << "center: " << sum.at<float>(IMG_SIZE/2, IMG_SIZE/2) << endl;
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