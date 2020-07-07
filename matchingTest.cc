#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <Eigen/Core>
#include <Eigen/Dense>

using namespace std;
using namespace cv;

Mat img, imgl, imgr;
float last_min_x = 0, last_min_y = 0;

void on_mouse(int event, int x, int y, int flags, void* param) {
    if (event == CV_EVENT_MOUSEMOVE) {
        if (abs(y - 960) > 760 || abs(x - 1500) > 1300)
            return;
        #define RECT_SIZE 15
        #define SREACH 100
        Rect rectl(x - RECT_SIZE, y - RECT_SIZE, RECT_SIZE * 2, RECT_SIZE * 2);
        Mat img_rectl = imgl(rectl);
        double min_norm = 1e10;
        float min_x = 0, min_y = 0;
        for (int i=-SREACH; i<SREACH*2+1; i++) {
            for (int j=-SREACH; j<SREACH*2+1; j++) {
                Rect rectr(x - RECT_SIZE + i, y - RECT_SIZE + j, RECT_SIZE*2, RECT_SIZE*2);
                Mat img_rectr = imgr(rectr);
                double res = norm(img_rectl, img_rectr, NORM_L2) + pow((i - last_min_x) / 255, 2) + pow((j - last_min_y) / 255, 2);
                if (res <= min_norm) {
                    min_norm = res;
                    min_x = i;
                    min_y = j;
                }
            }
        }
        last_min_x = min_x;
        last_min_y = min_y;
        float trans_f[][3] = {1, 0, -min_x, 0, 1, -min_y};
        Mat trans(2, 3, CV_32F, trans_f);
        Mat imgr_trans;
        warpAffine(imgr, imgr_trans, trans, Size(imgl.cols, imgl.rows));
        img = (imgl + imgr_trans) / 2;
        imshow("img", img);
        waitKey(1);
    }
}

double maskFunc(int col, int cols, int width) {
    if (col < cols / 2 - width / 2)
        return 1.0;
    else if (col > cols / 2 + width / 2)
        return 0.0;
    else
        return 1.0 - 1.0 * (col + width / 2 - cols / 2) / width;
}

int main(int argc, char* argv[])
{
    imgl = imread("/home/dinger/mine/Dataset/Classroom/im0.png");
    imgr = imread("/home/dinger/mine/Dataset/Classroom/im1.png");
    printf("size: %d, %d\n", imgl.rows, imgl.cols);

    // Convert Mat to float data type
    imgl.convertTo(imgl, CV_32FC3);
    imgr.convertTo(imgr, CV_32FC3);
    imgl /= 255;
    imgr /= 255;

    Mat maskl = Mat(Size(imgl.cols, imgl.rows), CV_32FC3);
    for (int i=0; i<imgl.cols; i++) {
        for (int j=0; j<imgl.rows; j++)
            maskl.at<Vec3f>(j, i) = Vec3f(maskFunc(i, imgl.cols, 200), maskFunc(i, imgl.cols, 200), maskFunc(i, imgl.cols, 200));
    }

    Mat maskr = Mat(Size(imgl.cols, imgl.rows), CV_32FC3, Vec3f(1.0f, 1.0f, 1.0f));
    maskr = maskr - maskl;
    // multiply(maskl, imgl, imgl);
    // multiply(maskr, imgr, imgr);
    img = (imgl + imgr) / 2;
    namedWindow("img", CV_WINDOW_NORMAL);
    resizeWindow("img", imgl.cols / 2, imgl.rows / 2);
    setMouseCallback("img", on_mouse);
    imshow("img", img);
    waitKey(0);
    return 0;
}