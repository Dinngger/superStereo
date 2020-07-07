#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <Eigen/Core>
#include <Eigen/Dense>

using namespace std;
using namespace cv;

double maskFunc(int col, int x, int width) {
    if (col < x - width / 2)
        return 1.0;
    else if (col > x + width / 2)
        return 0.0;
    else
        return 1.0 - 1.0 * (col + width / 2 - x) / width;
}

Mat combine(Mat imgl, Mat imgr, int x) {
    Mat maskl = Mat(Size(imgl.cols, imgl.rows), CV_32FC3);
    for (int i=0; i<imgl.cols; i++) {
        for (int j=0; j<imgl.rows; j++)
            maskl.at<Vec3f>(j, i) = Vec3f(maskFunc(i, x, 200), maskFunc(i, x, 200), maskFunc(i, x, 200));
    }

    Mat maskr = Mat(Size(imgl.cols, imgl.rows), CV_32FC3, Vec3f(1.0f, 1.0f, 1.0f));
    maskr = maskr - maskl;
    Mat maskedl, maskedr;
    multiply(maskl, imgl, maskedl);
    multiply(maskr, imgr, maskedr);
    return maskedl + maskedr;
}

Mat img, imgl, imgr;
float last_min_x = 0, last_min_y = 0;

void on_mouse(int event, int x, int y, int flags, void* param) {
    if (event == CV_EVENT_MOUSEMOVE) {
        if ((abs(y - img.rows / 2) > (img.rows / 2 - 200)) || (abs(x - img.cols / 2) > (img.cols / 2 - 200)))
            return;
        #define RECT_SIZE 10
        #define SREACH 70
        Rect rectl(x - RECT_SIZE, y - RECT_SIZE, RECT_SIZE * 2, RECT_SIZE * 2);
        Mat img_rectl = imgl(rectl);
        double min_norm = 1e10;
        float min_x = 0, min_y = 0;
        for (int i=-SREACH; i<SREACH*2+1; i++) {
            for (int j=-SREACH; j<SREACH*2+1; j++) {
                Rect rectr(x - RECT_SIZE + i, y - RECT_SIZE + j, RECT_SIZE*2, RECT_SIZE*2);
                Mat img_rectr = imgr(rectr);
                double res = norm(img_rectl, img_rectr, NORM_L2) + pow((i - last_min_x) / 10, 2) + pow((j - last_min_y) / 10, 2);
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
        img = combine(imgl, imgr_trans, x);
        imshow("img", img);
        waitKey(1);
    }
}

int main(int argc, char* argv[])
{
    imgl = imread("../data/im0.png");
    imgr = imread("../data/im1.png");
    resize(imgl, imgl, Size(imgl.cols/2, imgl.rows/2));
    resize(imgr, imgr, Size(imgr.cols/2, imgr.rows/2));
    printf("size: %d, %d\n", imgl.rows, imgl.cols);

    // Convert Mat to float data type
    imgl.convertTo(imgl, CV_32FC3);
    imgr.convertTo(imgr, CV_32FC3);
    imgl /= 255;
    imgr /= 255;

    img = combine(imgl, imgr, imgl.cols / 2);

    namedWindow("img", CV_WINDOW_NORMAL);
    resizeWindow("img", imgl.cols, imgl.rows);
    setMouseCallback("img", on_mouse);
    imshow("img", img);
    waitKey(0);
    return 0;
}