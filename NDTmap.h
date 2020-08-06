#ifndef NDT_MAP_HPP
#define NDT_MAP_HPP

#include <string>
#include <stdio.h>
#include <math.h>
#include <algorithm>
#include <memory>
#include <iomanip>
#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include <fstream>

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/io/pcd_io.h>

struct hash_code {
        uint32_t hash_code_x_;
        uint32_t hash_code_y_;
        uint32_t hash_code_z_;
};

// struct map_hash_code {
//         uint32_t hash_code_x_;
//         uint32_t hash_code_y_;
// };

template<class T1, class T2>
class NDTmap {
    // map<string, string> base32;
public:
    void initParam(pcl::PointCloud<pcl::PointXYZ>::ConstPtr globalmap);
    T2 encodeBin(T1 w, T1 left, T1 right, int step, bool mode);
    hash_code encode(T1 x, T1 y, T1 z);
    uint32_t encodeMap(T1 x, T1 y);
    T2 encodeBinMap(T1 w, T1 left, T1 right, int step);
    uint32_t BinStr2Dec(T2 binaryString);

    T1 encodeMidBin(T1 w, T1 left, T1 right, int step, bool mode);
    std::vector<T1> encodeMid(T1 x, T1 y, T1 z);

private:
    T1 cell_size_;
    int step_, step_z_;
    int step_map_;
    T1 min_x_, max_x_, min_y_, max_y_, min_z_, max_z_;
    T1 mid_;
};

template<class T1, class T2>
void NDTmap<T1, T2>::initParam(pcl::PointCloud<pcl::PointXYZ>::ConstPtr globalmap) {
    cell_size_ = 1.5;

    // pcl::PointCloud<pcl::PointXYZ>::Ptr globalmap(new pcl::PointCloud<pcl::PointXYZ>());
    // pcl::io::loadPCDFile(pcdFilename, *globalmap);
    min_x_ = DBL_MAX;
    max_x_ = -DBL_MAX;
    min_y_ = DBL_MAX;
    max_y_ = -DBL_MAX;
    min_z_ = DBL_MAX;
    max_z_ = -DBL_MAX;
    for (size_t i = 0; i < globalmap->points.size(); i++) {
        if (globalmap->points[i].x < min_x_) {
            min_x_ = globalmap->points[i].x;
        }
        if (globalmap->points[i].x > max_x_) {
            max_x_ = globalmap->points[i].x;
        }
        if (globalmap->points[i].y < min_y_) {
            min_y_ = globalmap->points[i].y;
        }
        if (globalmap->points[i].y > max_y_) {
            max_y_ = globalmap->points[i].y;
        }
        if (globalmap->points[i].z < min_z_) {
            min_z_ = globalmap->points[i].z;
        }
        if (globalmap->points[i].z > max_z_) {
            max_z_ = globalmap->points[i].z;
        }
    }
    std::cout.precision(15);
    
    T1 x_center = (max_x_ + min_x_)/2;
    T1 y_center = (max_y_ + min_y_)/2;
    T1 z_center = (max_z_ + min_z_)/2;

    T1 max_length = std::max(max_x_ - min_x_, max_y_ - min_y_);
    T1 length = 2 * cell_size_;
    step_ = step_z_ = 1;
    while (length < max_length) {
        length = 2 * length;
        step_++;
    }

    step_map_ = step_ - 4;
    min_x_ = x_center - length/2;
    max_x_ = x_center + length/2;
    min_y_ = y_center - length/2;
    max_y_ = y_center + length/2;

    length = 2 * cell_size_;
    max_length = max_z_ - min_z_;
    while (length < max_length) {
        length = 2 * length;
        step_z_++;
    }

    min_z_ = z_center - length/2;
    max_z_ = z_center + length/2;

    std::cout << "min_x: " << min_x_ << std::endl;
    std::cout << "max_x: " << max_x_ << std::endl;
    std::cout << "min_y: " << min_y_ << std::endl;
    std::cout << "max_y: " << max_y_ << std::endl;
    std::cout << "min_z: " << min_z_ << std::endl;
    std::cout << "max_z: " << max_z_ << std::endl;

    std::cout << "step: " << step_ << std::endl;
    std::cout << "step_z_: " << step_z_ << std::endl;
}

template<class T1, class T2>
hash_code NDTmap<T1, T2>::encode(T1 x, T1 y, T1 z) {
    T2 s_x = "", s_y = "", s_z = "";
    T2 s = "", ss = "";

    s_x = encodeBin(x, min_x_, max_x_, 1, true);
    s_y = encodeBin(y, min_y_, max_y_, 1, true);
    s_z = encodeBin(z, min_z_, max_z_, 1, false);
    reverse(s_x.begin(), s_x.end());
    reverse(s_y.begin(), s_y.end());
    reverse(s_z.begin(), s_z.end());

    // std::cout << "s x: " << s_x << std::endl;
    // std::cout << "s z: " << s_z << std::endl;

    hash_code code_struct;
    uint32_t code = BinStr2Dec(s_x);
    code_struct.hash_code_x_ = code;
    code = BinStr2Dec(s_y);
    code_struct.hash_code_y_ = code;
    code = BinStr2Dec(s_z);
    code_struct.hash_code_z_ = code;

    return code_struct;
}

template<class T1, class T2>
uint32_t NDTmap<T1, T2>::encodeMap(T1 x, T1 y) {
    T2 s_x = "", s_y = "";
    T2 s = "";

    s_x = encodeBinMap(x, min_x_, max_x_, 1);
    s_y = encodeBinMap(y, min_y_, max_y_, 1);
    reverse(s_x.begin(), s_x.end());
    reverse(s_y.begin(), s_y.end());

    s = s_x + s_y;

    uint32_t code = BinStr2Dec(s);
    return code;
}

template<class T1, class T2>
T2 NDTmap<T1, T2>::encodeBin(T1 w, T1 left, T1 right, int step, bool mode) {
    // int step = step_
    if (mode) {
        if (step > step_) {
            return "";
        }
        double mid = left / 2 + right / 2;

        if (w > left && w <= mid) {
            return encodeBin(w, left, mid, step+1, mode) + "0";
            // if (step == max_step_) {
            // }
        } else {
            return encodeBin(w, mid, right, step+1, mode) + "1";
        }
    } else {
        // std::cout << "z: " << w << std::endl;
        // std::cout << "left: " << left << std::endl;
        // std::cout << "right: " << right << std::endl;
        if (step > step_z_) {
            return "";
        }
        double mid = left / 2 + right / 2;

        if (w > left && w <= mid) {
            return encodeBin(w, left, mid, step+1, mode) + "0";
            // if (step == max_step_) {
            // }
        } else {
            return encodeBin(w, mid, right, step+1, mode) + "1";
        }
    }
}

template<class T1, class T2>
T2 NDTmap<T1, T2>::encodeBinMap(T1 w, T1 left, T1 right, int step) {
    if (step > step_map_) {
        return "";
    }
    double mid = left / 2 + right / 2;

    if (w > left && w <= mid) {
        return encodeBinMap(w, left, mid, step+1) + "0";
        // if (step == max_step_) {
        // }
    } else {
        return encodeBinMap(w, mid, right, step+1) + "1";
    }
}

template<class T1, class T2>
uint32_t NDTmap<T1, T2>::BinStr2Dec(T2 binaryString) {
    uint32_t parseBinary = 0;
    for (int i = binaryString.length() - 1; i >= 0; --i) {
        if (binaryString[i] == '1') {
            parseBinary += 1 << (binaryString.length() - 1 - i);
        }
    }
    return parseBinary;
}

std::string int2Binstr(uint32_t n) {
    std::string res_string = "";
    for (size_t i = 0; i < 6; i++) {
        res_string += (n & (1 << (5 - i))) ? "1" : "0";
    }
    return res_string;
}



template<class T1, class T2>
T1 NDTmap<T1, T2>::encodeMidBin(T1 w, T1 left, T1 right, int step, bool mode) {
    if (mode) {
        if (step > step_) {
            mid_ = left * 0.5 + right *0.5;
            return mid_;
        }
        mid_ = left * 0.5 + right *0.5;

        if (w > left && w <= mid_) {
            mid_ = encodeMidBin(w, left, mid_, step+1, true);
            // if (step == max_step_) {
            // }
        } else if (w > mid_ && w <= right) {
            mid_ = encodeMidBin(w, mid_, right, step+1, true);
        }
    } else {
        if (step > step_z_) {
            mid_ = left * 0.5 + right *0.5;
            return mid_;
        }
        mid_ = left * 0.5 + right *0.5;

        if (w > left && w <= mid_) {
            mid_ = encodeMidBin(w, left, mid_, step+1, false);
            // if (step == max_step_) {
            // }
        } else if (w > mid_ && w <= right) {
            mid_ = encodeMidBin(w, mid_, right, step+1, false);
        }
    }
}

template<class T1, class T2>
std::vector<T1> NDTmap<T1, T2>::encodeMid(T1 x, T1 y, T1 z) {
    double mid_x, mid_y, mid_z;

    mid_x = encodeMidBin(x, min_x_, max_x_, 1, true);
    mid_y = encodeMidBin(y, min_y_, max_y_, 1, true);
    mid_z = encodeMidBin(z, min_z_, max_z_, 1, false);

    std::vector<double> mid;
    mid.push_back(mid_x);
    mid.push_back(mid_y);
    mid.push_back(mid_z);
    return mid;
}


#endif