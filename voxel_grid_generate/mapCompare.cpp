#include <iostream>
#include <map>
#include <vector>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <fstream>
using namespace std;
using namespace Eigen;

vector<Vector3d> a_means;
vector<Matrix3d> a_icovs;
Vector3d a_shift = Vector3d::Zero();
vector<Vector3d> b_means;
vector<Matrix3d> b_icovs;
Vector3d b_shift = Vector3d::Zero();

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
    Vector3d toVector3d() const {
        return Vector3d(elem.x, elem.y, elem.z);
    }
};

inline bool operator < (const voxelIdx& lhs, const voxelIdx& rhs) {
    return (lhs.elem.x < rhs.elem.x) || (lhs.elem.x == rhs.elem.x && lhs.elem.y < rhs.elem.y) || (lhs.elem.x == rhs.elem.x && lhs.elem.y == rhs.elem.y && lhs.elem.z < rhs.elem.z);
}


std::map<voxelIdx, uint32_t> a_idx, b_idx;

int main(int argc, char** argv) {
    if (argc < 4) {
        printf("usage: csv1 csv2 output\n");
        return 0;
    }
    fstream fa(argv[1], ios::in);
    fstream fb(argv[2], ios::in);
    fstream fo(argv[3], ios::out);
    fa.exceptions(fstream::failbit);
    fb.exceptions(fstream::failbit);
    fo.precision(15);
    if(!fa.is_open()) {
        printf("Failed to open %s;\n", argv[1]);
        return 0;
    }
    if(!fb.is_open()) {
        printf("Failed to open %s;\n", argv[2]);
        return 0;
    }
    if(!fo.is_open()) {
        printf("Failed to open %s;\n", argv[3]);
        return 0;
    }
    Vector3d map_center = Vector3d(-11278.500000, -2598.000000, 6.000000);
    for (int i=0; i<3; i++)
        fa >> a_shift(i);
    for (int i=0; i<3; i++)
        fb >> b_shift(i);
    for (int i=0; i<3; i++)
        fo << b_shift(i) << "\n";
    a_shift -= map_center;
    cout << "a_shift:\n" << a_shift << "\nb_shift:\n" << b_shift << "\n";
    uint32_t cnt = 0;
    while (true) {
        try {
            voxelIdx _idx;
            for (int i=0; i<3; i++)
                fa >> _idx.list[i];
            Vector3d mean;
            Matrix3d icov;
            for (int i=0; i<3; i++)
                fa >> mean(i);
            mean -= map_center;
            for (int i=0; i<3; i++)
                for (int j=0; j<3; j++)
                    fa >> icov(i, j);
            if (icov.norm() < 1e-3)
                continue;
            a_means.push_back(mean);
            a_icovs.push_back(icov);
            a_idx[_idx] = cnt;
            cnt++;
        } catch (fstream::failure& e) {
            if(fa.eof()) {
                break;
            } else {
                printf("Broken File a %s\n", e.what());
                return -1;
            }
        }
    }
    cout << "finished read file a. cnt=" << cnt << endl;
    cnt = 0;
    while (true) {
        try {
            voxelIdx _idx;
            for (int i=0; i<3; i++)
                fb >> _idx.list[i];
            Vector3d mean;
            Matrix3d icov;
            for (int i=0; i<3; i++)
                fb >> mean(i);
            for (int i=0; i<3; i++)
                for (int j=0; j<3; j++)
                    fb >> icov(i, j);
            if (icov.norm() < 1e-3)
                continue;
            b_means.push_back(mean);
            b_icovs.push_back(icov);
            b_idx[_idx] = cnt;
            cnt++;
        } catch (fstream::failure& e) {
            if(fb.eof()) {
                break;
            } else {
                printf("Broken File b %s\n", e.what());
                return -1;
            }
        }
    }
    cout << "finished read file b. cnt=" << cnt << endl;
    int found_cnt = 0;
    for (auto pair : a_idx) {
        Vector3d mean_a = a_means[pair.second];
        Matrix3d icov_a = a_icovs[pair.second];
        Vector3d ida = pair.first.toVector3d();
        Vector3d idb = (ida * 1.5 + a_shift - b_shift + Vector3d(0.75, 0.75, 0.75)) / 1.5;
        if (idb(0) < 0 || idb(1) < 0 || idb(2) < 0)
            continue;
        voxelIdx vidb(idb);
        if (b_idx.count(vidb) == 0) {
            // printf("not found voxel in map b\n");
            continue;
        }
        found_cnt++;
        uint32_t id = b_idx[vidb];
        Vector3d mean_b = b_means[id];
        Matrix3d icov_b = b_icovs[id];
        for (int i=0; i<3; i++)
            fo << vidb.list[i] << "," << vidb.list[i] << "\n";
        for (int i=0; i<3; i++)
            fo << mean_b(i) << "," << mean_a(i) << "\n";
        for (int i=0; i<3; i++)
            for (int j=0; j<3; j++)
                fo << icov_b(i, j) << "," << icov_a(i, j) << "\n";
        // printf("mean error: %f, icov error: %f\n",
        //     (mean_a - mean_b).norm(), (icov_a - icov_b).norm());
    }
    printf("found voxel: %d\n", found_cnt);
    fa.close();
    fb.close();
    return 0;
}