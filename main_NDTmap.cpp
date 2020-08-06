#include "NDTmap.h"
#include "map_construct.hpp"
#include <iostream>
#include <fstream>
#include <dirent.h>

typedef std::unordered_map<uint32_t, std::vector<hash_code>> umap1;
pcl::PointCloud<pcl::PointXYZ>::Ptr globalmap(new pcl::PointCloud<pcl::PointXYZ>());

void getAllFiles(char* file_dir) {
    DIR* dir = opendir(file_dir);
    struct dirent *dirp;
    pcl::PointCloud<pcl::PointXYZ>::Ptr pcd_file(new pcl::PointCloud<pcl::PointXYZ>());
    while ((dirp = readdir(dir)) != nullptr) {
        if (dirp->d_type == DT_REG) {
            printf("%s\n", dirp->d_name);
            std::string full_name = std::string(file_dir) + std::string(dirp->d_name);
            pcl::io::loadPCDFile(full_name, *pcd_file);
            *globalmap = *globalmap + *pcd_file;
        }
    }
}

int main(int argc, char **argv) {
    std::ofstream ofs, ofs1;
    ofs.open("/home/dinger/mine/superStereo/ndtmap.csv", std::ios::out);
    ofs.precision(15);

    // ofs1.open("/home/dinger/NINESQUARE_NDT_MAP.csv", std::ios::out);
    // ofs1.precision(15);

    umap1 map_iv;

    char* pcd_dir;
    // if (argc > 1) {
    //     pcd_filename = argv[1];
    //     std::cout << "argv 1: " << argv[1] << std::endl;
    // } else {
    //     pcd_dir = "/media/achao/XC_MAP/nine_square/";
    // }
    pcd_dir = "../pcd_data/";
    getAllFiles(pcd_dir);

    for (size_t i = 0; i < globalmap->points.size(); i++) {
        globalmap->points[i].x = globalmap->points[i].x;
        globalmap->points[i].y = globalmap->points[i].y;
    }

    NDTmap<double, std::string> ndtmap_;
    std::unique_ptr<mapConstruct> map_hash;
    map_hash.reset(new mapConstruct());
    // NDTmap<double, std::string> ndtmap_;
    ndtmap_.initParam(globalmap);
    hash_code temp_code_struct, map_code;

    std::cout << "globalmap size: " << globalmap->points.size() << std::endl;


    for (size_t i = 0; i < globalmap->points.size(); i++) {
        temp_code_struct = ndtmap_.encode(globalmap->points[i].x, globalmap->points[i].y, globalmap->points[i].z);
        // std::cout << "xyz: " << globalmap->points[i].x << " " << globalmap->points[i].y << " " << globalmap->points[i].z << std::endl;
        // pcl::PointXYZ p(globalmap->points[i].x, globalmap->points[i].y, globalmap->points[i].z);
        // std::cout << "temp code x: " << temp_code_struct.hash_code_x_ << std::endl;
        // std::cout << "temp code y: " << temp_code_struct.hash_code_y_ << std::endl;
        // std::cout << "temp code z: " << temp_code_struct.hash_code_z_ << std::endl;
        bool temp = map_hash->construct(temp_code_struct, globalmap->points[i]);
    }

    for (auto iter = map_hash->map_.begin(); iter != map_hash->map_.end(); iter++) {
        if (iter->second->points.size() > 10) {
            ofs << iter->first.hash_code_x_ << std::endl;
            ofs << iter->first.hash_code_y_ << std::endl;
            ofs << iter->first.hash_code_z_ << std::endl;

            std::vector<double> mid = ndtmap_.encodeMid(iter->second->points[0].x, iter->second->points[0].y, iter->second->points[0].z);
            ofs << mid[0] << std::endl;
            ofs << mid[1] << std::endl;
            ofs << mid[2] << std::endl;

            std::vector<double> statistics = map_hash->statistics(iter->second);
            for (size_t i = 0; i < statistics.size(); i++) {
                ofs << statistics[i] << std::endl;
            }

            uint32_t map_code1 = ndtmap_.encodeMap(iter->second->points[0].x, iter->second->points[0].y);
            umap1::iterator got = map_iv.find(map_code1);
            if (got == map_iv.end()) {
                std::vector<hash_code> temp_code_vec;
                temp_code_vec.push_back(iter->first);
                map_iv[map_code1] = temp_code_vec;
            } else {
                got->second.push_back(iter->first);
            }
            // std::vector<double> cov = map
        }
    }

    // for (auto iter = map_iv.begin(); iter != map_iv.end(); iter++) {
    //     ofs1 << iter->first << std::endl;
    //     ofs1 << iter->second.size() << std::endl;
    //     for (size_t i = 0; i < iter->second.size(); i++) {
    //         ofs1 << iter->second[i].hash_code_x_ << std::endl;
    //         ofs1 << iter->second[i].hash_code_y_ << std::endl;
    //         ofs1 << iter->second[i].hash_code_z_ << std::endl;
    //     }
    // }

    // ndtmap_.encode();
}