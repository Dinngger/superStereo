#include <iostream>
#include <fstream>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/registration/registration.h>
#include <Eigen/Dense>
#include <Eigen/Core>
#include "voxel_grid_covariance_omp_impl.hpp"
using namespace std;
using namespace pcl;

int main(int argc, char** argv) {
    if (argc < 3) {
        cout << "usage: pcl_file_name csv_file_name";
        return 0;
    }
    pcl::PointCloud<pcl::PointXYZ>::Ptr pcd_file(new pcl::PointCloud<pcl::PointXYZ>());
    pclomp::VoxelGridCovariance<PointXYZ> target_cells;
    pcl::io::loadPCDFile(argv[1], *pcd_file);
    target_cells.setLeafSize(1.5, 1.5, 1.5);
    target_cells.setInputCloud(pcd_file);
    // Initiate voxel structure.
    target_cells.filter(true);
    target_cells.saveLeaf(argv[2]);
    return 0;
}