# ifndef MAP_CONSTRUCT_HPP
# define MAP_CONSTRUCT_HPP

#include <map>
#include <unordered_map>
#include <string>
#include <stdio.h>
#include <math.h>
#include <memory>
#include <iomanip>
#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include <fstream>
#include <limits>

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include "NDTmap.h"

// struct statistics_info {
//     std::vector<double> miu;
//     std::vector<double> sigma_inv;
// };

struct HashFunc {
	std::size_t operator()(const hash_code &key) const {
		using std::size_t;
		using std::hash;
 
		return ((hash<int>()(key.hash_code_x_)
			^ (hash<int>()(key.hash_code_y_) << 1)) >> 1)
			^ (hash<int>()(key.hash_code_z_) << 1);
	}
};

struct EqualKey
{
	bool operator() (const hash_code &lhs, const hash_code &rhs) const {
		return lhs.hash_code_x_  == rhs.hash_code_x_
			&& lhs.hash_code_y_ == rhs.hash_code_y_
			&& lhs.hash_code_z_  == rhs.hash_code_z_;
	}
};


typedef std::unordered_map<hash_code, pcl::PointCloud<pcl::PointXYZ>::Ptr, HashFunc, EqualKey> umap;
class mapConstruct {
 public:
    umap map_;
    bool first_;

    mapConstruct() {
        first_ = true;
    }


// NDTmap<double, std::string> ndtmap_;




    bool construct(hash_code input, pcl::PointXYZ p) {
        umap::const_iterator got = map_.find(input);
        if (got == map_.end()) {
            pcl::PointCloud<pcl::PointXYZ>::Ptr block(new pcl::PointCloud<pcl::PointXYZ>());
            block->points.push_back(p);
            map_[input] = block;
            return true;
        } else {
            got->second->points.push_back(p);
            return false;
        }
    }

    std::vector<double> statistics(pcl::PointCloud<pcl::PointXYZ>::Ptr input) {
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eigensolver;
        Eigen::Matrix3d eigen_val, eigen_vec;
        double mean_x, mean_y, mean_z;
        mean_x = 0;
        mean_y = 0;
        mean_z = 0;
        for (size_t i = 0; i < input->points.size(); i++) {
            mean_x += input->points[i].x;
            mean_y += input->points[i].y;
            mean_z += input->points[i].z;
        }
        mean_x /= input->points.size();
        mean_y /= input->points.size();
        mean_z /= input->points.size();

        std::vector<double> statistics;
        statistics.push_back(mean_x);
        statistics.push_back(mean_y);
        statistics.push_back(mean_z);

        double cov_xx = 0., cov_yy = 0., cov_zz = 0.;
        double cov_xy = 0., cov_xz = 0., cov_yz = 0.;
        for (int i = 0; i < input->points.size(); i++) {
            cov_xx += (input->points[i].x - mean_x) *
                    (input->points[i].x - mean_x);
            cov_xy += (input->points[i].x - mean_x) *
                    (input->points[i].y - mean_y);
            cov_xz += (input->points[i].x - mean_x) *
                    (input->points[i].z - mean_z);
            cov_yy += (input->points[i].y - mean_y) *
                    (input->points[i].y - mean_y);
            cov_yz += (input->points[i].y - mean_y) *
                    (input->points[i].z - mean_z);
            cov_zz += (input->points[i].z - mean_z) *
                    (input->points[i].z - mean_z);
        }
        Eigen::MatrixXd Cov(3, 3);
        Eigen::MatrixXd icov(3, 3);
        Cov << cov_xx, cov_xy, cov_xz, cov_xy, cov_yy, cov_yz, cov_xz, cov_yz,
            cov_zz;
        Cov /= input->points.size();

        eigensolver.compute(Cov);
        eigen_val = eigensolver.eigenvalues().asDiagonal();
        eigen_vec = eigensolver.eigenvectors();

        if (eigen_val(0, 0) < 0 || eigen_val(1, 1) < 0 || eigen_val(2, 2) <= 0) {
            std::cout << "eigen value calculate error!" << std::endl;
        }

        // Avoids matrices near singularities (eq 6.11)[Magnusson 2009]

        double min_covar_eigvalue = 0.02 * eigen_val(2, 2);
        if (eigen_val(0, 0) < min_covar_eigvalue) {
            eigen_val(0, 0) = min_covar_eigvalue;

            if (eigen_val(1, 1) < min_covar_eigvalue) {
                eigen_val(1, 1) = min_covar_eigvalue;
            }

            Cov = eigen_vec * eigen_val * eigen_vec.inverse();
        }
        // leaf.evals_ = eigen_val.diagonal ();

        icov = Cov.inverse();
        if (icov.maxCoeff() == std::numeric_limits<double>::infinity()
            || icov.minCoeff() == -std::numeric_limits<double>::infinity()) {
            std::cout << "sigma inverse calculate error!" << std::endl;
        }
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                statistics.push_back(icov(i, j));
            }
        }
        return statistics;
    }
};


#endif