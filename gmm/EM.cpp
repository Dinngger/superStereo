#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/filters/filter.h>
#include <pcl/visualization/cloud_viewer.h>
#include <Eigen/Eigen>
#include <Eigen/Dense>
#include <vector>
#include <string>
#include <sstream>

using namespace Eigen;
//初始化
void init_parm(int K, std::vector<double> &weight, Matrix3Xd &mean, std::vector<Matrix3d> &Cov, Matrix3Xd &Point)
{
    for (int i = 0; i < K; i++)
        weight[i] = 1 / (double)K;
    for (int i = 0; i < K; i++)
    {
        int n = rand()%Point.cols();
        mean(0,i) = Point(0,n);
        mean(1,i) = Point(1,n);
        mean(2,i) = Point(2,n);
    }
    for (int i = 0; i < K; i++)
        Cov[i] = Matrix3d::Random();
        // Cov[i]=Vector3d::Random()*Vector3d::Random().transpose();
}

//计算每次改变的|thetaK-thetaK+1|,看是否到达截止条件
bool cutoffed(Matrix3Xd mean1, Matrix3Xd mean2, std::vector<Matrix3d> &Cov1, std::vector<Matrix3d> &Cov2, double eps = 1e-5)
{
    double submax, temp;
    Matrix3Xd mean = (mean1 - mean2).cwiseAbs(); //取绝对值
    int num = Cov1.size();
    std::vector<Matrix3d> Cov(num);
    VectorXd max_cov(num);
    for (int i = 0; i < num; i++)
    {
        Cov[i] = (Cov1[i] - Cov2[i]).cwiseAbs();
        max_cov(i) = Cov[i].maxCoeff();
    }

    submax = mean.maxCoeff();
    temp = max_cov.maxCoeff();
    if (temp > submax)
        submax = temp;

    if (submax >= eps)
        return false;
    else
        return true;
}

//计算rjk
void E_step(const int K, const int N, Matrix<double, Dynamic, Dynamic> &rjk, std::vector<double> &weight, const Matrix3Xd &Point, Matrix3Xd &mean, std::vector<Matrix3d> &Cov)
{
    Matrix<double, Dynamic, Dynamic> ak_fai(N, K);
    //把点带入模型计算
    // int count = 0;
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < K; j++)
        {
            //行列式接近0,协方差矩阵接近奇异
            double detCov = abs(Cov[j].determinant());
            double fenmu = sqrt(pow(2 * M_PI, 3) * detCov);
            // if(fenmu!=fenmu)
            //     count++;
            // if(fenmu<1e-100)
            //     count++;
            double exp_coef = -0.5 * (Point.col(i) - mean.col(j)).transpose() * Cov[j].inverse() * (Point.col(i) - mean.col(j));
            // if (exp_coef!=exp_coef)
            // {
            //     count++;
            //     cout << exp_coef << endl;
            // }
            // if (exp_coef > 0)
            // {
            //     count++;
            //     cout << exp_coef << endl;
            // }
            double gaussP = std::exp(exp_coef) / fenmu;
            // if (fenmu < 1e-5)
            //     cout << "fenmu:" << j+1 << fenmu << ",";
            // if (exp_coef > 0)
            //     cout << "exp_coef:" << exp_coef << endl;
            // cout << "gaussP" << j+1 << ":" << gaussP << ",";
            // if (detCov < 1e-7)
            //     detCov = 1e-7;
            // if (gaussP != gaussP)
            //     count++;
            ak_fai(i, j) = weight[j] * gaussP;
            // if(ak_fai(i,j)!=ak_fai(i,j))
            //     count++;
                // ak_fai(i,j) = 1e-7;
        }
        // cout << endl;
    }
    // cout << "cout: " << count << endl;
    // cout << "ak_fai" << endl << ak_fai.block<3,5>(1000,0) << endl;

    //根据计算出来的值算点属于每个模型的概率
    VectorXd sum = ak_fai.transpose().colwise().sum();
    // cout << "sum size = " << sum.size() << endl;
    int flag = 0;
    for (int i = 0; i < N; i++)
    {
        flag = 0;
        for (int j = 0; j < K; j++)
        {
            rjk(i, j) = ak_fai(i, j) / sum(i);
        if(rjk(i,j)!=rjk(i,j)&&flag==0)
        {
            rjk(i,j) = 1;
            flag = 1;
            for(int k = 0; k < j; k++)
                rjk(i,k) = 0;
            continue;
        }
        if(flag == 1)
            rjk(i,j) = 0;
        }
        //sum(i)近似为0,ak_fai(i,j)不正确
    }
    // cout << "rjk1" << endl << rjk.block<10,5>(20000,0) << endl;
    // cout << "rjk2" << endl << rjk.block<4,5>(1000,0) << endl;
}

void M_step(const int K, const int N, Matrix<double, Dynamic, Dynamic> &rjk, std::vector<double> &weight, const Matrix3Xd &Point, Matrix3Xd &mean, std::vector<Matrix3d> &Cov)
{
    mean.setZero();
    for (int i = 0; i < K; i++)
        Cov[i].setZero();
    VectorXd Nk = rjk.colwise().sum();

    // for(int i = 0; i < K; i++)
    // if(Nk[i]!=Nk[i])
    //     Nk[i] = 1e-7;

    for (int i = 0; i < K; i++)
        weight[i] = Nk(i) / N;
    
    for (int i = 0; i < K; i++)
    for (int j = 0; j < N; j++)
    {
        mean.col(i) += Point.col(j) * rjk(j, i) / Nk(i);
    }

    for (int i = 0; i < K; i++)
        for (int j = 0; j < N; j++)
        {
            Cov[i] += (Point.col(j) - mean.col(i)) * (Point.col(j) - mean.col(i)).transpose() * rjk(j, i) / Nk(i);
        }
}
void EM_algo(Matrix3Xd &Point, const int K, const int N, int *map,std::vector<double>& weight, Matrix3Xd &mean, std::vector<Matrix3d> &Cov)
{
    const double eps = 1e-5;

    init_parm(K, weight, mean, Cov, Point); //初始化

    Matrix<double, Dynamic, Dynamic> rjk(N, K);
    Matrix3Xd mean_last;
    std::vector<Matrix3d> Cov_last;
    int count = 0;
    while (true)
    {
        count++;
        mean_last = mean;
        Cov_last = Cov;

        E_step(K, N, rjk, weight, Point, mean, Cov);
        M_step(K, N, rjk, weight, Point, mean, Cov);
        if (cutoffed(mean, mean_last, Cov, Cov_last,eps))
            break;
        // for (int i = 0; i < K; i++)
        // {
        //     cout << "协方差:" << endl << Cov[i] << endl;
        // }
    }
    cout << "迭代次数：" << count << endl;
    VectorXd::Index maxcol;
    for (int i = 0; i < N; i++)
    {
        rjk.row(i).maxCoeff(&maxcol);
        map[i] = maxcol;
    }
}

//根据索引拷贝点云
template<typename T1, typename T2>
void mycopyPointCloud(const pcl::PointCloud<T1> &cloud_in,
                    const std::vector<int> &indices,
                    pcl::PointCloud<T2> &cloud_out)
{
    cloud_out.points.resize(indices.size());
    cloud_out.header=cloud_in.header;
    cloud_out.width=static_cast<uint32_t>(indices.size());
    cloud_out.height   = 1;
    cloud_out.is_dense = cloud_in.is_dense;
    cloud_out.sensor_orientation_ = cloud_in.sensor_orientation_;
    cloud_out.sensor_origin_ = cloud_in.sensor_origin_;
    int N = indices.size();
    for(int i = 0; i < N; ++i)
    { 
      cloud_out[i].x = cloud_in[indices[i]].x;
      cloud_out[i].y = cloud_in[indices[i]].y;
      cloud_out[i].z = cloud_in[indices[i]].z;
    }
}

//计算每个高斯分布对应点云数目
void compute_num(const std::vector<int> &incices, std::vector<int> &num)
{
    int N = incices.size();
    int K = num.size();
    for(int i = 0; i < K; i++)
    {
        num[i] = 0;
    }
    for(int i = 0; i < N; i++)
    {
        num[incices[i]]++;
    }
}

//根据映射画出点云
template <typename PointT>
void my_view_cloud(const pcl::PointCloud<PointT>& cloud, std::vector<int> map, int N, int K)
{
    //统计各个高斯模型点的数目
    std::vector<int> num(K);
    compute_num(map,num);

    for(int i = 0; i < K; i++)
        cout << "num" << i+1 << ":" << num[i] << endl;
    
    //分别展示每部分的点云
    boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("my view"));
    viewer->setBackgroundColor(0, 0, 0);
    for(int i = 0; i < K; i++)
    {
        //生成这部分点云索引
        int Kgauss_num = num[i];
        std::vector<int> new_map(Kgauss_num);
        int count = 0;
        for(int j = 0; j < N; j++)
        {
            if(map[j] == i)
            {
                new_map[count++] = j;
            }
        }

        //根据索引拷贝点云
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr gauss_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
        mycopyPointCloud(cloud, new_map, *gauss_cloud);

        //显示点云
        std::stringstream ID;
        std::string IDResult;
        ID << i;
        ID >> IDResult;
        pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZRGB> color(gauss_cloud, rand()%255, rand()%255, rand()%255); // green
        viewer->addPointCloud<pcl::PointXYZRGB>(gauss_cloud, color, IDResult);

    }
    while (!viewer->wasStopped())
    {
        viewer->spinOnce(100);
        boost::this_thread::sleep(boost::posix_time::microseconds(100000));
    }
}

int main()
{
    //加载点云
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    if (pcl::io::loadPCDFile<pcl::PointXYZ>("highway4.pcd", *cloud) == -1)
    {
        PCL_ERROR("Couldn't read the file.\n");
        return (-1);
    }
    // if (pcl::io::loadPCDFile<pcl::PointXYZ>("bunny.pcd", *cloud) == -1)
    // {
    //     PCL_ERROR("Couldn't read the file.\n");
    //     return (-1);
    // }

    //移除无效点
    std::vector<int> indices;
    pcl::removeNaNFromPointCloud(*cloud, *cloud, indices);

    //GMM
    int K = 40;
    int N = cloud->points.size();
    Matrix3Xd Point(3, N);
    int map[N];
    std::vector<double> weight(K);
    Matrix3Xd mean(3, K);
    std::vector<Matrix3d> Cov(K);   
    for (int i = 0; i < N; i++)
    {
        Point(0, i) = cloud->points[i].x;
        Point(1, i) = cloud->points[i].y;
        Point(2, i) = cloud->points[i].z;
    }

    //有问题
    //EM算法构建地图
    EM_algo(Point, K, N, map, weight, mean, Cov);
    // for(int i=0;i<N;i++)
    //     if(map[i]!=0) cout << map[i] << endl;
    for(int i = 0; i < K; i++)
        cout << "weight" << i+1 << ": " << weight[i] << endl;

    //根据索引显示点云
    std::vector<int> mapn(N);
    for(int i = 0; i < N; i++)
    {
        mapn[i] = map[i];
    }
    my_view_cloud(*cloud, mapn, N, K);
    return 0;
}