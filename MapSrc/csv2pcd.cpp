#include <fstream>
#include <pcl/io/pcd_io.h>

using namespace pcl;
using namespace std;

int32_t main(int32_t argc, char* argv[]) {
    PointCloud<PointXYZ>::Ptr cloud(new PointCloud<PointXYZ>);
    fstream csv(argv[1], ios_base::in);
    char lalala;
    float x, y, z;
    while(csv >> x >> lalala >> y >> lalala >> z) {
        cloud->push_back(PointXYZ(x, y, z));
    }
    io::savePCDFile(argv[2], *cloud);
    return 0;
}