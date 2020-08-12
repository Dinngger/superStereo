#include <cstdint>
#include <cstdio>
#include <fstream>
#include <vector>
#include <map>
#include <exception>
#include <cmath>

using namespace std;

#define REL_AVG_WIDTH 10
#define REL_AVG_SHIFT 8
#define REL_AVG_MASK ((1 << REL_AVG_WIDTH) - 1)
#define ICOV_WIDTH 18
#define ICOV_SHIFT 7
#define ICOV_MASK ((1 << ICOV_WIDTH) - 1)
#define ICOV_DYNAMIC_SHIFT_WIDTH 2
#define CELL_X_SIZE 1.5
#define CELL_Y_SIZE 1.5
#define CELL_Z_SIZE 1.5
const float cellSize[3] = {CELL_X_SIZE, CELL_Y_SIZE, CELL_Z_SIZE};

uint32_t overflowVoxelCnt = 0;
uint32_t shiftCnt[(1 << ICOV_DYNAMIC_SHIFT_WIDTH)];

struct voxelInfo {
    float center[3];
    float avg[3];
    float icov[9];
};

uint32_t quantize(float data, uint32_t width, uint32_t shift) {
    uint32_t mask = (1 << width) - 1;
    float boundary = 1 << (width - 1);
    float shiftedData = data * (1 << shift);
    if (shiftedData >= 1.0 * boundary) {
        char errLog[128];
        sprintf(errLog, "Raw: %f, Shifted: %f, Width: %d, Shift: %d\n", data, shiftedData, width, shift);
        throw overflow_error(errLog);
    } else if(shiftedData < -1.0 * boundary) {
        char errLog[128];
        sprintf(errLog, "Raw: %f, Shifted: %f, Width: %d, Shift: %d\n", data, shiftedData, width, shift);
        throw underflow_error(errLog);
    }
    uint32_t fp = static_cast<uint32_t>(shiftedData) & mask;
    if (fp == 0) {
        if (data > 0) fp = 1;
        else fp = (-1) & mask;
    }
    return fp;
}

void packVoxelInfo(float* center, float* avg, float* icov, vector<uint64_t>& mem, vector<uint8_t>& shiftMem) {
    float maxIcov = icov[0];
    for (uint32_t i = 0; i < 9; i++) {
        if (icov[i] > maxIcov) {
            maxIcov = icov[i];
        }
    }
    uint32_t requiredIntWidth = ceil(log2(abs(maxIcov)));
    uint32_t shift = 0;
    if(requiredIntWidth > ICOV_WIDTH - ICOV_SHIFT - 1) {
        shift = requiredIntWidth - (ICOV_WIDTH - ICOV_SHIFT - 1);
        for (int i = 0; i < 9; i++) {
            icov[i] /= (1 << shift);
        }
        if (shift >= (1 << ICOV_DYNAMIC_SHIFT_WIDTH)) {
            printf("Overflow! Data: %f. Need %d int bits\n", maxIcov, requiredIntWidth);
            overflowVoxelCnt++;
            shift = (1 << ICOV_DYNAMIC_SHIFT_WIDTH) - 1;
        } else {
            shiftCnt[shift]++;
        }
    } else {
        shiftCnt[0]++;
    }
    shiftMem.push_back(shift);
    
    for (uint32_t row = 0; row < 3; row++) {
        uint64_t temp = 0xFFFFFFFFFFFFFFFFL;
        try {
            float refAvg = avg[row] - (center[row] - cellSize[row] / 2);
            temp = quantize(refAvg, REL_AVG_WIDTH, REL_AVG_SHIFT);
            for (int32_t i = 2; i >= 0; i--) {
                temp <<= ICOV_WIDTH;
                uint32_t fpIcov = quantize(icov[row * 3 + i], ICOV_WIDTH, ICOV_SHIFT);
                temp |= fpIcov;
                if (fpIcov == 0) {
                    printf("ERR: icov: %f\n", icov[row * 3 + i]);
                }
            }
        } catch(runtime_error& e) {
            printf("Error catched during voxel quantization: %s", e.what());
            return;
        }
        mem.push_back(temp);
    }
}

int32_t main(int32_t argc, char* argv[]) {
    if (argc != 3) {
        printf("Usage: mapPackage mapCSV binaryFilename\n");
        return -1;
    }
    char* mapFilename = argv[1];
    fstream file(mapFilename, ios_base::in);
    file.exceptions(fstream::failbit);
    if(!file.is_open()) {
        printf("Failed to open;\n");
    }
    for (uint32_t i = 0; i < 1 << ICOV_DYNAMIC_SHIFT_WIDTH; i++) {
        shiftCnt[i] = 0;
    }
    vector<uint32_t> idxList;
    vector<uint8_t> shiftList;
    vector<float> centerList;
    vector<uint64_t> infoList;
    vector<uint64_t>::iterator infoListIter;
    float mapLb[3];
    float mapShift[3];
    try {
        for (int i=0; i<3; i++)
            file >> mapShift[i];
    } catch (fstream::failure& e) {
        if(file.eof()) {
            printf("Incomplete File %s\n", e.what());
            return -1;
        } else {
            printf("Broken File %s\n", e.what());
            return -1;
        }
    }
    while(true) {
        uint32_t coordinate[3];
        float center[3];
        float avg[3];
        float icov[9];
        try {
            for (uint32_t i = 0; i < 3; i++) {
                file >> coordinate[i];
            }
            for (uint32_t i = 0; i < 3; i++) {
                center[i] = 1.5 * coordinate[i] + mapShift[i] + 0.75;
            }
            for (uint32_t i = 0; i < 3; i++) {
                file >> avg[i];
            }
            for (uint32_t i = 0; i < 9; i++) {
                file >> icov[i];
            } 
            if(idxList.size() == 0) {
                // Fill the map LB;
                for(uint32_t i = 0; i < 3; i++) {
                    mapLb[i] = center[i] - cellSize[i] / 2 - coordinate[i] * cellSize[i];
                }
            } else {
                // Validate the map LB
                for (uint32_t i = 0; i < 3; i++) {
                    float expLb = center[i] - cellSize[i] / 2 - coordinate[i] * cellSize[i];
                    float err = mapLb[i] - expLb;
                    if(abs(err) > 0.1) {
                        printf("Wrong LB!\n");
                    }
                }
            }
            packVoxelInfo(center, avg, icov, infoList, shiftList);
            idxList.insert(idxList.end(), coordinate, coordinate + 3);
            centerList.insert(centerList.end(), center, center + 3);
        } catch (fstream::failure& e) {
            if(file.eof()) {
                break;
            } else {
                printf("Broken File %s\n", e.what());
                return -1;
            }
        }
    }
    printf("Loaded %d voxels\n", int(idxList.size() / 3));
    printf("Map LB: (%f, %f, %f)\n", mapLb[0], mapLb[1], mapLb[2]);
    printf("%d Voxels overflowed\n", overflowVoxelCnt);
    printf("Shift Cnt: ");
    for (int i = 0; i < 1 << ICOV_DYNAMIC_SHIFT_WIDTH; i++) {
        printf("%d ", shiftCnt[i]);
    }
    printf("\n");
    fstream mapFile(argv[2], ios_base::out);
    mapFile << idxList.size() / 3 << endl;
    mapFile << REL_AVG_WIDTH << ' ' << REL_AVG_SHIFT << ' ' << ICOV_WIDTH << ' ' << ICOV_SHIFT << ' ' << ICOV_DYNAMIC_SHIFT_WIDTH << endl;
    mapFile << CELL_X_SIZE << ' ' << CELL_Y_SIZE << ' ' << CELL_Z_SIZE << endl;
    mapFile << mapLb[0] << ' ' << mapLb[1] << ' ' << mapLb[2] << endl;
    mapFile << 64;
    for (uint32_t i = mapFile.tellp(); i < 64; i++) {
        mapFile << ' ';
    }
    mapFile.write((char*)&idxList[0], idxList.size() * sizeof(uint32_t));
    mapFile.write((char*)&centerList[0], centerList.size() * sizeof(float));
    mapFile.write((char*)&shiftList[0], shiftList.size() * sizeof(uint8_t));
    mapFile.write((char*)&infoList[0], infoList.size() * sizeof(uint64_t));
    mapFile.close();
    return 0;
}