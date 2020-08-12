#include "mapLoad.h"
#include <cstdint>
#include <cstdio>
#include <map>
#include <fstream>
#include <cstring>
#include <vector>
#include <pcl/io/pcd_io.h>
#include "geoHashOp.h"

using namespace std;
using namespace pcl;

void loadMap(const char* filename, mapInfoType& mapInfo) {
    fstream file(filename, ios_base::in);
    uint32_t voxelNum;
    uint32_t avgWidth, avgShift, icovWidth, icovShift, dynIcovShift;
    uint32_t startPos;
    file >> voxelNum;
    printf("Voxel Num: %d\n", voxelNum);
    file >> avgWidth >> avgShift >> icovWidth >> icovShift >> dynIcovShift;
    for (uint32_t i = 0; i < 3; i++) {
        float s;
        file >> s;
        if(s != cellSize[i]) {
            throw runtime_error("Map version does not match the loader.\n");
        }
    }
    for (uint32_t i = 0; i < 3; i++) {
        file >> mapInfo.mapLb[i];
    }
    printf("Map LB: (%f, %f, %f)\n", mapInfo.mapLb[0], mapInfo.mapLb[1], mapInfo.mapLb[2]);
    if (avgWidth != REL_AVG_WIDTH || avgShift != REL_AVG_SHIFT || icovWidth != ICOV_WIDTH || icovShift != ICOV_SHIFT || dynIcovShift != ICOV_DYNAMIC_SHIFT) {
        throw runtime_error("Map version does not match the loader.\n");
    }
    file >> startPos;
    file.seekg(startPos);
    uint32_t* idxList = new uint32_t[voxelNum * 3];
    mapInfo.center = new float[voxelNum * 3];
    mapInfo.voxelInfo = new uint64_t[voxelNum * 3];
    mapInfo.dynShift = new uint8_t[voxelNum];
    file.read((char*)idxList, voxelNum * 3 * sizeof(uint32_t));
    file.read((char*)mapInfo.center, voxelNum * 3 * sizeof(float));
    file.read((char*)mapInfo.dynShift, voxelNum);
    file.read((char*)mapInfo.voxelInfo, voxelNum * 3 * sizeof(uint64_t));
    uint32_t* iter = idxList;
    for (uint32_t i = 0; i < voxelNum; i++) {
        voxelIdx temp;
        memcpy(temp.list, iter, 3 * sizeof(uint32_t));
        iter += 3;
        mapInfo.idx.insert(make_pair(temp, i));
    }
    delete idxList;
}


void packMap(
    voxelIdx* voxelIdxList, uint32_t voxelNum, 
    mapInfoType& mapInfo, 
    uint64_t* metadata, uint32_t* metadataLen, uint32_t* metadataByte,
    uint64_t* voxelMem, uint32_t* voxelLen, uint32_t* voxelByte,
    float* mapCenter
) {
    if(voxelNum > VOXEL_NUM_LIMIT) {
        printf("Too many voxel! Get %d voxels to pack.\n", voxelNum);
        return;
    }
    voxelIdx minIdx, maxIdx;
    minIdx = maxIdx = voxelIdxList[0];
    for (uint32_t i = 0; i < voxelNum; i++) {
        for (uint32_t j = 0; j < 3; j++) {
            if(minIdx.list[j] > voxelIdxList[i].list[j]) {
                minIdx.list[j] = voxelIdxList[i].list[j];
            }
            if(maxIdx.list[j] < voxelIdxList[i].list[j]) {
                maxIdx.list[j] = voxelIdxList[i].list[j];
            }
        }
    }

    for (uint32_t i = 0; i < 3; i++) {
        voxelIdx diff = maxIdx - minIdx;
        if (diff.list[i] >= (1 << geohashLen[i])) {
            printf("Map shape error! On dim %d, min: %d, max: %d\n", i, minIdx.list[i], maxIdx.list[i]);
            return;
        }
    }

    map<voxelIdx, uint32_t>::iterator mapIter;
    uint32_t mapLb[3];
    uint32_t mapUb[3];
    int32_t mapLbIdx[3];
    float firstCenter[3];
    for (uint32_t i = 0; i < 3; i++) {
        mapLbIdx[i] = minIdx.list[i] - ((1 << geohashLen[i]) - (maxIdx.list[i] - minIdx.list[i] + 1)) / 2;
    }
    printf("Min idx: (%d, %d, %d)\n", minIdx.elem.x, minIdx.elem.y, minIdx.elem.z);
    printf("Max idx: (%d, %d, %d)\n", maxIdx.elem.x, maxIdx.elem.y, maxIdx.elem.z);
    printf("MapLb idx: (%d, %d, %d)\n", mapLbIdx[0], mapLbIdx[1], mapLbIdx[2]);

    memset(metadata + 2, 0xFF, METADATA_SIZE * sizeof(uint64_t));
    for (uint32_t i = 0; i < voxelNum; i++) {
        mapIter = mapInfo.idx.find(voxelIdxList[i]);
        if(mapIter == mapInfo.idx.end()) {
            printf("Not found!\n");
            continue;
        }
        uint32_t idx = mapIter->second;
        voxelIdx mapIdx;
        for (uint32_t d = 0; d < 3; d++) {
            mapIdx.list[d] = int32_t(voxelIdxList[i].list[d]) - mapLbIdx[d];
        }
        uint32_t metadataAddr = mapIdx.concat();
        uint32_t metadataContent = (mapInfo.dynShift[idx] << METADATA_ADDR_WIDTH) | i;
        metadata[metadataAddr + 2] = metadataContent;
        for (uint32_t j = 0; j < 3; j++) {
            *voxelMem++ = mapInfo.voxelInfo[idx * 3 + j];
        }
        if (i == 0) {
            for (uint32_t j = 0; j < 3; j++) {
                float lb = mapInfo.center[idx * 3 + j] - cellSize[j] / 2 - (int32_t(voxelIdxList[0].list[j]) - mapLbIdx[j]) * cellSize[j];
                float ub = lb + cellSize[j] * (1 << geohashLen[j]);
                printf("LB %d=%f UB=%f\n", j, lb, ub);
                mapCenter[j] = (ub + lb) / 2;
                lb = -cellSize[j] * (1 << (geohashLen[j] - 1));
                ub = cellSize[j] * (1 << (geohashLen[j] - 1));
                mapLb[j] = int32_t(lb * (1 << BOUNDARY_SHIFT)) & ((1 << BOUNDARY_WIDTH) - 1);
                mapUb[j] = int32_t(ub * (1 << BOUNDARY_SHIFT)) & ((1 << BOUNDARY_WIDTH) - 1);
            }
        }
    }
    // LB - metadata[0] | UB - metadata[1]
    metadata[0] = metadata[1] = 0;
    for (uint32_t i = 0; i < 3; i++) {
        metadata[0] = (metadata[0] << BOUNDARY_WIDTH) | mapLb[i];
        metadata[1] = (metadata[1] << BOUNDARY_WIDTH) | mapUb[i];
    }
    *metadataLen = METADATA_SIZE;
    *metadataByte = METADATA_SIZE * 8 + 16;
    *voxelLen = voxelNum;
    *voxelByte = voxelNum * 3 * 8;
}

// Debug Function
// Given a list of voxelidx, dumpPCD locates these voxels in mapInfo and save the voxel avg,
// and save them into "${filenamePrefix}Avg.pcd" and "${filenamePrefix}Center.pcd" perspectively.
void dumpPCD(const char* filenamePrefix, vector<voxelIdx>& dumpIdx, mapInfoType& mapInfo) {
    PointCloud<PointXYZ>::Ptr centerCloud(new PointCloud<PointXYZ>);
    PointCloud<PointXYZ>::Ptr avgCloud(new PointCloud<PointXYZ>);
    vector<voxelIdx>::iterator iter;
    map<voxelIdx, uint32_t>::iterator mapIter;
    for (iter = dumpIdx.begin(); iter != dumpIdx.end(); iter++) {
        mapIter = mapInfo.idx.find(*iter);
        if (mapIter == mapInfo.idx.end()) {
            printf("Voxel not found\n");
            continue;
        }
        uint32_t idx = mapIter->second;
        float* center = mapInfo.center + idx * 3;
        float avg[3];
        for (uint32_t i = 0; i < 3; i++) {
            float relAvg;
            relAvg = float(mapInfo.voxelInfo[idx * 3 + i] >> (18 * 3)) / (1 << REL_AVG_SHIFT);
            avg[i] = relAvg + mapInfo.mapLb[i] + cellSize[i] * iter->list[i];
        }
        centerCloud->push_back(PointXYZ(center[0], center[1], center[2]));
        avgCloud->push_back(PointXYZ(avg[0], avg[1], avg[2]));
    }
    io::savePCDFile(string(filenamePrefix) + string("Center.pcd"), *centerCloud);
    io::savePCDFile(string(filenamePrefix) + string("Avg.pcd"), *avgCloud);
}

// dumpPCDFromBinary scanns the metadata, locates the voxelInfo in the voxelMem and saves the avg to PCD.
void dumpPCDFromBinary(const char* filename, uint32_t voxelNum, uint64_t* metadata, uint64_t* voxelMem) {
    PointCloud<PointXYZ>::Ptr cloud(new PointCloud<PointXYZ>);
    uint32_t addrMask = (1 << METADATA_ADDR_WIDTH) - 1;
    uint32_t shiftMask = (1 << 2) - 1;
    float lb[3], ub[3];
    uint64_t metadataLb = metadata[0];
    uint64_t metadataUb = metadata[1];
    for (uint32_t i = 0; i < 3; i++) {
        lb[2 - i] = float(int16_t(metadataLb & ((1 << BOUNDARY_WIDTH) - 1))) / (1 << BOUNDARY_SHIFT);
        ub[2 - i] = float(int16_t(metadataUb & ((1 << BOUNDARY_WIDTH) - 1))) / (1 << BOUNDARY_SHIFT);
        metadataLb >>= BOUNDARY_WIDTH;
        metadataUb >>= BOUNDARY_WIDTH;
    }
    printf("Unpack LB: (%f, %f, %f)\n", lb[0], lb[1], lb[2]);
    uint32_t voxelCnt = 0;
    float max[3], min[3];
    for (uint32_t i = 2; i < METADATA_SIZE + 2; i++) {
        uint32_t addr = metadata[i] & addrMask;
        uint32_t shift = (metadata[i] >> METADATA_ADDR_WIDTH) & shiftMask;
        if(addr == addrMask) {
            continue;
        }
        voxelCnt++;
        uint64_t* voxelInfo = voxelMem + addr * 3;
        voxelIdx idx;
        idx.unpack(i - 2);
        float avg[3];
        for (uint32_t j = 0; j < 3; j++) {
            float relAvg = float(voxelInfo[j] >> (18 * 3)) / (1 << REL_AVG_SHIFT);
            avg[j] = relAvg + lb[j] + idx.list[j] * cellSize[j];
            if(i == 2) {
                max[j] = min[j] = avg[j];
            } else {
                if(avg[j] > max[j]) max[j] = avg[j];
                if(avg[j] < min[j]) min[j] = avg[j];
            }
        }
        cloud->push_back(PointXYZ(avg[0], avg[1], avg[2]));
    }
    printf("Max: (%f, %f, %f) Min: (%f, %f, %f)\n", max[0], max[1], max[2], min[0], min[1], min[2]);
    if (voxelCnt != voxelNum) {
        printf("Metadata does not match the voxelNum! VoxelNum: %d, metadata content: %d\n", voxelNum, voxelCnt);
    }
    io::savePCDFile(filename, *cloud);
}

void dumpLocalMap(uint64_t* metadata, uint64_t* voxelMem, localMapType& localMap, float* lb, float* ub) {
    PointCloud<PointXYZ>::Ptr cloud(new PointCloud<PointXYZ>);
    uint32_t addrMask = (1 << METADATA_ADDR_WIDTH) - 1;
    uint32_t shiftMask = (1 << 2) - 1;
    uint64_t metadataLb = metadata[0];
    uint64_t metadataUb = metadata[1];
    for (uint32_t i = 0; i < 3; i++) {
        lb[2 - i] = float(int16_t(metadataLb & ((1 << BOUNDARY_WIDTH) - 1))) / (1 << BOUNDARY_SHIFT);
        ub[2 - i] = float(int16_t(metadataUb & ((1 << BOUNDARY_WIDTH) - 1))) / (1 << BOUNDARY_SHIFT);
        metadataLb >>= BOUNDARY_WIDTH;
        metadataUb >>= BOUNDARY_WIDTH;
    }
    printf("Unpack LB: (%f, %f, %f)\n", lb[0], lb[1], lb[2]);
    uint32_t voxelCnt = 0;
    float max[3] = {0, 0, 0}, min[3] = {0, 0, 0};
    fstream test("localMapTest.csv", ios_base::out);
    for (uint32_t i = 2; i < METADATA_SIZE + 2; i++) {
        uint32_t addr = metadata[i] & addrMask;
        uint32_t shift = (metadata[i] >> METADATA_ADDR_WIDTH) & shiftMask;
        if(addr == addrMask) {
            continue;
        }
        voxelCnt++;
        voxelIdx idx;
        idx.unpack(i - 2);
        voxelInfoType voxelInfo;
        for (uint32_t j = 0; j < 3; j++) {
            uint64_t temp = *(voxelMem + addr * 3 + j);
            for (uint32_t col = 0; col < 3; col++) {
                int32_t icovRead = temp & ((1 << 18) - 1);
                if (icovRead >> 17) {
                    // Is negative
                    icovRead = (((~icovRead) + 1) & ((1 << 18) - 1)) * -1;
                }
                voxelInfo.icov[j][col] = float(icovRead << shift) / (1 << ICOV_SHIFT);
                temp >>= ICOV_WIDTH;
            }
            float relAvg = float(temp) / (1 << REL_AVG_SHIFT);
            voxelInfo.avg[j] = relAvg + lb[j] + idx.list[j] * cellSize[j];
            if(i == 2) {
                max[j] = min[j] = voxelInfo.avg[j];
            } else {
                if(voxelInfo.avg[j] > max[j]) max[j] = voxelInfo.avg[j];
                if(voxelInfo.avg[j] < min[j]) min[j] = voxelInfo.avg[j];
            }
        }
        test << "0 0 0 ";
        for (uint32_t j = 0; j < 3; j++) {
            test << voxelInfo.avg[j] << " ";
        }
        for (uint32_t j = 0; j < 9; j++) {
            test << voxelInfo.icov[j / 3][j % 3];
            if (j != 8) test << " ";
        }
        test << endl;
        localMap.insert(make_pair(i - 2, voxelInfo));
    }
    test.close();
    printf("Max: (%f, %f, %f) Min: (%f, %f, %f)\n", max[0], max[1], max[2], min[0], min[1], min[2]);
}

template <int GEOHASH_WIDTH>
uint32_t coordinate2Geohash(float x, float lb, float ub) {
	uint32_t tempGeohash = 0;
    float sepPoint;
    bool ge;
    for(uint8_t i = 0; i < GEOHASH_WIDTH; i++) {
        sepPoint = (lb + ub) / 2;
        tempGeohash |= x >= sepPoint ? 1 : 0;
        if (x >= sepPoint) {
            tempGeohash |= 1;
            lb = sepPoint;
        } else {
            ub = sepPoint;
        }
        tempGeohash <<= 1;
    }
    return reverseBits(tempGeohash, GEOHASH_WIDTH);
}

bool readMap(float x, float y, float z, float* lb, float* ub, localMapType& localMap, voxelInfoType& voxelInfo) {
    voxelIdx idx;
    idx.elem.x = coordinate2Geohash<GEOHASH_X_WIDTH>(x, lb[0], ub[0]);
    idx.elem.y = coordinate2Geohash<GEOHASH_Y_WIDTH>(y, lb[1], ub[1]);
    idx.elem.z = coordinate2Geohash<GEOHASH_Z_WIDTH>(z, lb[2], ub[2]);
    localMapType::iterator iter = localMap.find(idx.concat());
    if(iter != localMap.end()) {
        voxelInfo = iter->second;
        return true;
    } else {
        return false;
    }
}

int main(int32_t argc, char* argv[]) {
    if(argc != 2) {
        printf("Usage: mapLoad binaryFilename");
        return -1;
    }
    printf("Test mapLoad\n");
    mapInfoType mapInfo;
    // Load the map from the binary map pack.
    loadMap(argv[1], mapInfo);

    // Dump the full map
    vector<voxelIdx> dumpVoxel;
    dumpVoxel.resize(mapInfo.idx.size());
    map<voxelIdx, uint32_t>::iterator mapIter;
    vector<voxelIdx>::iterator dumpIter = dumpVoxel.begin();
    for (mapIter = mapInfo.idx.begin(); mapIter != mapInfo.idx.end(); mapIter++) {
        *dumpIter++ = mapIter->first;
    }
    dumpPCD("fullmap", dumpVoxel, mapInfo);

    // Search some voxel blocks
    voxelIdx mapIdxBuffer[METADATA_SIZE];
    uint32_t voxelNum = 0;
    voxelIdx startPoint;
    startPoint.elem.x = 240;
    startPoint.elem.y = 650;
    startPoint.elem.z = 5;
    for (uint32_t x = 0; x < (1 << GEOHASH_X_WIDTH) && voxelNum < VOXEL_NUM_LIMIT; x++) {
        for (uint32_t y = 0; y < (1 << GEOHASH_Y_WIDTH) && voxelNum < VOXEL_NUM_LIMIT; y++) {
            for (uint32_t z = 0; z < (1 << GEOHASH_Z_WIDTH) && voxelNum < VOXEL_NUM_LIMIT; z++) {
                voxelIdx temp;
                temp.elem.x = startPoint.elem.x + x;
                temp.elem.y = startPoint.elem.y + y;
                temp.elem.z = startPoint.elem.z + z;
                if (mapInfo.idx.find(temp) != mapInfo.idx.end()) {
                    mapIdxBuffer[voxelNum] = temp;
                    voxelNum++;
                }
            }
        }
    }
    printf("Test Map 1: Found %d voxels\n", voxelNum);
    // Pack the voxel blocks
    uint64_t metadata[METADATA_SIZE + 2];
    uint64_t voxelMem[VOXEL_NUM_LIMIT * 3];
    uint32_t metadataLen, metadataByte;
    uint32_t voxelMemLen, voxelMemByte;
    float mapCenter[3];
    if (voxelNum > 0) {
        packMap(
            mapIdxBuffer, voxelNum, 
            mapInfo, 
            metadata, &metadataLen, &metadataByte, 
            voxelMem, &voxelMemLen, &voxelMemByte, 
            mapCenter
        );
        printf("Map Center (%f, %f, %f)\n", mapCenter[0], mapCenter[1], mapCenter[2]);
        printf("Metadata Len: %d, byte: %d\n", metadataLen, metadataByte);
        printf("VoxelMem Len: %d, byte: %d\n", voxelMemLen, voxelMemByte);
        // Dump the concated map
        dumpPCDFromBinary("selectedMap1.pcd", voxelNum, metadata, voxelMem);
    }

    voxelNum = 0;
    startPoint.elem.x = 515;
    startPoint.elem.y = 355;
    startPoint.elem.z = 2;
    for (uint32_t x = 0; x < (1 << GEOHASH_X_WIDTH) && voxelNum < VOXEL_NUM_LIMIT; x++) {
        for (uint32_t y = 0; y < (1 << GEOHASH_Y_WIDTH) && voxelNum < VOXEL_NUM_LIMIT; y++) {
            for (uint32_t z = 0; z < (1 << GEOHASH_Z_WIDTH) && voxelNum < VOXEL_NUM_LIMIT; z++) {
                voxelIdx temp;
                temp.elem.x = startPoint.elem.x + x;
                temp.elem.y = startPoint.elem.y + y;
                temp.elem.z = startPoint.elem.z + z;
                if (mapInfo.idx.find(temp) != mapInfo.idx.end()) {
                    mapIdxBuffer[voxelNum] = temp;
                    voxelNum++;
                }
            }
        }
    }
    printf("Test Map 2: Found %d voxels\n", voxelNum);
    if(voxelNum > 0) {
        // Pack the voxel blocks
        packMap(
            mapIdxBuffer, voxelNum, 
            mapInfo, 
            metadata, &metadataLen, &metadataByte, 
            voxelMem, &voxelMemLen, &voxelMemByte, 
            mapCenter
        );
        printf("Map Center (%f, %f, %f)\n", mapCenter[0], mapCenter[1], mapCenter[2]);
        printf("Metadata Len: %d, byte: %d\n", metadataLen, metadataByte);
        printf("VoxelMem Len: %d, byte: %d\n", voxelMemLen, voxelMemByte);
        // Dump the concated map
        dumpPCDFromBinary("selectedMap2.pcd", voxelNum, metadata, voxelMem);
    }

    printf("***********************************************************************\n");
    fstream listFile("voxelIdxListFile.csv", ios_base::in);
    voxelIdx vList[2291];
    char x;
    for (uint32_t i = 0; i < 1397; i++) {
        listFile >> vList[i].elem.x >> x >> vList[i].elem.y >> x >> vList[i].elem.z;
    }
        packMap(
            vList, 1397, 
            mapInfo, 
            metadata, &metadataLen, &metadataByte, 
            voxelMem, &voxelMemLen, &voxelMemByte, 
            mapCenter
        );
        fstream newMetadata("newMetadata.bin", ios_base::out | ios_base::binary);
        newMetadata.write((char*)metadata, metadataByte);
        newMetadata.close();
        fstream newVoxelMem("newVoxelMem.bin", ios_base::out | ios_base::binary);
        newVoxelMem.write((char*)voxelMem, voxelMemByte);
        newVoxelMem.close();
        printf("Map Center (%f, %f, %f)\n", mapCenter[0], mapCenter[1], mapCenter[2]);
        printf("Metadata Len: %d, byte: %d\n", metadataLen, metadataByte);
        printf("VoxelMem Len: %d, byte: %d\n", voxelMemLen, voxelMemByte);
        // Dump the concated map
        dumpPCDFromBinary("selectedMap3.pcd", 1397, metadata, voxelMem);
        // Dump the local map
        localMapType localMap;
        float lb[3], ub[3];
        dumpLocalMap(metadata, voxelMem, localMap, lb, ub);
    return 0;
}
