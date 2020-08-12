#ifndef __MAP_LOAD_H
#define __MAP_LOAD_H

#include <cstdint>
#include <map>

#define REL_AVG_WIDTH 10
#define REL_AVG_SHIFT 8
#define REL_AVG_MASK ((1 << REL_AVG_WIDTH) - 1)
#define ICOV_WIDTH 18
#define ICOV_SHIFT 7
#define ICOV_MASK ((1 << ICOV_WIDTH) - 1)
#define ICOV_DYNAMIC_SHIFT 2
#define CELL_X_SIZE 1.5
#define CELL_Y_SIZE 1.5
#define CELL_Z_SIZE 1.5
#define GEOHASH_X_WIDTH 7
#define GEOHASH_Y_WIDTH 7
#define GEOHASH_Z_WIDTH 4
#define GEOHASH_X_MASK ((1 << GEOHASH_X_WIDTH) - 1)
#define GEOHASH_Y_MASK ((1 << GEOHASH_Y_WIDTH) - 1)
#define GEOHASH_Z_MASK ((1 << GEOHASH_Z_WIDTH) - 1)
#define METADATA_SIZE (1 << (GEOHASH_X_WIDTH + GEOHASH_Y_WIDTH + GEOHASH_Z_WIDTH))
#define METADATA_ADDR_WIDTH 14
#define BOUNDARY_WIDTH 16
#define BOUNDARY_SHIFT 8
#define VOXEL_NUM_LIMIT 10000
const float cellSize[3] = {CELL_X_SIZE, CELL_Y_SIZE, CELL_Z_SIZE};
const uint8_t geohashLen[3] = {GEOHASH_X_WIDTH, GEOHASH_Y_WIDTH, GEOHASH_Z_WIDTH};

union voxelIdx {
    struct {
        uint32_t x;
        uint32_t y;
        uint32_t z;
    } elem;
    uint32_t list[3];
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
};
bool operator < (const voxelIdx& lhs, const voxelIdx& rhs) {
    return (lhs.elem.x < rhs.elem.x) || (lhs.elem.x == rhs.elem.x && lhs.elem.y < rhs.elem.y) || (lhs.elem.x == rhs.elem.x && lhs.elem.y == rhs.elem.y && lhs.elem.z < rhs.elem.z);
}

struct mapInfoType {
    std::map<voxelIdx, uint32_t> idx;
    float mapLb[3];
    float* center;
    uint8_t* dynShift;
    uint64_t* voxelInfo;
};

struct voxelInfoType {
    float avg[3];
    float icov[3][3];
};
typedef std::map<uint32_t, voxelInfoType> localMapType;

// Load the binary map package and save them into mapInfo
void loadMap(const char* filename, mapInfoType& mapInfo);
// Pack the map indicated by the voxelIdxList
// metadata and voxelMem are 64-bit aligned
// fill the "dmaTransferLen" Reg with metadataLen and voxelLen
// metadataByte and voxelByte are bytes to be transfered with dma
void packMap(
    voxelIdx* voxelIdxList, uint32_t voxelNum, 
    mapInfoType& mapInfo, 
    uint64_t* metadata, uint32_t* metadataLen, uint32_t* metadataByte,
    uint64_t* voxelMem, uint32_t* voxelLen, uint32_t* voxelByte,
    float* mapCenter
);


void dumpLocalMap(uint64_t* metadata, uint64_t* voxelMem, localMapType& localMap, float* lb, float* ub);
bool readMap(float x, float y, float z, float* lb, float* ub, localMapType& localMap, voxelInfoType& voxelInfo);

#endif