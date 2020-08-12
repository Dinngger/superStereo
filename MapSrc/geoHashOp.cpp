#include "geoHashOp.h"
#include <cmath>
float quantize(float x, uint32_t fpWidth, uint32_t fpShift) {
    double ret = x;
    ret *= 1 << fpShift;
    ret = round(ret);
    ret /= 1 << fpShift;
    return float(ret);
}

uint32_t genBinaryGeoHash(float point[3], 
    float lowerBound[3], float upperBound[3], 
    uint32_t hashLen, uint32_t actHashDim, 
    uint32_t fpWidth, uint32_t fpShift
    ) {
    uint32_t code = 0;
    float tempLb[3], tempUb[3];
    for(uint32_t i = 0; i < 3; i++) {
    	tempLb[i] = lowerBound[i];
    	tempUb[i] = upperBound[i];
    }
    for(uint32_t i = 0; i < hashLen; i++) {
        uint32_t dim = i % actHashDim;
        float mid = quantize((tempUb[dim] + tempLb[dim]) / 2, fpWidth, fpShift);
        code <<= 1;
        if(point[dim] >= mid) {
            code |= 1;
            tempLb[dim] = mid;
        } else {
        	tempUb[dim] = mid;
        }
    }
    return code;
}

uint32_t reverseBits(uint32_t bits, uint32_t len) {
    uint32_t ret = 0;
    for(uint32_t i = 0; i < len; i++) {
        ret <<= 1;
        ret |= bits & 1;
        bits >>= 1;
    }
    return ret;
}

void getSepCode(uint32_t code, uint32_t codeLen, uint32_t actHashDim, uint32_t* sepCode) {
    for(uint32_t i = 0; i < actHashDim; i++) {
        sepCode[i] = 0;
    }
    uint32_t reverseCode = reverseBits(code, codeLen);
    for(uint32_t i = 0; i < codeLen; i++) {
        sepCode[i % actHashDim] <<= 1;
        sepCode[i % actHashDim] |= reverseCode & 1;
        reverseCode >>= 1;
    }
}

uint32_t mergeSepCode(uint32_t* sepCode, uint32_t codeLen, uint32_t actHashDim) {
    uint32_t reverseSepCode[actHashDim];
    for(uint32_t i = 0; i < actHashDim; i++) {
        reverseSepCode[i] = reverseBits(sepCode[i], codeLen / actHashDim);
    }
    uint32_t ret = 0;
    for(uint32_t i = 0; i < codeLen; i++) {
        ret <<= 1;
        ret |= reverseSepCode[i % actHashDim] & 1;
        reverseSepCode[i % actHashDim] >>= 1;
    }
    return ret;
}

uint32_t getSortCode(uint32_t code, uint32_t codeLen, uint32_t actHashDim) {
    uint32_t sepCode[actHashDim];
    uint32_t mergedCode = 0;
    getSepCode(code, codeLen, actHashDim, sepCode);
    for(uint32_t d = 0; d < actHashDim; d++) {
        mergedCode <<= (codeLen / actHashDim);
        mergedCode |= sepCode[actHashDim - d - 1];
    }
    return mergedCode;
}
