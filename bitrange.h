#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdlib>
class BitRange {
  public:
    BitRange(size_t size) : size(size) {
        nmemb = size / 64 + ((size % 64) != 0);
        buf = (uint64_t *)calloc(nmemb, sizeof(uint64_t));
    }

    virtual ~BitRange() { free(buf); }

    void set(size_t bitNum) { buf[bitNum / 64] |= (1 << (bitNum % 64)); }
    void unset(size_t bitNum) { buf[bitNum / 64] &= ~(1 << (bitNum % 64)); }
    bool get(size_t bitNum) { return buf[bitNum / 64] & (1 << (bitNum % 64)); }

    size_t size;
    size_t nmemb;
    uint64_t *buf;
};
