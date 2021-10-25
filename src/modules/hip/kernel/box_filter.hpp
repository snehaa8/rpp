#include <hip/hip_runtime.h>
#include "hip/rpp_hip_common.hpp"

__device__ void box_filter_3x3_row_hip_compute(uchar *srcPtr, d_float8 *dst_f8)
{
    uint src;
    float src_f;
    uint *src_uchar4 = (uint *)srcPtr;
    src = src_uchar4[0];
    src_f = rpp_hip_unpack0(src);
    dst_f8->x.x = fmaf(src_f, 0.1111111f, dst_f8->x.x);
    src_f = rpp_hip_unpack1(src);
    dst_f8->x.x = fmaf(src_f, 0.1111111f, dst_f8->x.x);
    dst_f8->x.y = fmaf(src_f, 0.1111111f, dst_f8->x.y);
    src_f = rpp_hip_unpack2(src);
    dst_f8->x.x = fmaf(src_f, 0.1111111f, dst_f8->x.x);
    dst_f8->x.y = fmaf(src_f, 0.1111111f, dst_f8->x.y);
    dst_f8->x.z = fmaf(src_f, 0.1111111f, dst_f8->x.z);
    src_f = rpp_hip_unpack3(src);
    dst_f8->x.y = fmaf(src_f, 0.1111111f, dst_f8->x.y);
    dst_f8->x.z = fmaf(src_f, 0.1111111f, dst_f8->x.z);
    dst_f8->x.w = fmaf(src_f, 0.1111111f, dst_f8->x.w);
    src = src_uchar4[1];
    src_f = rpp_hip_unpack0(src);
    dst_f8->x.z = fmaf(src_f, 0.1111111f, dst_f8->x.z);
    dst_f8->x.w = fmaf(src_f, 0.1111111f, dst_f8->x.w);
    dst_f8->y.x = fmaf(src_f, 0.1111111f, dst_f8->y.x);
    src_f = rpp_hip_unpack1(src);
    dst_f8->x.w = fmaf(src_f, 0.1111111f, dst_f8->x.w);
    dst_f8->y.x = fmaf(src_f, 0.1111111f, dst_f8->y.x);
    dst_f8->y.y = fmaf(src_f, 0.1111111f, dst_f8->y.y);
    src_f = rpp_hip_unpack2(src);
    dst_f8->y.x = fmaf(src_f, 0.1111111f, dst_f8->y.x);
    dst_f8->y.y = fmaf(src_f, 0.1111111f, dst_f8->y.y);
    dst_f8->y.z = fmaf(src_f, 0.1111111f, dst_f8->y.z);
    src_f = rpp_hip_unpack3(src);
    dst_f8->y.y = fmaf(src_f, 0.1111111f, dst_f8->y.y);
    dst_f8->y.z = fmaf(src_f, 0.1111111f, dst_f8->y.z);
    dst_f8->y.w = fmaf(src_f, 0.1111111f, dst_f8->y.w);
    src = src_uchar4[2];
    src_f = rpp_hip_unpack0(src);
    dst_f8->y.z = fmaf(src_f, 0.1111111f, dst_f8->y.z);
    dst_f8->y.w = fmaf(src_f, 0.1111111f, dst_f8->y.w);
    src_f = rpp_hip_unpack1(src);
    dst_f8->y.w = fmaf(src_f, 0.1111111f, dst_f8->y.w);
}

__device__ void box_filter_5x5_row_hip_compute(uchar *srcPtr, d_float8 *dst_f8)
{
    uint src;
    float src_f;
    uint *src_uchar4 = (uint *)srcPtr;
    src = src_uchar4[0];
    src_f = rpp_hip_unpack0(src);
    dst_f8->x.x = fmaf(src_f, 0.04f, dst_f8->x.x);
    src_f = rpp_hip_unpack1(src);
    dst_f8->x.x = fmaf(src_f, 0.04f, dst_f8->x.x);
    dst_f8->x.y = fmaf(src_f, 0.04f, dst_f8->x.y);
    src_f = rpp_hip_unpack2(src);
    dst_f8->x.x = fmaf(src_f, 0.04f, dst_f8->x.x);
    dst_f8->x.y = fmaf(src_f, 0.04f, dst_f8->x.y);
    dst_f8->x.z = fmaf(src_f, 0.04f, dst_f8->x.z);
    src_f = rpp_hip_unpack3(src);
    dst_f8->x.x = fmaf(src_f, 0.04f, dst_f8->x.x);
    dst_f8->x.y = fmaf(src_f, 0.04f, dst_f8->x.y);
    dst_f8->x.z = fmaf(src_f, 0.04f, dst_f8->x.z);
    dst_f8->x.w = fmaf(src_f, 0.04f, dst_f8->x.w);
    src = src_uchar4[1];
    src_f = rpp_hip_unpack0(src);
    dst_f8->x.x = fmaf(src_f, 0.04f, dst_f8->x.x);
    dst_f8->x.y = fmaf(src_f, 0.04f, dst_f8->x.y);
    dst_f8->x.z = fmaf(src_f, 0.04f, dst_f8->x.z);
    dst_f8->x.w = fmaf(src_f, 0.04f, dst_f8->x.w);
    dst_f8->y.x = fmaf(src_f, 0.04f, dst_f8->y.x);
    src_f = rpp_hip_unpack1(src);
    dst_f8->x.y = fmaf(src_f, 0.04f, dst_f8->x.y);
    dst_f8->x.z = fmaf(src_f, 0.04f, dst_f8->x.z);
    dst_f8->x.w = fmaf(src_f, 0.04f, dst_f8->x.w);
    dst_f8->y.x = fmaf(src_f, 0.04f, dst_f8->y.x);
    dst_f8->y.y = fmaf(src_f, 0.04f, dst_f8->y.y);
    src_f = rpp_hip_unpack2(src);
    dst_f8->x.z = fmaf(src_f, 0.04f, dst_f8->x.z);
    dst_f8->x.w = fmaf(src_f, 0.04f, dst_f8->x.w);
    dst_f8->y.x = fmaf(src_f, 0.04f, dst_f8->y.x);
    dst_f8->y.y = fmaf(src_f, 0.04f, dst_f8->y.y);
    dst_f8->y.z = fmaf(src_f, 0.04f, dst_f8->y.z);
    src_f = rpp_hip_unpack3(src);
    dst_f8->x.w = fmaf(src_f, 0.04f, dst_f8->x.w);
    dst_f8->y.x = fmaf(src_f, 0.04f, dst_f8->y.x);
    dst_f8->y.y = fmaf(src_f, 0.04f, dst_f8->y.y);
    dst_f8->y.z = fmaf(src_f, 0.04f, dst_f8->y.z);
    dst_f8->y.w = fmaf(src_f, 0.04f, dst_f8->y.w);
    src = src_uchar4[2];
    src_f = rpp_hip_unpack0(src);
    dst_f8->y.x = fmaf(src_f, 0.04f, dst_f8->y.x);
    dst_f8->y.y = fmaf(src_f, 0.04f, dst_f8->y.y);
    dst_f8->y.z = fmaf(src_f, 0.04f, dst_f8->y.z);
    dst_f8->y.w = fmaf(src_f, 0.04f, dst_f8->y.w);
    src_f = rpp_hip_unpack1(src);
    dst_f8->y.y = fmaf(src_f, 0.04f, dst_f8->y.y);
    dst_f8->y.z = fmaf(src_f, 0.04f, dst_f8->y.z);
    dst_f8->y.w = fmaf(src_f, 0.04f, dst_f8->y.w);
    src_f = rpp_hip_unpack2(src);
    dst_f8->y.z = fmaf(src_f, 0.04f, dst_f8->y.z);
    dst_f8->y.w = fmaf(src_f, 0.04f, dst_f8->y.w);
    src_f = rpp_hip_unpack3(src);
    dst_f8->y.w = fmaf(src_f, 0.04f, dst_f8->y.w);
}

__device__ void box_filter_7x7_row_hip_compute(uchar *srcPtr, d_float8 *dst_f8)
{
    uint src;
    float src_f;
    uint *src_uchar4 = (uint *)srcPtr;
    src = src_uchar4[0];
    src_f = rpp_hip_unpack0(src);
    dst_f8->x.x = fmaf(src_f, 0.02040816f, dst_f8->x.x);
    src_f = rpp_hip_unpack1(src);
    dst_f8->x.x = fmaf(src_f, 0.02040816f, dst_f8->x.x);
    dst_f8->x.y = fmaf(src_f, 0.02040816f, dst_f8->x.y);
    src_f = rpp_hip_unpack2(src);
    dst_f8->x.x = fmaf(src_f, 0.02040816f, dst_f8->x.x);
    dst_f8->x.y = fmaf(src_f, 0.02040816f, dst_f8->x.y);
    dst_f8->x.z = fmaf(src_f, 0.02040816f, dst_f8->x.z);
    src_f = rpp_hip_unpack3(src);
    dst_f8->x.x = fmaf(src_f, 0.02040816f, dst_f8->x.x);
    dst_f8->x.y = fmaf(src_f, 0.02040816f, dst_f8->x.y);
    dst_f8->x.z = fmaf(src_f, 0.02040816f, dst_f8->x.z);
    dst_f8->x.w = fmaf(src_f, 0.02040816f, dst_f8->x.w);
    src = src_uchar4[1];
    src_f = rpp_hip_unpack0(src);
    dst_f8->x.x = fmaf(src_f, 0.02040816f, dst_f8->x.x);
    dst_f8->x.y = fmaf(src_f, 0.02040816f, dst_f8->x.y);
    dst_f8->x.z = fmaf(src_f, 0.02040816f, dst_f8->x.z);
    dst_f8->x.w = fmaf(src_f, 0.02040816f, dst_f8->x.w);
    dst_f8->y.x = fmaf(src_f, 0.02040816f, dst_f8->y.x);
    src_f = rpp_hip_unpack1(src);
    dst_f8->x.x = fmaf(src_f, 0.02040816f, dst_f8->x.x);
    dst_f8->x.y = fmaf(src_f, 0.02040816f, dst_f8->x.y);
    dst_f8->x.z = fmaf(src_f, 0.02040816f, dst_f8->x.z);
    dst_f8->x.w = fmaf(src_f, 0.02040816f, dst_f8->x.w);
    dst_f8->y.x = fmaf(src_f, 0.02040816f, dst_f8->y.x);
    dst_f8->y.y = fmaf(src_f, 0.02040816f, dst_f8->y.y);
    src_f = rpp_hip_unpack2(src);
    dst_f8->x.x = fmaf(src_f, 0.02040816f, dst_f8->x.x);
    dst_f8->x.y = fmaf(src_f, 0.02040816f, dst_f8->x.y);
    dst_f8->x.z = fmaf(src_f, 0.02040816f, dst_f8->x.z);
    dst_f8->x.w = fmaf(src_f, 0.02040816f, dst_f8->x.w);
    dst_f8->y.x = fmaf(src_f, 0.02040816f, dst_f8->y.x);
    dst_f8->y.y = fmaf(src_f, 0.02040816f, dst_f8->y.y);
    dst_f8->y.z = fmaf(src_f, 0.02040816f, dst_f8->y.z);
    src_f = rpp_hip_unpack3(src);
    dst_f8->x.y = fmaf(src_f, 0.02040816f, dst_f8->x.y);
    dst_f8->x.z = fmaf(src_f, 0.02040816f, dst_f8->x.z);
    dst_f8->x.w = fmaf(src_f, 0.02040816f, dst_f8->x.w);
    dst_f8->y.x = fmaf(src_f, 0.02040816f, dst_f8->y.x);
    dst_f8->y.y = fmaf(src_f, 0.02040816f, dst_f8->y.y);
    dst_f8->y.z = fmaf(src_f, 0.02040816f, dst_f8->y.z);
    dst_f8->y.w = fmaf(src_f, 0.02040816f, dst_f8->y.w);
    src = src_uchar4[2];
    src_f = rpp_hip_unpack0(src);
    dst_f8->x.z = fmaf(src_f, 0.02040816f, dst_f8->x.z);
    dst_f8->x.w = fmaf(src_f, 0.02040816f, dst_f8->x.w);
    dst_f8->y.x = fmaf(src_f, 0.02040816f, dst_f8->y.x);
    dst_f8->y.y = fmaf(src_f, 0.02040816f, dst_f8->y.y);
    dst_f8->y.z = fmaf(src_f, 0.02040816f, dst_f8->y.z);
    dst_f8->y.w = fmaf(src_f, 0.02040816f, dst_f8->y.w);
    src_f = rpp_hip_unpack1(src);
    dst_f8->x.w = fmaf(src_f, 0.02040816f, dst_f8->x.w);
    dst_f8->y.x = fmaf(src_f, 0.02040816f, dst_f8->y.x);
    dst_f8->y.y = fmaf(src_f, 0.02040816f, dst_f8->y.y);
    dst_f8->y.z = fmaf(src_f, 0.02040816f, dst_f8->y.z);
    dst_f8->y.w = fmaf(src_f, 0.02040816f, dst_f8->y.w);
    src_f = rpp_hip_unpack2(src);
    dst_f8->y.x = fmaf(src_f, 0.02040816f, dst_f8->y.x);
    dst_f8->y.y = fmaf(src_f, 0.02040816f, dst_f8->y.y);
    dst_f8->y.z = fmaf(src_f, 0.02040816f, dst_f8->y.z);
    dst_f8->y.w = fmaf(src_f, 0.02040816f, dst_f8->y.w);
    src_f = rpp_hip_unpack3(src);
    dst_f8->y.y = fmaf(src_f, 0.02040816f, dst_f8->y.y);
    dst_f8->y.z = fmaf(src_f, 0.02040816f, dst_f8->y.z);
    dst_f8->y.w = fmaf(src_f, 0.02040816f, dst_f8->y.w);
    src = src_uchar4[3];
    src_f = rpp_hip_unpack0(src);
    dst_f8->y.z = fmaf(src_f, 0.02040816f, dst_f8->y.z);
    dst_f8->y.w = fmaf(src_f, 0.02040816f, dst_f8->y.w);
    src_f = rpp_hip_unpack1(src);
    dst_f8->y.w = fmaf(src_f, 0.02040816f, dst_f8->y.w);
}

__device__ void box_filter_9x9_row_hip_compute(uchar *srcPtr, d_float8 *dst_f8)
{
    uint src;
    float src_f;
    uint *src_uchar4 = (uint *)srcPtr;
    src = src_uchar4[0];
    src_f = rpp_hip_unpack0(src);
    dst_f8->x.x = fmaf(src_f, 0.01234568f, dst_f8->x.x);
    src_f = rpp_hip_unpack1(src);
    dst_f8->x.x = fmaf(src_f, 0.01234568f, dst_f8->x.x);
    dst_f8->x.y = fmaf(src_f, 0.01234568f, dst_f8->x.y);
    src_f = rpp_hip_unpack2(src);
    dst_f8->x.x = fmaf(src_f, 0.01234568f, dst_f8->x.x);
    dst_f8->x.y = fmaf(src_f, 0.01234568f, dst_f8->x.y);
    dst_f8->x.z = fmaf(src_f, 0.01234568f, dst_f8->x.z);
    src_f = rpp_hip_unpack3(src);
    dst_f8->x.x = fmaf(src_f, 0.01234568f, dst_f8->x.x);
    dst_f8->x.y = fmaf(src_f, 0.01234568f, dst_f8->x.y);
    dst_f8->x.z = fmaf(src_f, 0.01234568f, dst_f8->x.z);
    dst_f8->x.w = fmaf(src_f, 0.01234568f, dst_f8->x.w);
    src = src_uchar4[1];
    src_f = rpp_hip_unpack0(src);
    dst_f8->x.x = fmaf(src_f, 0.01234568f, dst_f8->x.x);
    dst_f8->x.y = fmaf(src_f, 0.01234568f, dst_f8->x.y);
    dst_f8->x.z = fmaf(src_f, 0.01234568f, dst_f8->x.z);
    dst_f8->x.w = fmaf(src_f, 0.01234568f, dst_f8->x.w);
    dst_f8->y.x = fmaf(src_f, 0.01234568f, dst_f8->y.x);
    src_f = rpp_hip_unpack1(src);
    dst_f8->x.x = fmaf(src_f, 0.01234568f, dst_f8->x.x);
    dst_f8->x.y = fmaf(src_f, 0.01234568f, dst_f8->x.y);
    dst_f8->x.z = fmaf(src_f, 0.01234568f, dst_f8->x.z);
    dst_f8->x.w = fmaf(src_f, 0.01234568f, dst_f8->x.w);
    dst_f8->y.x = fmaf(src_f, 0.01234568f, dst_f8->y.x);
    dst_f8->y.y = fmaf(src_f, 0.01234568f, dst_f8->y.y);
    src_f = rpp_hip_unpack2(src);
    dst_f8->x.x = fmaf(src_f, 0.01234568f, dst_f8->x.x);
    dst_f8->x.y = fmaf(src_f, 0.01234568f, dst_f8->x.y);
    dst_f8->x.z = fmaf(src_f, 0.01234568f, dst_f8->x.z);
    dst_f8->x.w = fmaf(src_f, 0.01234568f, dst_f8->x.w);
    dst_f8->y.x = fmaf(src_f, 0.01234568f, dst_f8->y.x);
    dst_f8->y.y = fmaf(src_f, 0.01234568f, dst_f8->y.y);
    dst_f8->y.z = fmaf(src_f, 0.01234568f, dst_f8->y.z);
    src_f = rpp_hip_unpack3(src);
    dst_f8->x.x = fmaf(src_f, 0.01234568f, dst_f8->x.x);
    dst_f8->x.y = fmaf(src_f, 0.01234568f, dst_f8->x.y);
    dst_f8->x.z = fmaf(src_f, 0.01234568f, dst_f8->x.z);
    dst_f8->x.w = fmaf(src_f, 0.01234568f, dst_f8->x.w);
    dst_f8->y.x = fmaf(src_f, 0.01234568f, dst_f8->y.x);
    dst_f8->y.y = fmaf(src_f, 0.01234568f, dst_f8->y.y);
    dst_f8->y.z = fmaf(src_f, 0.01234568f, dst_f8->y.z);
    dst_f8->y.w = fmaf(src_f, 0.01234568f, dst_f8->y.w);
    src = src_uchar4[2];
    src_f = rpp_hip_unpack0(src);
    dst_f8->x.x = fmaf(src_f, 0.01234568f, dst_f8->x.x);
    dst_f8->x.y = fmaf(src_f, 0.01234568f, dst_f8->x.y);
    dst_f8->x.z = fmaf(src_f, 0.01234568f, dst_f8->x.z);
    dst_f8->x.w = fmaf(src_f, 0.01234568f, dst_f8->x.w);
    dst_f8->y.x = fmaf(src_f, 0.01234568f, dst_f8->y.x);
    dst_f8->y.y = fmaf(src_f, 0.01234568f, dst_f8->y.y);
    dst_f8->y.z = fmaf(src_f, 0.01234568f, dst_f8->y.z);
    dst_f8->y.w = fmaf(src_f, 0.01234568f, dst_f8->y.w);
    src_f = rpp_hip_unpack1(src);
    dst_f8->x.y = fmaf(src_f, 0.01234568f, dst_f8->x.y);
    dst_f8->x.z = fmaf(src_f, 0.01234568f, dst_f8->x.z);
    dst_f8->x.w = fmaf(src_f, 0.01234568f, dst_f8->x.w);
    dst_f8->y.x = fmaf(src_f, 0.01234568f, dst_f8->y.x);
    dst_f8->y.y = fmaf(src_f, 0.01234568f, dst_f8->y.y);
    dst_f8->y.z = fmaf(src_f, 0.01234568f, dst_f8->y.z);
    dst_f8->y.w = fmaf(src_f, 0.01234568f, dst_f8->y.w);
    src_f = rpp_hip_unpack2(src);
    dst_f8->x.z = fmaf(src_f, 0.01234568f, dst_f8->x.z);
    dst_f8->x.w = fmaf(src_f, 0.01234568f, dst_f8->x.w);
    dst_f8->y.x = fmaf(src_f, 0.01234568f, dst_f8->y.x);
    dst_f8->y.y = fmaf(src_f, 0.01234568f, dst_f8->y.y);
    dst_f8->y.z = fmaf(src_f, 0.01234568f, dst_f8->y.z);
    dst_f8->y.w = fmaf(src_f, 0.01234568f, dst_f8->y.w);
    src_f = rpp_hip_unpack3(src);
    dst_f8->x.w = fmaf(src_f, 0.01234568f, dst_f8->x.w);
    dst_f8->y.x = fmaf(src_f, 0.01234568f, dst_f8->y.x);
    dst_f8->y.y = fmaf(src_f, 0.01234568f, dst_f8->y.y);
    dst_f8->y.z = fmaf(src_f, 0.01234568f, dst_f8->y.z);
    dst_f8->y.w = fmaf(src_f, 0.01234568f, dst_f8->y.w);
    src = src_uchar4[3];
    src_f = rpp_hip_unpack0(src);
    dst_f8->y.x = fmaf(src_f, 0.01234568f, dst_f8->y.x);
    dst_f8->y.y = fmaf(src_f, 0.01234568f, dst_f8->y.y);
    dst_f8->y.z = fmaf(src_f, 0.01234568f, dst_f8->y.z);
    dst_f8->y.w = fmaf(src_f, 0.01234568f, dst_f8->y.w);
    src_f = rpp_hip_unpack1(src);
    dst_f8->y.y = fmaf(src_f, 0.01234568f, dst_f8->y.y);
    dst_f8->y.z = fmaf(src_f, 0.01234568f, dst_f8->y.z);
    dst_f8->y.w = fmaf(src_f, 0.01234568f, dst_f8->y.w);
    src_f = rpp_hip_unpack2(src);
    dst_f8->y.z = fmaf(src_f, 0.01234568f, dst_f8->y.z);
    dst_f8->y.w = fmaf(src_f, 0.01234568f, dst_f8->y.w);
    src_f = rpp_hip_unpack3(src);
    dst_f8->y.w = fmaf(src_f, 0.01234568f, dst_f8->y.w);
}

// template <typename T>
// __global__ void brightness_pkd_tensor(T *srcPtr,
//                                       int nStrideSrc,
//                                       int hStrideSrc,
//                                       T *dstPtr,
//                                       int nStrideDst,
//                                       int hStrideDst,
//                                       float *alpha,
//                                       float *beta,
//                                       RpptROIPtr roiTensorPtrSrc)
// {
//     int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
//     int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
//     int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

//     if ((id_y >= roiTensorPtrSrc[id_z].xywhROI.roiHeight) || (id_x >= roiTensorPtrSrc[id_z].xywhROI.roiWidth * 3))
//     {
//         return;
//     }

//     uint srcIdx = (id_z * nStrideSrc) + ((id_y + roiTensorPtrSrc[id_z].xywhROI.xy.y) * hStrideSrc) + (id_x + roiTensorPtrSrc[id_z].xywhROI.xy.x * 3);
//     uint dstIdx = (id_z * nStrideDst) + (id_y * hStrideDst) + id_x;

//     float4 alpha_f4 = (float4)alpha[id_z];
//     float4 beta_f4 = (float4)beta[id_z];

//     d_float8 src_f8, dst_f8;

//     rpp_hip_load8_and_unpack_to_float8(srcPtr, srcIdx, &src_f8);
//     brightness_hip_compute(srcPtr, &src_f8, &dst_f8, &alpha_f4, &beta_f4);
//     rpp_hip_pack_float8_and_store8(dstPtr, dstIdx, &dst_f8);
// }

// Handles PLN1->PLN1, PLN3->PLN3 for any combination of kernelSize = 3/5/7 and T = U8/F32/F16/I8
template <typename T>
__global__ void box_filter_pln_tensor(T *srcPtr,
                                      int nStrideSrc,
                                      int cStrideSrc,
                                      int hStrideSrc,
                                      T *dstPtr,
                                      int nStrideDst,
                                      int cStrideDst,
                                      int hStrideDst,
                                      int channelsDst,
                                      uint kernelSize,
                                      uint padLength,
                                      uint2 tileSize,
                                      RpptROIPtr roiTensorPtrSrc)
{
    int hipThreadIdx_x8 = hipThreadIdx_x << 3;
    int id_x_o = (hipBlockIdx_x * tileSize.x * 8) + hipThreadIdx_x8;
    int id_y_o = hipBlockIdx_y * tileSize.y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    int id_x_i = id_x_o - padLength;
    int id_y_i = id_y_o - padLength;
    d_float8 sum_f8;
    __shared__ uchar src_lds[16][128];

    uint srcIdx = (id_z * nStrideSrc) + ((id_y_i + roiTensorPtrSrc[id_z].xywhROI.xy.y) * hStrideSrc) + (id_x_i + roiTensorPtrSrc[id_z].xywhROI.xy.x);
    uint dstIdx = (id_z * nStrideDst) + (id_y_o * hStrideDst) + id_x_o;
    sum_f8.x = (float4) 0;
    sum_f8.y = (float4) 0;
    if ((id_x_i >= 0) && (id_x_i < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
        (id_y_i >= 0) && (id_y_i < roiTensorPtrSrc[id_z].xywhROI.roiHeight))
        rpp_hip_lds_load8(srcPtr, srcIdx, &src_lds[hipThreadIdx_y][hipThreadIdx_x8]);
    else
        *(uint2 *)&src_lds[hipThreadIdx_y][hipThreadIdx_x8] = make_uint2(0, 0);
    __syncthreads();
    if ((id_x_o < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
        (id_y_o < roiTensorPtrSrc[id_z].xywhROI.roiHeight) &&
        (hipThreadIdx_x < tileSize.x) &&
        (hipThreadIdx_y < tileSize.y))
    {
        if (kernelSize == 3)
            for(int row = 0; row < 3; row++)
                box_filter_3x3_row_hip_compute(&src_lds[hipThreadIdx_y + row][hipThreadIdx_x8], &sum_f8);
        else if (kernelSize == 5)
            for(int row = 0; row < 5; row++)
                box_filter_5x5_row_hip_compute(&src_lds[hipThreadIdx_y + row][hipThreadIdx_x8], &sum_f8);
        else if (kernelSize == 7)
            for(int row = 0; row < 7; row++)
                box_filter_7x7_row_hip_compute(&src_lds[hipThreadIdx_y + row][hipThreadIdx_x8], &sum_f8);
        else if (kernelSize == 9)
            for(int row = 0; row < 9; row++)
                box_filter_9x9_row_hip_compute(&src_lds[hipThreadIdx_y + row][hipThreadIdx_x8], &sum_f8);
        rpp_hip_adjust_range(dstPtr, &sum_f8);
        rpp_hip_pack_float8_and_store8(dstPtr, dstIdx, &sum_f8);
    }

    if (channelsDst == 3)
    {
        __syncthreads();
        srcIdx += cStrideSrc;
        dstIdx += cStrideDst;
        sum_f8.x = (float4) 0;
        sum_f8.y = (float4) 0;
        if ((id_x_i >= 0) && (id_x_i < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
            (id_y_i >= 0) && (id_y_i < roiTensorPtrSrc[id_z].xywhROI.roiHeight))
            rpp_hip_lds_load8(srcPtr, srcIdx, &src_lds[hipThreadIdx_y][hipThreadIdx_x8]);
        else
            *(uint2 *)&src_lds[hipThreadIdx_y][hipThreadIdx_x8] = make_uint2(0, 0);
        __syncthreads();
        if ((id_x_o < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
            (id_y_o < roiTensorPtrSrc[id_z].xywhROI.roiHeight) &&
            (hipThreadIdx_x < tileSize.x) &&
            (hipThreadIdx_y < tileSize.y))
        {
            if (kernelSize == 3)
                for(int row = 0; row < 3; row++)
                    box_filter_3x3_row_hip_compute(&src_lds[hipThreadIdx_y + row][hipThreadIdx_x8], &sum_f8);
            else if (kernelSize == 5)
                for(int row = 0; row < 5; row++)
                    box_filter_5x5_row_hip_compute(&src_lds[hipThreadIdx_y + row][hipThreadIdx_x8], &sum_f8);
            else if (kernelSize == 7)
                for(int row = 0; row < 7; row++)
                    box_filter_7x7_row_hip_compute(&src_lds[hipThreadIdx_y + row][hipThreadIdx_x8], &sum_f8);
            else if (kernelSize == 9)
                for(int row = 0; row < 9; row++)
                    box_filter_9x9_row_hip_compute(&src_lds[hipThreadIdx_y + row][hipThreadIdx_x8], &sum_f8);
            rpp_hip_adjust_range(dstPtr, &sum_f8);
            rpp_hip_pack_float8_and_store8(dstPtr, dstIdx, &sum_f8);
        }

        __syncthreads();
        srcIdx += cStrideSrc;
        dstIdx += cStrideDst;
        sum_f8.x = (float4) 0;
        sum_f8.y = (float4) 0;
        if ((id_x_i >= 0) && (id_x_i < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
            (id_y_i >= 0) && (id_y_i < roiTensorPtrSrc[id_z].xywhROI.roiHeight))
            rpp_hip_lds_load8(srcPtr, srcIdx, &src_lds[hipThreadIdx_y][hipThreadIdx_x8]);
        else
            *(uint2 *)&src_lds[hipThreadIdx_y][hipThreadIdx_x8] = make_uint2(0, 0);
        __syncthreads();
        if ((id_x_o < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
            (id_y_o < roiTensorPtrSrc[id_z].xywhROI.roiHeight) &&
            (hipThreadIdx_x < tileSize.x) &&
            (hipThreadIdx_y < tileSize.y))
        {
            if (kernelSize == 3)
                for(int row = 0; row < 3; row++)
                    box_filter_3x3_row_hip_compute(&src_lds[hipThreadIdx_y + row][hipThreadIdx_x8], &sum_f8);
            else if (kernelSize == 5)
                for(int row = 0; row < 5; row++)
                    box_filter_5x5_row_hip_compute(&src_lds[hipThreadIdx_y + row][hipThreadIdx_x8], &sum_f8);
            else if (kernelSize == 7)
                for(int row = 0; row < 7; row++)
                    box_filter_7x7_row_hip_compute(&src_lds[hipThreadIdx_y + row][hipThreadIdx_x8], &sum_f8);
            else if (kernelSize == 9)
                for(int row = 0; row < 9; row++)
                    box_filter_9x9_row_hip_compute(&src_lds[hipThreadIdx_y + row][hipThreadIdx_x8], &sum_f8);
            rpp_hip_adjust_range(dstPtr, &sum_f8);
            rpp_hip_pack_float8_and_store8(dstPtr, dstIdx, &sum_f8);
        }
    }
}

// template <typename T>
// __global__ void brightness_pkd3_pln3_tensor(T *srcPtr,
//                                             int nStrideSrc,
//                                             int hStrideSrc,
//                                             T *dstPtr,
//                                             int nStrideDst,
//                                             int cStrideDst,
//                                             int hStrideDst,
//                                             float *alpha,
//                                             float *beta,
//                                             RpptROIPtr roiTensorPtrSrc)
// {
//     int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
//     int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
//     int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

//     if ((id_y >= roiTensorPtrSrc[id_z].xywhROI.roiHeight) || (id_x >= roiTensorPtrSrc[id_z].xywhROI.roiWidth))
//     {
//         return;
//     }

//     uint srcIdx = (id_z * nStrideSrc) + ((id_y + roiTensorPtrSrc[id_z].xywhROI.xy.y) * hStrideSrc) + ((id_x + roiTensorPtrSrc[id_z].xywhROI.xy.x) * 3);
//     uint dstIdx = (id_z * nStrideDst) + (id_y * hStrideDst) + id_x;

//     float4 alpha_f4 = (float4)alpha[id_z];
//     float4 beta_f4 = (float4)beta[id_z];

//     d_float24 src_f24, dst_f24;

//     rpp_hip_load24_pkd3_and_unpack_to_float24_pln3(srcPtr, srcIdx, &src_f24);
//     brightness_hip_compute(srcPtr, &src_f24.x, &dst_f24.x, &alpha_f4, &beta_f4);
//     rpp_hip_pack_float8_and_store8(dstPtr, dstIdx, &dst_f24.x);

//     dstIdx += cStrideDst;

//     brightness_hip_compute(srcPtr, &src_f24.y, &dst_f24.y, &alpha_f4, &beta_f4);
//     rpp_hip_pack_float8_and_store8(dstPtr, dstIdx, &dst_f24.y);

//     dstIdx += cStrideDst;

//     brightness_hip_compute(srcPtr, &src_f24.z, &dst_f24.z, &alpha_f4, &beta_f4);
//     rpp_hip_pack_float8_and_store8(dstPtr, dstIdx, &dst_f24.z);
// }

// template <typename T>
// __global__ void brightness_pln3_pkd3_tensor(T *srcPtr,
//                                             int nStrideSrc,
//                                             int cStrideSrc,
//                                             int hStrideSrc,
//                                             T *dstPtr,
//                                             int nStrideDst,
//                                             int hStrideDst,
//                                             float *alpha,
//                                             float *beta,
//                                             RpptROIPtr roiTensorPtrSrc)
// {
//     int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
//     int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
//     int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

//     if ((id_y >= roiTensorPtrSrc[id_z].xywhROI.roiHeight) || (id_x >= roiTensorPtrSrc[id_z].xywhROI.roiWidth))
//     {
//         return;
//     }

//     uint srcIdx = (id_z * nStrideSrc) + ((id_y + roiTensorPtrSrc[id_z].xywhROI.xy.y) * hStrideSrc) + (id_x + roiTensorPtrSrc[id_z].xywhROI.xy.x);
//     uint dstIdx = (id_z * nStrideDst) + (id_y * hStrideDst) + id_x * 3;

//     float4 alpha_f4 = (float4)(alpha[id_z]);
//     float4 beta_f4 = (float4)(beta[id_z]);

//     d_float24 src_f24, dst_f24;

//     rpp_hip_load24_pln3_and_unpack_to_float24_pkd3(srcPtr, srcIdx, cStrideSrc, &src_f24);
//     brightness_hip_compute(srcPtr, &src_f24.x, &dst_f24.x, &alpha_f4, &beta_f4);
//     brightness_hip_compute(srcPtr, &src_f24.y, &dst_f24.y, &alpha_f4, &beta_f4);
//     brightness_hip_compute(srcPtr, &src_f24.z, &dst_f24.z, &alpha_f4, &beta_f4);
//     rpp_hip_pack_float24_and_store24(dstPtr, dstIdx, &dst_f24);
// }

template <typename T>
RppStatus hip_exec_box_filter_tensor(T *srcPtr,
                                     RpptDescPtr srcDescPtr,
                                     T *dstPtr,
                                     RpptDescPtr dstDescPtr,
                                     Rpp32u kernelSize,
                                     RpptROIPtr roiTensorPtrSrc,
                                     rpp::Handle& handle)
{
    int localThreads_x = 16;
    int localThreads_y = 16;
    int localThreads_z = 1;
    int globalThreads_x = (dstDescPtr->strides.hStride + 7) >> 3;
    int globalThreads_y = dstDescPtr->h;
    int globalThreads_z = handle.GetBatchSize();

    uint padLength = kernelSize / 2;
    uint padLengthTwice = padLength * 2;
    uint2 tileSize;
    tileSize.x = (128 - padLengthTwice) / 8;
    tileSize.y = 16 - padLengthTwice;

    // if ((srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NHWC))
    // {
    //     hipLaunchKernelGGL(brightness_pkd_tensor,
    //                        dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
    //                        dim3(localThreads_x, localThreads_y, localThreads_z),
    //                        0,
    //                        handle.GetStream(),
    //                        srcPtr,
    //                        srcDescPtr->strides.nStride,
    //                        srcDescPtr->strides.hStride,
    //                        dstPtr,
    //                        dstDescPtr->strides.nStride,
    //                        dstDescPtr->strides.hStride,
    //                        handle.GetInitHandle()->mem.mgpu.floatArr[0].floatmem,
    //                        handle.GetInitHandle()->mem.mgpu.floatArr[1].floatmem,
    //                        roiTensorPtrSrc);
    // }
    // else if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))





    if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
    {
        hipLaunchKernelGGL(box_filter_pln_tensor,
                           dim3(ceil((float)globalThreads_x/tileSize.x), ceil((float)globalThreads_y/tileSize.y), ceil((float)globalThreads_z/localThreads_z)),
                           dim3(localThreads_x, localThreads_y, localThreads_z),
                           0,
                           handle.GetStream(),
                           srcPtr,
                           srcDescPtr->strides.nStride,
                           srcDescPtr->strides.cStride,
                           srcDescPtr->strides.hStride,
                           dstPtr,
                           dstDescPtr->strides.nStride,
                           dstDescPtr->strides.cStride,
                           dstDescPtr->strides.hStride,
                           dstDescPtr->c,
                           kernelSize,
                           padLength,
                           tileSize,
                           roiTensorPtrSrc);
    }





    // else if ((srcDescPtr->c == 3) && (dstDescPtr->c == 3))
    // {
    //     if ((srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
    //     {
    //         hipLaunchKernelGGL(brightness_pkd3_pln3_tensor,
    //                            dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
    //                            dim3(localThreads_x, localThreads_y, localThreads_z),
    //                            0,
    //                            handle.GetStream(),
    //                            srcPtr,
    //                            srcDescPtr->strides.nStride,
    //                            srcDescPtr->strides.hStride,
    //                            dstPtr,
    //                            dstDescPtr->strides.nStride,
    //                            dstDescPtr->strides.cStride,
    //                            dstDescPtr->strides.hStride,
    //                            handle.GetInitHandle()->mem.mgpu.floatArr[0].floatmem,
    //                            handle.GetInitHandle()->mem.mgpu.floatArr[1].floatmem,
    //                            roiTensorPtrSrc);
    //     }
    //     else if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
    //     {
    //         globalThreads_x = (srcDescPtr->strides.hStride + 7) >> 3;
    //         hipLaunchKernelGGL(brightness_pln3_pkd3_tensor,
    //                            dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
    //                            dim3(localThreads_x, localThreads_y, localThreads_z),
    //                            0,
    //                            handle.GetStream(),
    //                            srcPtr,
    //                            srcDescPtr->strides.nStride,
    //                            srcDescPtr->strides.cStride,
    //                            srcDescPtr->strides.hStride,
    //                            dstPtr,
    //                            dstDescPtr->strides.nStride,
    //                            dstDescPtr->strides.hStride,
    //                            handle.GetInitHandle()->mem.mgpu.floatArr[0].floatmem,
    //                            handle.GetInitHandle()->mem.mgpu.floatArr[1].floatmem,
    //                            roiTensorPtrSrc);
    //     }
    // }

    return RPP_SUCCESS;
}
