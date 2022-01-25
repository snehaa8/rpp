#include <hip/hip_runtime.h>
#include "hip/rpp_hip_common.hpp"
#include "func_specific/spatter_mask.hpp"
#include <random>

__device__ void spatter_hip_compute(uchar *srcPtr, d_float8 *src_f8, d_float8 *dst_f8, d_float8 *mask_f8, d_float8 *maskInv_f8, float4 *pix_f4)
{
    dst_f8->x = (src_f8->x * maskInv_f8->x) + (*pix_f4 * mask_f8->x);
    dst_f8->y = (src_f8->y * maskInv_f8->y) + (*pix_f4 * mask_f8->y);
}

__device__ void spatter_hip_compute(float *srcPtr, d_float8 *src_f8, d_float8 *dst_f8, d_float8 *mask_f8, d_float8 *maskInv_f8, float4 *pix_f4)
{
    float4 pixNorm_f4 = *pix_f4 * (float4) ONE_OVER_255;
    dst_f8->x = (src_f8->x * maskInv_f8->x) + (pixNorm_f4 * mask_f8->x);
    dst_f8->y = (src_f8->y * maskInv_f8->y) + (pixNorm_f4 * mask_f8->y);
}

__device__ void spatter_hip_compute(schar *srcPtr, d_float8 *src_f8, d_float8 *dst_f8, d_float8 *mask_f8, d_float8 *maskInv_f8, float4 *pix_f4)
{
    dst_f8->x = ((src_f8->x + (float4)128) * maskInv_f8->x) + (*pix_f4 * mask_f8->x) - (float4)128;
    dst_f8->y = ((src_f8->y + (float4)128) * maskInv_f8->y) + (*pix_f4 * mask_f8->y) - (float4)128;
}

__device__ void spatter_hip_compute(half *srcPtr, d_float8 *src_f8, d_float8 *dst_f8, d_float8 *mask_f8, d_float8 *maskInv_f8, float4 *pix_f4)
{
    float4 pixNorm_f4 = *pix_f4 * (float4) ONE_OVER_255;
    dst_f8->x = (src_f8->x * maskInv_f8->x) + (pixNorm_f4 * mask_f8->x);
    dst_f8->y = (src_f8->y * maskInv_f8->y) + (pixNorm_f4 * mask_f8->y);
}

template <typename T>
__global__ void spatter_pkd_tensor(T *srcPtr,
                                   uint2 srcStridesNH,
                                   T *dstPtr,
                                   uint2 dstStridesNH,
                                   float *spatterMaskPtr,
                                   float *spatterMaskInvPtr,
                                   uint2 *maskLocArr,
                                   float3 spatterColor,
                                   RpptROIPtr roiTensorPtrSrc)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if ((id_y >= roiTensorPtrSrc[id_z].xywhROI.roiHeight) || (id_x >= roiTensorPtrSrc[id_z].xywhROI.roiWidth))
    {
        return;
    }

    uint srcIdx = (id_z * srcStridesNH.x) + ((id_y + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNH.y) + ((id_x + roiTensorPtrSrc[id_z].xywhROI.xy.x) * 3);
    uint dstIdx = (id_z * dstStridesNH.x) + (id_y * dstStridesNH.y) + id_x * 3;
    uint maskIdx = (SPATTER_MAX_WIDTH * (maskLocArr[id_z].y + id_y)) + maskLocArr[id_z].x + id_x;

    d_float8 mask_f8, maskInv_f8;
    mask_f8 = *(d_float8 *)&spatterMaskPtr[maskIdx];
    maskInv_f8 = *(d_float8 *)&spatterMaskInvPtr[maskIdx];
    float4 r_f4 = (float4)(spatterColor.x);
    float4 g_f4 = (float4)(spatterColor.y);
    float4 b_f4 = (float4)(spatterColor.z);

    d_float24 src_f24, dst_f24;

    rpp_hip_load24_pkd3_and_unpack_to_float24_pln3(srcPtr, srcIdx, &src_f24);
    spatter_hip_compute(srcPtr, &src_f24.x, &dst_f24.x, &mask_f8, &maskInv_f8, &r_f4);
    spatter_hip_compute(srcPtr, &src_f24.y, &dst_f24.y, &mask_f8, &maskInv_f8, &g_f4);
    spatter_hip_compute(srcPtr, &src_f24.z, &dst_f24.z, &mask_f8, &maskInv_f8, &b_f4);
    rpp_hip_pack_float24_pln3_and_store24_pkd3(dstPtr, dstIdx, &dst_f24);
}

template <typename T>
__global__ void spatter_pln_tensor(T *srcPtr,
                                   uint3 srcStridesNCH,
                                   T *dstPtr,
                                   uint3 dstStridesNCH,
                                   int channelsDst,
                                   float *spatterMaskPtr,
                                   float *spatterMaskInvPtr,
                                   uint2 *maskLocArr,
                                   float3 spatterColor,
                                   RpptROIPtr roiTensorPtrSrc)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if ((id_y >= roiTensorPtrSrc[id_z].xywhROI.roiHeight) || (id_x >= roiTensorPtrSrc[id_z].xywhROI.roiWidth))
    {
        return;
    }

    uint srcIdx = (id_z * srcStridesNCH.x) + ((id_y + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNCH.z) + (id_x + roiTensorPtrSrc[id_z].xywhROI.xy.x);
    uint dstIdx = (id_z * dstStridesNCH.x) + (id_y * dstStridesNCH.z) + id_x;
    uint maskIdx = (SPATTER_MAX_WIDTH * (maskLocArr[id_z].y + id_y)) + maskLocArr[id_z].x + id_x;

    d_float8 mask_f8, maskInv_f8;
    mask_f8 = *(d_float8 *)&spatterMaskPtr[maskIdx];
    maskInv_f8 = *(d_float8 *)&spatterMaskInvPtr[maskIdx];
    float4 r_f4 = (float4)(spatterColor.x);
    float4 g_f4 = (float4)(spatterColor.y);
    float4 b_f4 = (float4)(spatterColor.z);

    d_float8 src_f8, dst_f8;

    rpp_hip_load8_and_unpack_to_float8(srcPtr, srcIdx, &src_f8);
    spatter_hip_compute(srcPtr, &src_f8, &dst_f8, &mask_f8, &maskInv_f8, &r_f4);
    rpp_hip_pack_float8_and_store8(dstPtr, dstIdx, &dst_f8);

    if (channelsDst == 3)
    {
        srcIdx += srcStridesNCH.y;
        dstIdx += dstStridesNCH.y;

        rpp_hip_load8_and_unpack_to_float8(srcPtr, srcIdx, &src_f8);
        spatter_hip_compute(srcPtr, &src_f8, &dst_f8, &mask_f8, &maskInv_f8, &g_f4);
        rpp_hip_pack_float8_and_store8(dstPtr, dstIdx, &dst_f8);

        srcIdx += srcStridesNCH.y;
        dstIdx += dstStridesNCH.y;

        rpp_hip_load8_and_unpack_to_float8(srcPtr, srcIdx, &src_f8);
        spatter_hip_compute(srcPtr, &src_f8, &dst_f8, &mask_f8, &maskInv_f8, &b_f4);
        rpp_hip_pack_float8_and_store8(dstPtr, dstIdx, &dst_f8);
    }
}

template <typename T>
__global__ void spatter_pkd3_pln3_tensor(T *srcPtr,
                                         uint2 srcStridesNH,
                                         T *dstPtr,
                                         uint3 dstStridesNCH,
                                         float *spatterMaskPtr,
                                         float *spatterMaskInvPtr,
                                         uint2 *maskLocArr,
                                         float3 spatterColor,
                                         RpptROIPtr roiTensorPtrSrc)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if ((id_y >= roiTensorPtrSrc[id_z].xywhROI.roiHeight) || (id_x >= roiTensorPtrSrc[id_z].xywhROI.roiWidth))
    {
        return;
    }

    uint srcIdx = (id_z * srcStridesNH.x) + ((id_y + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNH.y) + ((id_x + roiTensorPtrSrc[id_z].xywhROI.xy.x) * 3);
    uint dstIdx = (id_z * dstStridesNCH.x) + (id_y * dstStridesNCH.z) + id_x;
    uint maskIdx = (SPATTER_MAX_WIDTH * (maskLocArr[id_z].y + id_y)) + maskLocArr[id_z].x + id_x;

    d_float8 mask_f8, maskInv_f8;
    mask_f8 = *(d_float8 *)&spatterMaskPtr[maskIdx];
    maskInv_f8 = *(d_float8 *)&spatterMaskInvPtr[maskIdx];
    float4 r_f4 = (float4)(spatterColor.x);
    float4 g_f4 = (float4)(spatterColor.y);
    float4 b_f4 = (float4)(spatterColor.z);

    d_float24 src_f24, dst_f24;

    rpp_hip_load24_pkd3_and_unpack_to_float24_pln3(srcPtr, srcIdx, &src_f24);
    spatter_hip_compute(srcPtr, &src_f24.x, &dst_f24.x, &mask_f8, &maskInv_f8, &r_f4);
    spatter_hip_compute(srcPtr, &src_f24.y, &dst_f24.y, &mask_f8, &maskInv_f8, &g_f4);
    spatter_hip_compute(srcPtr, &src_f24.z, &dst_f24.z, &mask_f8, &maskInv_f8, &b_f4);
    rpp_hip_pack_float24_pln3_and_store24_pln3(dstPtr, dstIdx, dstStridesNCH.y, &dst_f24);
}

template <typename T>
__global__ void spatter_pln3_pkd3_tensor(T *srcPtr,
                                         uint3 srcStridesNCH,
                                         T *dstPtr,
                                         uint2 dstStridesNH,
                                         float *spatterMaskPtr,
                                         float *spatterMaskInvPtr,
                                         uint2 *maskLocArr,
                                         float3 spatterColor,
                                         RpptROIPtr roiTensorPtrSrc)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if ((id_y >= roiTensorPtrSrc[id_z].xywhROI.roiHeight) || (id_x >= roiTensorPtrSrc[id_z].xywhROI.roiWidth))
    {
        return;
    }

    uint srcIdx = (id_z * srcStridesNCH.x) + ((id_y + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNCH.z) + (id_x + roiTensorPtrSrc[id_z].xywhROI.xy.x);
    uint dstIdx = (id_z * dstStridesNH.x) + (id_y * dstStridesNH.y) + id_x * 3;
    uint maskIdx = (SPATTER_MAX_WIDTH * (maskLocArr[id_z].y + id_y)) + maskLocArr[id_z].x + id_x;

    d_float8 mask_f8, maskInv_f8;
    mask_f8 = *(d_float8 *)&spatterMaskPtr[maskIdx];
    maskInv_f8 = *(d_float8 *)&spatterMaskInvPtr[maskIdx];
    float4 r_f4 = (float4)(spatterColor.x);
    float4 g_f4 = (float4)(spatterColor.y);
    float4 b_f4 = (float4)(spatterColor.z);

    d_float24 src_f24, dst_f24;

    rpp_hip_load24_pln3_and_unpack_to_float24_pln3(srcPtr, srcIdx, srcStridesNCH.y, &src_f24);
    spatter_hip_compute(srcPtr, &src_f24.x, &dst_f24.x, &mask_f8, &maskInv_f8, &r_f4);
    spatter_hip_compute(srcPtr, &src_f24.y, &dst_f24.y, &mask_f8, &maskInv_f8, &g_f4);
    spatter_hip_compute(srcPtr, &src_f24.z, &dst_f24.z, &mask_f8, &maskInv_f8, &b_f4);
    rpp_hip_pack_float24_pln3_and_store24_pkd3(dstPtr, dstIdx, &dst_f24);
}

template <typename T>
RppStatus hip_exec_spatter_tensor(T *srcPtr,
                                  RpptDescPtr srcDescPtr,
                                  T *dstPtr,
                                  RpptDescPtr dstDescPtr,
                                  uint2 *maskLocArr,
                                  RpptRGB spatterColor,
                                  RpptROIPtr roiTensorPtrSrc,
                                  rpp::Handle& handle)
{
    int localThreads_x = 16;
    int localThreads_y = 16;
    int localThreads_z = 1;
    int globalThreads_x = (dstDescPtr->strides.hStride + 7) >> 3;
    int globalThreads_y = dstDescPtr->h;
    int globalThreads_z = handle.GetBatchSize();

    float3 spatterColor_f3;
    if (dstDescPtr->c == 3)
    {
        spatterColor_f3 = make_float3((float)spatterColor.B, (float)spatterColor.G, (float)spatterColor.R);
    }
    else if (dstDescPtr->c == 1)
    {
        float meanGreyVal = ((float)spatterColor.B + (float)spatterColor.G + (float)spatterColor.R) * 0.3333;
        spatterColor_f3 = make_float3(meanGreyVal, meanGreyVal, meanGreyVal);
    }

    float *spatterMaskPtr, *spatterMaskInvPtr;
    Rpp32u maskSize = 2073600 * sizeof(float);
    hipMalloc(&spatterMaskPtr, maskSize);
    hipMalloc(&spatterMaskInvPtr, maskSize);
    hipMemcpy(spatterMaskPtr, spatterMask, maskSize, hipMemcpyHostToDevice);
    hipMemcpy(spatterMaskInvPtr, spatterMaskInv, maskSize, hipMemcpyHostToDevice);

    if ((srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NHWC))
    {
        globalThreads_x = (dstDescPtr->strides.hStride / 3 + 7) >> 3;
        hipLaunchKernelGGL(spatter_pkd_tensor,
                           dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
                           dim3(localThreads_x, localThreads_y, localThreads_z),
                           0,
                           handle.GetStream(),
                           srcPtr,
                           make_uint2(srcDescPtr->strides.nStride, srcDescPtr->strides.hStride),
                           dstPtr,
                           make_uint2(dstDescPtr->strides.nStride, dstDescPtr->strides.hStride),
                           spatterMaskPtr,
                           spatterMaskInvPtr,
                           maskLocArr,
                           spatterColor_f3,
                           roiTensorPtrSrc);
    }
    else if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
    {
        hipLaunchKernelGGL(spatter_pln_tensor,
                           dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
                           dim3(localThreads_x, localThreads_y, localThreads_z),
                           0,
                           handle.GetStream(),
                           srcPtr,
                           make_uint3(srcDescPtr->strides.nStride, srcDescPtr->strides.cStride, srcDescPtr->strides.hStride),
                           dstPtr,
                           make_uint3(dstDescPtr->strides.nStride, dstDescPtr->strides.cStride, dstDescPtr->strides.hStride),
                           dstDescPtr->c,
                           spatterMaskPtr,
                           spatterMaskInvPtr,
                           maskLocArr,
                           spatterColor_f3,
                           roiTensorPtrSrc);
    }
    else if ((srcDescPtr->c == 3) && (dstDescPtr->c == 3))
    {
        if ((srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            hipLaunchKernelGGL(spatter_pkd3_pln3_tensor,
                               dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
                               dim3(localThreads_x, localThreads_y, localThreads_z),
                               0,
                               handle.GetStream(),
                               srcPtr,
                               make_uint2(srcDescPtr->strides.nStride, srcDescPtr->strides.hStride),
                               dstPtr,
                               make_uint3(dstDescPtr->strides.nStride, dstDescPtr->strides.cStride, dstDescPtr->strides.hStride),
                               spatterMaskPtr,
                               spatterMaskInvPtr,
                               maskLocArr,
                               spatterColor_f3,
                               roiTensorPtrSrc);
        }
        else if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            globalThreads_x = (srcDescPtr->strides.hStride + 7) >> 3;
            hipLaunchKernelGGL(spatter_pln3_pkd3_tensor,
                               dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
                               dim3(localThreads_x, localThreads_y, localThreads_z),
                               0,
                               handle.GetStream(),
                               srcPtr,
                               make_uint3(srcDescPtr->strides.nStride, srcDescPtr->strides.cStride, srcDescPtr->strides.hStride),
                               dstPtr,
                               make_uint2(dstDescPtr->strides.nStride, dstDescPtr->strides.hStride),
                               spatterMaskPtr,
                               spatterMaskInvPtr,
                               maskLocArr,
                               spatterColor_f3,
                               roiTensorPtrSrc);
        }
    }

    hipFree(&spatterMaskPtr);
    hipFree(&spatterMaskInvPtr);

    return RPP_SUCCESS;
}
