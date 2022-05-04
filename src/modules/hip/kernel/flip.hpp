#include <hip/hip_runtime.h>
#include "hip/rpp_hip_common.hpp"

template <typename T>
__global__ void flip_pkd_tensor(T *srcPtr,
                                uint2 srcStridesNH,
                                T *dstPtr,
                                uint2 dstStridesNH,
                                uint2 dstDimsWH,
                                unsigned int *horizontalTensor,
                                unsigned int *verticalTensor,
                                RpptROIPtr roiTensorPtrSrc)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if ((id_y >= dstDimsWH.y) || (id_x >= dstDimsWH.x))
    {
        return;
    }

    uint horizontalFlag = horizontalTensor[id_z];
    uint verticalFlag = verticalTensor[id_z];
    uint srcIdx = (id_z * srcStridesNH.x);
    uint hFactor = (horizontalFlag) ? (roiTensorPtrSrc[id_z].ltrbROI.rb.x - id_x - 7) * 3 : (id_x + roiTensorPtrSrc[id_z].ltrbROI.lt.x) * 3;
    uint vFactor = (verticalFlag) ? (roiTensorPtrSrc[id_z].ltrbROI.rb.y - id_y) * srcStridesNH.y : (id_y + roiTensorPtrSrc[id_z].ltrbROI.lt.y) * srcStridesNH.y;
    srcIdx += hFactor + vFactor;

    d_float24 pix_f24;
    if(horizontalFlag)
        rpp_hip_load24_pkd3_and_unpack_to_float24_pln3_mirror(srcPtr + srcIdx, &pix_f24);
    else
        rpp_hip_load24_pkd3_and_unpack_to_float24_pln3(srcPtr + srcIdx, &pix_f24);
    
    uint dstIdx = (id_z * dstStridesNH.x) + (id_y * dstStridesNH.y) + id_x * 3;
    rpp_hip_pack_float24_pln3_and_store24_pkd3(dstPtr + dstIdx, &pix_f24);
}

template <typename T>
__global__ void flip_pln_tensor(T *srcPtr,
                                uint3 srcStridesNCH,
                                T *dstPtr,
                                uint3 dstStridesNCH,
                                uint2 dstDimsWH,
                                int channelsDst,
                                unsigned int *horizontalTensor,
                                unsigned int *verticalTensor,
                                RpptROIPtr roiTensorPtrSrc)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if ((id_y >= dstDimsWH.y) || (id_x >= dstDimsWH.x))
    {
        return;
    }

    uint horizontalFlag = horizontalTensor[id_z];
    uint verticalFlag = verticalTensor[id_z];
    uint srcIdx = (id_z * srcStridesNCH.x);
    uint hFactor = (horizontalFlag) ? (roiTensorPtrSrc[id_z].ltrbROI.rb.x - id_x - 7) : (id_x + roiTensorPtrSrc[id_z].ltrbROI.lt.x);
    uint vFactor = (verticalFlag) ? (roiTensorPtrSrc[id_z].ltrbROI.rb.y - id_y) * srcStridesNCH.z : (id_y + roiTensorPtrSrc[id_z].ltrbROI.lt.y) * srcStridesNCH.z;
    srcIdx += hFactor + vFactor;
    uint dstIdx = (id_z * dstStridesNCH.x) + (id_y * dstStridesNCH.z) + id_x;
    
    d_float8 pix_f8;
    if(horizontalFlag)
    {
        rpp_hip_load8_and_unpack_to_float8_mirror(srcPtr + srcIdx, &pix_f8);
        rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &pix_f8);

        if (channelsDst == 3)
        {
            srcIdx += srcStridesNCH.y;
            dstIdx += dstStridesNCH.y;

            rpp_hip_load8_and_unpack_to_float8_mirror(srcPtr + srcIdx, &pix_f8);
            rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &pix_f8);

            srcIdx += srcStridesNCH.y;
            dstIdx += dstStridesNCH.y;

            rpp_hip_load8_and_unpack_to_float8_mirror(srcPtr + srcIdx, &pix_f8);
            rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &pix_f8);
        }
    }
    else
    {
        rpp_hip_load8_and_unpack_to_float8(srcPtr + srcIdx, &pix_f8);
        rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &pix_f8);

        if (channelsDst == 3)
        {
            srcIdx += srcStridesNCH.y;
            dstIdx += dstStridesNCH.y;

            rpp_hip_load8_and_unpack_to_float8(srcPtr + srcIdx, &pix_f8);
            rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &pix_f8);

            srcIdx += srcStridesNCH.y;
            dstIdx += dstStridesNCH.y;

            rpp_hip_load8_and_unpack_to_float8(srcPtr + srcIdx, &pix_f8);
            rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &pix_f8);
        }
    }
}

template <typename T>
__global__ void flip_pkd3_pln3_tensor(T *srcPtr,
                                      uint2 srcStridesNH,
                                      T *dstPtr,
                                      uint3 dstStridesNCH,
                                      uint2 dstDimsWH,
                                      unsigned int *horizontalTensor,
                                      unsigned int *verticalTensor,
                                      RpptROIPtr roiTensorPtrSrc)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if ((id_y >= dstDimsWH.y) || (id_x >= dstDimsWH.x))
    {
        return;
    }

    uint horizontalFlag = horizontalTensor[id_z];
    uint verticalFlag = verticalTensor[id_z];
    uint srcIdx = (id_z * srcStridesNH.x);
    uint hFactor = (horizontalFlag) ? (roiTensorPtrSrc[id_z].ltrbROI.rb.x - id_x - 7) * 3 : (id_x + roiTensorPtrSrc[id_z].ltrbROI.lt.x) * 3;
    uint vFactor = (verticalFlag) ? (roiTensorPtrSrc[id_z].ltrbROI.rb.y - id_y) * srcStridesNH.y : (id_y + roiTensorPtrSrc[id_z].ltrbROI.lt.y) * srcStridesNH.y;
    srcIdx += hFactor + vFactor;

    d_float24 pix_f24;
    if(horizontalFlag)
        rpp_hip_load24_pkd3_and_unpack_to_float24_pln3_mirror(srcPtr + srcIdx, &pix_f24);
    else
        rpp_hip_load24_pkd3_and_unpack_to_float24_pln3(srcPtr + srcIdx, &pix_f24);

    uint dstIdx = (id_z * dstStridesNCH.x) + (id_y * dstStridesNCH.z) + id_x;
    rpp_hip_pack_float24_pln3_and_store24_pln3(dstPtr + dstIdx, dstStridesNCH.y, &pix_f24);
}

template <typename T>
__global__ void flip_pln3_pkd3_tensor(T *srcPtr,
                                      uint3 srcStridesNCH,
                                      T *dstPtr,
                                      uint2 dstStridesNH,
                                      uint2 dstDimsWH,
                                      unsigned int *horizontalTensor,
                                      unsigned int *verticalTensor,
                                      RpptROIPtr roiTensorPtrSrc)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if ((id_y >= dstDimsWH.y) || (id_x >= dstDimsWH.x))
    {
        return;
    }
    
    uint horizontalFlag = horizontalTensor[id_z];
    uint verticalFlag = verticalTensor[id_z];
    uint srcIdx = (id_z * srcStridesNCH.x);
    uint hFactor = (horizontalFlag) ? (roiTensorPtrSrc[id_z].ltrbROI.rb.x - id_x - 7) : (id_x + roiTensorPtrSrc[id_z].ltrbROI.lt.x);
    uint vFactor = (verticalFlag) ? (roiTensorPtrSrc[id_z].ltrbROI.rb.y - id_y) * srcStridesNCH.z : (id_y + roiTensorPtrSrc[id_z].ltrbROI.lt.y) * srcStridesNCH.z;
    srcIdx += hFactor + vFactor;

    d_float24 pix_f24;
    if(horizontalFlag)
        rpp_hip_load24_pln3_and_unpack_to_float24_pln3_mirror(srcPtr + srcIdx, srcStridesNCH.y, &pix_f24);
    else
        rpp_hip_load24_pln3_and_unpack_to_float24_pln3(srcPtr + srcIdx, srcStridesNCH.y, &pix_f24);

    uint dstIdx = (id_z * dstStridesNH.x) + (id_y * dstStridesNH.y) + id_x * 3;
    rpp_hip_pack_float24_pln3_and_store24_pkd3(dstPtr + dstIdx, &pix_f24);
}

template <typename T>
RppStatus hip_exec_flip_tensor(T *srcPtr,
                               RpptDescPtr srcDescPtr,
                               T *dstPtr,
                               RpptDescPtr dstDescPtr,
                               RpptROIPtr roiTensorPtrSrc,
                               RpptRoiType roiType,
                               rpp::Handle& handle)
{
    if (roiType == RpptRoiType::XYWH)
        hip_exec_roi_converison_xywh_to_ltrb(roiTensorPtrSrc, handle);

    int localThreads_x = LOCAL_THREADS_X;
    int localThreads_y = LOCAL_THREADS_Y;
    int localThreads_z = LOCAL_THREADS_Z;
    int globalThreads_x = (dstDescPtr->strides.hStride + 7) >> 3;
    int globalThreads_y = dstDescPtr->h;
    int globalThreads_z = handle.GetBatchSize();

    if ((srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NHWC))
    {
        hipLaunchKernelGGL(flip_pkd_tensor,
                           dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
                           dim3(localThreads_x, localThreads_y, localThreads_z),
                           0,
                           handle.GetStream(),
                           srcPtr,
                           make_uint2(srcDescPtr->strides.nStride, srcDescPtr->strides.hStride),
                           dstPtr,
                           make_uint2(dstDescPtr->strides.nStride, dstDescPtr->strides.hStride),
                           make_uint2(dstDescPtr->w - 8, dstDescPtr->h),
                           handle.GetInitHandle()->mem.mgpu.uintArr[0].uintmem,
                           handle.GetInitHandle()->mem.mgpu.uintArr[1].uintmem,
                           roiTensorPtrSrc);
    }
    else if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
    {
        hipLaunchKernelGGL(flip_pln_tensor,
                           dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
                           dim3(localThreads_x, localThreads_y, localThreads_z),
                           0,
                           handle.GetStream(),
                           srcPtr,
                           make_uint3(srcDescPtr->strides.nStride, srcDescPtr->strides.cStride, srcDescPtr->strides.hStride),
                           dstPtr,
                           make_uint3(dstDescPtr->strides.nStride, dstDescPtr->strides.cStride, dstDescPtr->strides.hStride),
                           make_uint2(dstDescPtr->w - 8, dstDescPtr->h),
                           dstDescPtr->c,
                           handle.GetInitHandle()->mem.mgpu.uintArr[0].uintmem,
                           handle.GetInitHandle()->mem.mgpu.uintArr[1].uintmem,
                           roiTensorPtrSrc);
    }
    else if ((srcDescPtr->c == 3) && (dstDescPtr->c == 3))
    {
        if ((srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            hipLaunchKernelGGL(flip_pkd3_pln3_tensor,
                               dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
                               dim3(localThreads_x, localThreads_y, localThreads_z),
                               0,
                               handle.GetStream(),
                               srcPtr,
                               make_uint2(srcDescPtr->strides.nStride, srcDescPtr->strides.hStride),
                               dstPtr,
                               make_uint3(dstDescPtr->strides.nStride, dstDescPtr->strides.cStride, dstDescPtr->strides.hStride),
                               make_uint2(dstDescPtr->w - 8, dstDescPtr->h),
                               handle.GetInitHandle()->mem.mgpu.uintArr[0].uintmem,
                               handle.GetInitHandle()->mem.mgpu.uintArr[1].uintmem,
                               roiTensorPtrSrc);
        }
        else if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            globalThreads_x = (srcDescPtr->strides.hStride + 7) >> 3;
            hipLaunchKernelGGL(flip_pln3_pkd3_tensor,
                               dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
                               dim3(localThreads_x, localThreads_y, localThreads_z),
                               0,
                               handle.GetStream(),
                               srcPtr,
                               make_uint3(srcDescPtr->strides.nStride, srcDescPtr->strides.cStride, srcDescPtr->strides.hStride),
                               dstPtr,
                               make_uint2(dstDescPtr->strides.nStride, dstDescPtr->strides.hStride),
                               make_uint2(dstDescPtr->w - 8, dstDescPtr->h),
                               handle.GetInitHandle()->mem.mgpu.uintArr[0].uintmem,
                               handle.GetInitHandle()->mem.mgpu.uintArr[1].uintmem,
                               roiTensorPtrSrc);
        }
    }

    return RPP_SUCCESS;
}