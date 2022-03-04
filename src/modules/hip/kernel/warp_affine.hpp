#include <hip/hip_runtime.h>
#include "hip/rpp_hip_common.hpp"

// -------------------- Set 0 - warp_affine device helpers --------------------

__device__ void warp_affine_srclocs_hip_compute(float affineMatrixElement, float4 locSrcComponent_f4, d_float8 *locSrcPtr_f8)
{
    d_float8 increment_f8;
    increment_f8.x.x = 0;
    increment_f8.x.y = affineMatrixElement;
    increment_f8.x.z = affineMatrixElement + increment_f8.x.y;
    increment_f8.x.w = affineMatrixElement + increment_f8.x.z;
    increment_f8.y = (float4)(affineMatrixElement + increment_f8.x.w) + increment_f8.x;
    locSrcPtr_f8->x = locSrcComponent_f4 + increment_f8.x;
    locSrcPtr_f8->y = locSrcComponent_f4 + increment_f8.y;
}

__device__ void warp_affine_roi_and_srclocs_hip_compute(d_int4 *roiSrc, int id_x, int id_y, d_float6_as_float3s *affineMatrix_f6, d_float16 *locSrc_f16)
{
    float2 locDst_f2, locSrc_f2;
    int roiHalfWidth = (roiSrc->y.x - roiSrc->x.x + 1) >> 1;
    int roiHalfHeight = (roiSrc->y.y - roiSrc->x.y + 1) >> 1;
    roiSrc->y.x -= 1;
    roiSrc->y.y -= 1;
    locDst_f2.x = (float) (id_x - roiHalfWidth);
    locDst_f2.y = (float) (id_y - roiHalfHeight);
    locSrc_f2.x = fmaf(locDst_f2.x, affineMatrix_f6->x.x, fmaf(locDst_f2.y, affineMatrix_f6->x.y, affineMatrix_f6->x.z)) + roiHalfWidth;
    locSrc_f2.y = fmaf(locDst_f2.x, affineMatrix_f6->y.x, fmaf(locDst_f2.y, affineMatrix_f6->y.y, affineMatrix_f6->y.z)) + roiHalfHeight;
    warp_affine_srclocs_hip_compute(affineMatrix_f6->x.x, (float4)locSrc_f2.x, &(locSrc_f16->x));    // Compute 8 locSrcX
    warp_affine_srclocs_hip_compute(affineMatrix_f6->y.x, (float4)locSrc_f2.y, &(locSrc_f16->y));    // Compute 8 locSrcY
}

// -------------------- Set 1 - Bilinear Interpolation --------------------

template <typename T>
__global__ void warp_affine_bilinear_pkd_tensor(T *srcPtr,
                                                uint2 srcStridesNH,
                                                T *dstPtr,
                                                uint2 dstStridesNH,
                                                uint2 dstDimsWH,
                                                d_float6_as_float3s *affineTensorPtr,
                                                RpptROIPtr roiTensorPtrSrc)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if ((id_y >= dstDimsWH.y) || (id_x >= dstDimsWH.x))
    {
        return;
    }

    uint srcIdx = (id_z * srcStridesNH.x);
    uint dstIdx = (id_z * dstStridesNH.x) + (id_y * dstStridesNH.y) + id_x * 3;

    d_float6_as_float3s affineMatrix_f6 = affineTensorPtr[id_z];
    d_int4 roiSrc = *(d_int4 *)&roiTensorPtrSrc[id_z];
    d_float16 locSrc_f16;
    warp_affine_roi_and_srclocs_hip_compute(&roiSrc, id_x, id_y, &affineMatrix_f6, &locSrc_f16);

    d_float24_as_float3s dst_f24;
    rpp_hip_interpolate24_bilinear_pkd3(&srcPtr[srcIdx], srcStridesNH.y, &locSrc_f16, &roiSrc, &dst_f24);
    rpp_hip_pack_float24_pkd3_and_store24_pkd3(dstPtr, dstIdx, (d_float24 *)&dst_f24);
}

template <typename T>
__global__ void warp_affine_bilinear_pln_tensor(T *srcPtr,
                                                uint3 srcStridesNCH,
                                                T *dstPtr,
                                                uint3 dstStridesNCH,
                                                uint2 dstDimsWH,
                                                int channelsDst,
                                                d_float6_as_float3s *affineTensorPtr,
                                                RpptROIPtr roiTensorPtrSrc)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if ((id_y >= dstDimsWH.y) || (id_x >= dstDimsWH.x))
    {
        return;
    }

    uint srcIdx = (id_z * srcStridesNCH.x);
    uint dstIdx = (id_z * dstStridesNCH.x) + (id_y * dstStridesNCH.z) + id_x;

    d_float6_as_float3s affineMatrix_f6 = affineTensorPtr[id_z];
    d_int4 roiSrc = *(d_int4 *)&roiTensorPtrSrc[id_z];
    d_float16 locSrc_f16;
    warp_affine_roi_and_srclocs_hip_compute(&roiSrc, id_x, id_y, &affineMatrix_f6, &locSrc_f16);

    d_float8 dst_f8;
    rpp_hip_interpolate8_bilinear_pln1(&srcPtr[srcIdx], srcStridesNCH.z, &locSrc_f16, &roiSrc, &dst_f8);
    rpp_hip_pack_float8_and_store8(dstPtr, dstIdx, &dst_f8);

    if (channelsDst == 3)
    {
        srcIdx += srcStridesNCH.y;
        dstIdx += dstStridesNCH.y;

        rpp_hip_interpolate8_bilinear_pln1(&srcPtr[srcIdx], srcStridesNCH.z, &locSrc_f16, &roiSrc, &dst_f8);
        rpp_hip_pack_float8_and_store8(dstPtr, dstIdx, &dst_f8);

        srcIdx += srcStridesNCH.y;
        dstIdx += dstStridesNCH.y;

        rpp_hip_interpolate8_bilinear_pln1(&srcPtr[srcIdx], srcStridesNCH.z, &locSrc_f16, &roiSrc, &dst_f8);
        rpp_hip_pack_float8_and_store8(dstPtr, dstIdx, &dst_f8);
    }
}

template <typename T>
__global__ void warp_affine_bilinear_pkd3_pln3_tensor(T *srcPtr,
                                                      uint2 srcStridesNH,
                                                      T *dstPtr,
                                                      uint3 dstStridesNCH,
                                                      uint2 dstDimsWH,
                                                      d_float6_as_float3s *affineTensorPtr,
                                                      RpptROIPtr roiTensorPtrSrc)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if ((id_y >= dstDimsWH.y) || (id_x >= dstDimsWH.x))
    {
        return;
    }

    uint srcIdx = (id_z * srcStridesNH.x);
    uint dstIdx = (id_z * dstStridesNCH.x) + (id_y * dstStridesNCH.z) + id_x;

    d_float6_as_float3s affineMatrix_f6 = affineTensorPtr[id_z];
    d_int4 roiSrc = *(d_int4 *)&roiTensorPtrSrc[id_z];
    d_float16 locSrc_f16;
    warp_affine_roi_and_srclocs_hip_compute(&roiSrc, id_x, id_y, &affineMatrix_f6, &locSrc_f16);

    d_float24_as_float3s dst_f24;
    rpp_hip_interpolate24_bilinear_pkd3(&srcPtr[srcIdx], srcStridesNH.y, &locSrc_f16, &roiSrc, &dst_f24);
    rpp_hip_pack_float24_pkd3_and_store24_pln3(dstPtr, dstIdx, dstStridesNCH.y, (d_float24 *)&dst_f24);
}

template <typename T>
__global__ void warp_affine_bilinear_pln3_pkd3_tensor(T *srcPtr,
                                                      uint3 srcStridesNCH,
                                                      T *dstPtr,
                                                      uint2 dstStridesNH,
                                                      uint2 dstDimsWH,
                                                      d_float6_as_float3s *affineTensorPtr,
                                                      RpptROIPtr roiTensorPtrSrc)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if ((id_y >= dstDimsWH.y) || (id_x >= dstDimsWH.x))
    {
        return;
    }

    uint srcIdx = (id_z * srcStridesNCH.x);
    uint dstIdx = (id_z * dstStridesNH.x) + (id_y * dstStridesNH.y) + id_x * 3;

    d_float6_as_float3s affineMatrix_f6 = affineTensorPtr[id_z];
    d_int4 roiSrc = *(d_int4 *)&roiTensorPtrSrc[id_z];
    d_float16 locSrc_f16;
    warp_affine_roi_and_srclocs_hip_compute(&roiSrc, id_x, id_y, &affineMatrix_f6, &locSrc_f16);

    d_float24 dst_f24;
    rpp_hip_interpolate24_bilinear_pln3(&srcPtr[srcIdx], &srcStridesNCH, &locSrc_f16, &roiSrc, &dst_f24);
    rpp_hip_pack_float24_pln3_and_store24_pkd3(dstPtr, dstIdx, &dst_f24);
}

// -------------------- Set 2 - Nearest Neighbor Interpolation --------------------

template <typename T>
__global__ void warp_affine_nearest_neighbor_pkd_tensor(T *srcPtr,
                                                uint2 srcStridesNH,
                                                T *dstPtr,
                                                uint2 dstStridesNH,
                                                uint2 dstDimsWH,
                                                d_float6_as_float3s *affineTensorPtr,
                                                RpptROIPtr roiTensorPtrSrc)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if ((id_y >= dstDimsWH.y) || (id_x >= dstDimsWH.x))
    {
        return;
    }

    uint srcIdx = (id_z * srcStridesNH.x);
    uint dstIdx = (id_z * dstStridesNH.x) + (id_y * dstStridesNH.y) + id_x * 3;

    d_float6_as_float3s affineMatrix_f6 = affineTensorPtr[id_z];
    d_int4 roiSrc = *(d_int4 *)&roiTensorPtrSrc[id_z];
    d_float16 locSrc_f16;
    warp_affine_roi_and_srclocs_hip_compute(&roiSrc, id_x, id_y, &affineMatrix_f6, &locSrc_f16);

    d_float24_as_float3s dst_f24;
    rpp_hip_interpolate24_nearest_neighbor_pkd3(&srcPtr[srcIdx], srcStridesNH.y, &locSrc_f16, &roiSrc, &dst_f24);
    rpp_hip_pack_float24_pkd3_and_store24_pkd3(dstPtr, dstIdx, (d_float24 *)&dst_f24);
}

template <typename T>
__global__ void warp_affine_nearest_neighbor_pln_tensor(T *srcPtr,
                                                        uint3 srcStridesNCH,
                                                        T *dstPtr,
                                                        uint3 dstStridesNCH,
                                                        uint2 dstDimsWH,
                                                        int channelsDst,
                                                        d_float6_as_float3s *affineTensorPtr,
                                                        RpptROIPtr roiTensorPtrSrc)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if ((id_y >= dstDimsWH.y) || (id_x >= dstDimsWH.x))
    {
        return;
    }

    uint srcIdx = (id_z * srcStridesNCH.x);
    uint dstIdx = (id_z * dstStridesNCH.x) + (id_y * dstStridesNCH.z) + id_x;

    d_float6_as_float3s affineMatrix_f6 = affineTensorPtr[id_z];
    d_int4 roiSrc = *(d_int4 *)&roiTensorPtrSrc[id_z];
    d_float16 locSrc_f16;
    warp_affine_roi_and_srclocs_hip_compute(&roiSrc, id_x, id_y, &affineMatrix_f6, &locSrc_f16);

    d_float8 dst_f8;
    rpp_hip_interpolate8_nearest_neighbor_pln1(&srcPtr[srcIdx], srcStridesNCH.z, &locSrc_f16, &roiSrc, &dst_f8);
    rpp_hip_pack_float8_and_store8(dstPtr, dstIdx, &dst_f8);

    if (channelsDst == 3)
    {
        srcIdx += srcStridesNCH.y;
        dstIdx += dstStridesNCH.y;

        rpp_hip_interpolate8_nearest_neighbor_pln1(&srcPtr[srcIdx], srcStridesNCH.z, &locSrc_f16, &roiSrc, &dst_f8);
        rpp_hip_pack_float8_and_store8(dstPtr, dstIdx, &dst_f8);

        srcIdx += srcStridesNCH.y;
        dstIdx += dstStridesNCH.y;

        rpp_hip_interpolate8_nearest_neighbor_pln1(&srcPtr[srcIdx], srcStridesNCH.z, &locSrc_f16, &roiSrc, &dst_f8);
        rpp_hip_pack_float8_and_store8(dstPtr, dstIdx, &dst_f8);
    }
}

template <typename T>
__global__ void warp_affine_nearest_neighbor_pkd3_pln3_tensor(T *srcPtr,
                                                              uint2 srcStridesNH,
                                                              T *dstPtr,
                                                              uint3 dstStridesNCH,
                                                              uint2 dstDimsWH,
                                                              d_float6_as_float3s *affineTensorPtr,
                                                              RpptROIPtr roiTensorPtrSrc)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if ((id_y >= dstDimsWH.y) || (id_x >= dstDimsWH.x))
    {
        return;
    }

    uint srcIdx = (id_z * srcStridesNH.x);
    uint dstIdx = (id_z * dstStridesNCH.x) + (id_y * dstStridesNCH.z) + id_x;

    d_float6_as_float3s affineMatrix_f6 = affineTensorPtr[id_z];
    d_int4 roiSrc = *(d_int4 *)&roiTensorPtrSrc[id_z];
    d_float16 locSrc_f16;
    warp_affine_roi_and_srclocs_hip_compute(&roiSrc, id_x, id_y, &affineMatrix_f6, &locSrc_f16);

    d_float24_as_float3s dst_f24;
    rpp_hip_interpolate24_nearest_neighbor_pkd3(&srcPtr[srcIdx], srcStridesNH.y, &locSrc_f16, &roiSrc, &dst_f24);
    rpp_hip_pack_float24_pkd3_and_store24_pln3(dstPtr, dstIdx, dstStridesNCH.y, (d_float24 *)&dst_f24);
}

template <typename T>
__global__ void warp_affine_nearest_neighbor_pln3_pkd3_tensor(T *srcPtr,
                                                              uint3 srcStridesNCH,
                                                              T *dstPtr,
                                                              uint2 dstStridesNH,
                                                              uint2 dstDimsWH,
                                                              d_float6_as_float3s *affineTensorPtr,
                                                              RpptROIPtr roiTensorPtrSrc)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if ((id_y >= dstDimsWH.y) || (id_x >= dstDimsWH.x))
    {
        return;
    }

    uint srcIdx = (id_z * srcStridesNCH.x);
    uint dstIdx = (id_z * dstStridesNH.x) + (id_y * dstStridesNH.y) + id_x * 3;

    d_float6_as_float3s affineMatrix_f6 = affineTensorPtr[id_z];
    d_int4 roiSrc = *(d_int4 *)&roiTensorPtrSrc[id_z];
    d_float16 locSrc_f16;
    warp_affine_roi_and_srclocs_hip_compute(&roiSrc, id_x, id_y, &affineMatrix_f6, &locSrc_f16);

    d_float24 dst_f24;
    rpp_hip_interpolate24_nearest_neighbor_pln3(&srcPtr[srcIdx], &srcStridesNCH, &locSrc_f16, &roiSrc, &dst_f24);
    rpp_hip_pack_float24_pln3_and_store24_pkd3(dstPtr, dstIdx, &dst_f24);
}

// -------------------- Set 3 - Kernel Executors --------------------

template <typename T>
RppStatus hip_exec_warp_affine_tensor(T *srcPtr,
                                      RpptDescPtr srcDescPtr,
                                      T *dstPtr,
                                      RpptDescPtr dstDescPtr,
                                      Rpp32f *affineTensor,
                                      RpptInterpolationType interpolationType,
                                      RpptROIPtr roiTensorPtrSrc,
                                      RpptRoiType roiType,
                                      rpp::Handle& handle)
{
    if (roiType == RpptRoiType::XYWH)
        hip_exec_roi_converison_xywh_to_ltrb(roiTensorPtrSrc, handle);

    int localThreads_x = 16;
    int localThreads_y = 16;
    int localThreads_z = 1;
    int globalThreads_x = (dstDescPtr->strides.hStride + 7) >> 3;
    int globalThreads_y = dstDescPtr->h;
    int globalThreads_z = handle.GetBatchSize();

    float *affineTensorPtr = handle.GetInitHandle()->mem.mgpu.maskArr.floatmem;
    hipMemcpy(affineTensorPtr, affineTensor, 6 * handle.GetBatchSize() * sizeof(float), hipMemcpyHostToDevice);

    if (interpolationType == RpptInterpolationType::BILINEAR)
    {
        if ((srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            hipLaunchKernelGGL(warp_affine_bilinear_pkd_tensor,
                            dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
                            dim3(localThreads_x, localThreads_y, localThreads_z),
                            0,
                            handle.GetStream(),
                            srcPtr,
                            make_uint2(srcDescPtr->strides.nStride, srcDescPtr->strides.hStride),
                            dstPtr,
                            make_uint2(dstDescPtr->strides.nStride, dstDescPtr->strides.hStride),
                            make_uint2(dstDescPtr->w, dstDescPtr->h),
                            (d_float6_as_float3s *)affineTensorPtr,
                            roiTensorPtrSrc);
        }
        else if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            hipLaunchKernelGGL(warp_affine_bilinear_pln_tensor,
                            dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
                            dim3(localThreads_x, localThreads_y, localThreads_z),
                            0,
                            handle.GetStream(),
                            srcPtr,
                            make_uint3(srcDescPtr->strides.nStride, srcDescPtr->strides.cStride, srcDescPtr->strides.hStride),
                            dstPtr,
                            make_uint3(dstDescPtr->strides.nStride, dstDescPtr->strides.cStride, dstDescPtr->strides.hStride),
                            make_uint2(dstDescPtr->w, dstDescPtr->h),
                            dstDescPtr->c,
                            (d_float6_as_float3s *)affineTensorPtr,
                            roiTensorPtrSrc);
        }
        else if ((srcDescPtr->c == 3) && (dstDescPtr->c == 3))
        {
            if ((srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
            {
                hipLaunchKernelGGL(warp_affine_bilinear_pkd3_pln3_tensor,
                                dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
                                dim3(localThreads_x, localThreads_y, localThreads_z),
                                0,
                                handle.GetStream(),
                                srcPtr,
                                make_uint2(srcDescPtr->strides.nStride, srcDescPtr->strides.hStride),
                                dstPtr,
                                make_uint3(dstDescPtr->strides.nStride, dstDescPtr->strides.cStride, dstDescPtr->strides.hStride),
                                make_uint2(dstDescPtr->w, dstDescPtr->h),
                                (d_float6_as_float3s *)affineTensorPtr,
                                roiTensorPtrSrc);
            }
            else if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
            {
                globalThreads_x = (srcDescPtr->strides.hStride + 7) >> 3;
                hipLaunchKernelGGL(warp_affine_bilinear_pln3_pkd3_tensor,
                                dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
                                dim3(localThreads_x, localThreads_y, localThreads_z),
                                0,
                                handle.GetStream(),
                                srcPtr,
                                make_uint3(srcDescPtr->strides.nStride, srcDescPtr->strides.cStride, srcDescPtr->strides.hStride),
                                dstPtr,
                                make_uint2(dstDescPtr->strides.nStride, dstDescPtr->strides.hStride),
                                make_uint2(dstDescPtr->w, dstDescPtr->h),
                                (d_float6_as_float3s *)affineTensorPtr,
                                roiTensorPtrSrc);
            }
        }
    }
    else if (interpolationType == RpptInterpolationType::NEAREST_NEIGHBOR)
    {
        if ((srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            hipLaunchKernelGGL(warp_affine_nearest_neighbor_pkd_tensor,
                            dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
                            dim3(localThreads_x, localThreads_y, localThreads_z),
                            0,
                            handle.GetStream(),
                            srcPtr,
                            make_uint2(srcDescPtr->strides.nStride, srcDescPtr->strides.hStride),
                            dstPtr,
                            make_uint2(dstDescPtr->strides.nStride, dstDescPtr->strides.hStride),
                            make_uint2(dstDescPtr->w, dstDescPtr->h),
                            (d_float6_as_float3s *)affineTensorPtr,
                            roiTensorPtrSrc);
        }
        else if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            hipLaunchKernelGGL(warp_affine_nearest_neighbor_pln_tensor,
                            dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
                            dim3(localThreads_x, localThreads_y, localThreads_z),
                            0,
                            handle.GetStream(),
                            srcPtr,
                            make_uint3(srcDescPtr->strides.nStride, srcDescPtr->strides.cStride, srcDescPtr->strides.hStride),
                            dstPtr,
                            make_uint3(dstDescPtr->strides.nStride, dstDescPtr->strides.cStride, dstDescPtr->strides.hStride),
                            make_uint2(dstDescPtr->w, dstDescPtr->h),
                            dstDescPtr->c,
                            (d_float6_as_float3s *)affineTensorPtr,
                            roiTensorPtrSrc);
        }
        else if ((srcDescPtr->c == 3) && (dstDescPtr->c == 3))
        {
            if ((srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
            {
                hipLaunchKernelGGL(warp_affine_nearest_neighbor_pkd3_pln3_tensor,
                                dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
                                dim3(localThreads_x, localThreads_y, localThreads_z),
                                0,
                                handle.GetStream(),
                                srcPtr,
                                make_uint2(srcDescPtr->strides.nStride, srcDescPtr->strides.hStride),
                                dstPtr,
                                make_uint3(dstDescPtr->strides.nStride, dstDescPtr->strides.cStride, dstDescPtr->strides.hStride),
                                make_uint2(dstDescPtr->w, dstDescPtr->h),
                                (d_float6_as_float3s *)affineTensorPtr,
                                roiTensorPtrSrc);
            }
            else if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
            {
                globalThreads_x = (srcDescPtr->strides.hStride + 7) >> 3;
                hipLaunchKernelGGL(warp_affine_nearest_neighbor_pln3_pkd3_tensor,
                                dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
                                dim3(localThreads_x, localThreads_y, localThreads_z),
                                0,
                                handle.GetStream(),
                                srcPtr,
                                make_uint3(srcDescPtr->strides.nStride, srcDescPtr->strides.cStride, srcDescPtr->strides.hStride),
                                dstPtr,
                                make_uint2(dstDescPtr->strides.nStride, dstDescPtr->strides.hStride),
                                make_uint2(dstDescPtr->w, dstDescPtr->h),
                                (d_float6_as_float3s *)affineTensorPtr,
                                roiTensorPtrSrc);
            }
        }
    }

    return RPP_SUCCESS;
}
