#include <hip/hip_runtime.h>
#include "rpp_hip_common.hpp"

__device__ void rmn_hip_compute(uchar *srcPtr, d_float8 *pix_f8, d_float8 *rmnParams_f8)
{
    pix_f8->f4[0] = rpp_hip_pixel_check_0to255((pix_f8->f4[0] - rmnParams_f8->f4[0]) * rmnParams_f8->f4[1]);
    pix_f8->f4[1] = rpp_hip_pixel_check_0to255((pix_f8->f4[1] - rmnParams_f8->f4[0]) * rmnParams_f8->f4[1]);
}

__device__ void rmn_hip_compute(float *srcPtr, d_float8 *pix_f8, d_float8 *rmnParams_f8)
{
    pix_f8->f4[0] = rpp_hip_pixel_check_0to1((pix_f8->f4[0] - rmnParams_f8->f4[0] * (float4) ONE_OVER_255) * rmnParams_f8->f4[1]);
    pix_f8->f4[1] = rpp_hip_pixel_check_0to1((pix_f8->f4[1] - rmnParams_f8->f4[0] * (float4) ONE_OVER_255) * rmnParams_f8->f4[1]);
}

__device__ void rmn_hip_compute(schar *srcPtr, d_float8 *pix_f8, d_float8 *rmnParams_f8)
{
    pix_f8->f4[0] = rpp_hip_pixel_check_0to255(((pix_f8->f4[0] + (float4)128) - rmnParams_f8->f4[0]) * rmnParams_f8->f4[1]) - (float4)128;
    pix_f8->f4[1] = rpp_hip_pixel_check_0to255(((pix_f8->f4[1] + (float4)128) - rmnParams_f8->f4[0]) * rmnParams_f8->f4[1]) - (float4)128;
}

__device__ void rmn_hip_compute(half *srcPtr, d_float8 *pix_f8, d_float8 *rmnParams_f8)
{
    pix_f8->f4[0] = rpp_hip_pixel_check_0to1((pix_f8->f4[0] - rmnParams_f8->f4[0] * (float4) ONE_OVER_255) * rmnParams_f8->f4[1]);
    pix_f8->f4[1] = rpp_hip_pixel_check_0to1((pix_f8->f4[1] - rmnParams_f8->f4[0] * (float4) ONE_OVER_255) * rmnParams_f8->f4[1]);
}

__device__ void resize_mirror_normalize_roi_and_srclocs_hip_compute(int4 *srcRoiPtr_i4, uint2 *dstDimsWH, int id_x, int id_y, d_float16 *locSrc_f16)
{
    float wRatio = (float)(srcRoiPtr_i4->z - srcRoiPtr_i4->x + 1) / dstDimsWH->x;
    float hRatio = (float)(srcRoiPtr_i4->w - srcRoiPtr_i4->y + 1) / dstDimsWH->y;
    float4 wOffset_f4 = (float4)((wRatio - 1) * 0.5f);
    float4 hOffset_f4 = (float4)((hRatio - 1) * 0.5f);

    d_float8 increment_f8, locDst_f8x, locDst_f8y;
    increment_f8.f4[0] = make_float4(0.0f, 1.0f, 2.0f, 3.0f);
    increment_f8.f4[1] = make_float4(4.0f, 5.0f, 6.0f, 7.0f);
    locDst_f8x.f4[0] = (float4)id_x + increment_f8.f4[0];
    locDst_f8x.f4[1] = (float4)id_x + increment_f8.f4[1];
    locDst_f8y.f4[0] = (float4)id_y;
    locDst_f8y.f4[1] = (float4)id_y;

    locSrc_f16->f8[0].f4[0] = (locDst_f8x.f4[0] * (float4)wRatio) + wOffset_f4 + (float4)srcRoiPtr_i4->x;  // Compute First 4 locSrcX
    locSrc_f16->f8[0].f4[1] = (locDst_f8x.f4[1] * (float4)wRatio) + wOffset_f4 + (float4)srcRoiPtr_i4->x;  // Compute Next 4 locSrcX
    locSrc_f16->f8[1].f4[0] = (locDst_f8y.f4[0] * (float4)hRatio) + hOffset_f4 + (float4)srcRoiPtr_i4->y;  // Compute First 4 locSrcY
    locSrc_f16->f8[1].f4[1] = (locDst_f8y.f4[1] * (float4)hRatio) + hOffset_f4 + (float4)srcRoiPtr_i4->y;  // Compute Next 4 locSrcY
}

__device__ void resize_mirror_normalize_roi_and_srclocs_hip_compute_mirror(int4 *srcRoiPtr_i4, uint2 *dstDimsWH, int id_x, int id_y, d_float16 *locSrc_f16)
{
    float wRatio = (float)(srcRoiPtr_i4->z - srcRoiPtr_i4->x + 1) / dstDimsWH->x;
    float hRatio = (float)(srcRoiPtr_i4->w - srcRoiPtr_i4->y + 1) / dstDimsWH->y;
    float4 wOffset_f4 = (float4)((wRatio - 1) * 0.5f);
    float4 hOffset_f4 = (float4)((hRatio - 1) * 0.5f);

    d_float8 decrement_f8, locDst_f8x, locDst_f8y;
    decrement_f8.f4[0] = make_float4(dstDimsWH->x - 1, dstDimsWH->x - 2, dstDimsWH->x - 3, dstDimsWH->x - 4);
    decrement_f8.f4[1] = make_float4(dstDimsWH->x - 5, dstDimsWH->x - 6, dstDimsWH->x - 7, dstDimsWH->x - 8);
    locDst_f8x.f4[0] = decrement_f8.f4[0] - (float4)id_x; 
    locDst_f8x.f4[1] = decrement_f8.f4[1] - (float4)id_x;
    locDst_f8y.f4[0] = (float4)id_y;
    locDst_f8y.f4[1] = (float4)id_y;

    locSrc_f16->f8[0].f4[0] = (locDst_f8x.f4[0] * (float4)wRatio) + wOffset_f4 + (float4)srcRoiPtr_i4->x;  // Compute First 4 locSrcX
    locSrc_f16->f8[0].f4[1] = (locDst_f8x.f4[1] * (float4)wRatio) + wOffset_f4 + (float4)srcRoiPtr_i4->x;  // Compute Next 4 locSrcX
    locSrc_f16->f8[1].f4[0] = (locDst_f8y.f4[0] * (float4)hRatio) + hOffset_f4 + (float4)srcRoiPtr_i4->y;  // Compute First 4 locSrcY
    locSrc_f16->f8[1].f4[1] = (locDst_f8y.f4[1] * (float4)hRatio) + hOffset_f4 + (float4)srcRoiPtr_i4->y;  // Compute Next 4 locSrcY
}

template <typename T>
__global__ void resize_mirror_normalize_bilinear_pkd_tensor(T *srcPtr,
                                                            uint2 srcStridesNH,
                                                            T *dstPtr,
                                                            uint2 dstStridesNH,
                                                            RpptImagePatchPtr dstImgSize,
                                                            float *meanTensor,
                                                            float *stdDevTensor,
                                                            uint *mirrorTensor,
                                                            RpptROIPtr roiTensorPtrSrc)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    uint2 dstDimsWH;
    dstDimsWH.x = dstImgSize[id_z].width;
    dstDimsWH.y = dstImgSize[id_z].height;

    if ((id_y >= dstDimsWH.y) || (id_x >= dstDimsWH.x))
    {
        return;
    }

    uint srcIdx = (id_z * srcStridesNH.x);
    uint dstIdx = (id_z * dstStridesNH.x) + (id_y * dstStridesNH.y) + id_x * 3;
    int4 srcRoi_i4 = *(int4 *)&roiTensorPtrSrc[id_z];
    d_float8 rmnParams_R_f8, rmnParams_G_f8, rmnParams_B_f8;
    //Get Params for R channel
    rmnParams_R_f8.f4[0] = (float4)meanTensor[id_z * 3];
    rmnParams_R_f8.f4[1] = (float4)(1 / stdDevTensor[id_z * 3]);

    //Get Params for G channel
    rmnParams_G_f8.f4[0] = (float4)meanTensor[id_z * 3 + 1];
    rmnParams_G_f8.f4[1] = (float4)(1 / stdDevTensor[id_z * 3 + 1]);

    //Get Params for B channel
    rmnParams_B_f8.f4[0] = (float4)meanTensor[id_z * 3 + 2];
    rmnParams_B_f8.f4[1] = (float4)(1 / stdDevTensor[id_z * 3 + 2]);

    d_float16 locSrc_f16;
    if(mirrorTensor[id_z] == 1)
        resize_mirror_normalize_roi_and_srclocs_hip_compute_mirror(&srcRoi_i4, &dstDimsWH, id_x, id_y, &locSrc_f16);
    else
        resize_mirror_normalize_roi_and_srclocs_hip_compute(&srcRoi_i4, &dstDimsWH, id_x, id_y, &locSrc_f16);

    d_float24 dst_f24;
    rpp_hip_interpolate24_bilinear_pkd3(srcPtr + srcIdx, srcStridesNH.y, &locSrc_f16, &srcRoi_i4, &dst_f24, false);
    
    d_float24 dst_f24_pln;
    rpp_hip_pack_float24_pkd3_to_pln3(&dst_f24, &dst_f24_pln);
    rmn_hip_compute(dstPtr, &dst_f24_pln.f8[0], &rmnParams_R_f8);
    rmn_hip_compute(dstPtr, &dst_f24_pln.f8[1], &rmnParams_G_f8);
    rmn_hip_compute(dstPtr, &dst_f24_pln.f8[2], &rmnParams_B_f8);
    rpp_hip_pack_float24_pln3_and_store24_pkd3(dstPtr + dstIdx, &dst_f24_pln);
}

template <typename T>   
__global__ void resize_mirror_normalize_bilinear_pln_tensor(T *srcPtr,
                                                            uint3 srcStridesNCH,
                                                            T *dstPtr,
                                                            uint3 dstStridesNCH,
                                                            RpptImagePatchPtr dstImgSize,
                                                            int channelsDst,
                                                            float *meanTensor,
                                                            float *stdDevTensor,
                                                            uint *mirrorTensor,
                                                            RpptROIPtr roiTensorPtrSrc)
{
    
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    uint2 dstDimsWH;
    dstDimsWH.x = dstImgSize[id_z].width;
    dstDimsWH.y = dstImgSize[id_z].height;

    if ((id_y >= dstDimsWH.y) || (id_x >= dstDimsWH.x))
    {
        return;
    }
    
    uint srcIdx = (id_z * srcStridesNCH.x);
    uint dstIdx = (id_z * dstStridesNCH.x) + (id_y * dstStridesNCH.z) + id_x;
    int4 srcRoi_i4 = *(int4 *)&roiTensorPtrSrc[id_z];
    d_float8 rmnParams_R_f8, rmnParams_G_f8, rmnParams_B_f8;
    //Get Params for R channel
    rmnParams_R_f8.f4[0] = (float4)meanTensor[id_z * 3];
    rmnParams_R_f8.f4[1] = (float4)(1 / stdDevTensor[id_z * 3]);

    //Get Params for G channel
    rmnParams_G_f8.f4[0] = (float4)meanTensor[id_z * 3 + 1];
    rmnParams_G_f8.f4[1] = (float4)(1 / stdDevTensor[id_z * 3 + 1]);

    //Get Params for B channel
    rmnParams_B_f8.f4[0] = (float4)meanTensor[id_z * 3 + 2];
    rmnParams_B_f8.f4[1] = (float4)(1 / stdDevTensor[id_z * 3 + 2]);
    
    d_float16 locSrc_f16;
    if(mirrorTensor[id_z] == 1)
        resize_mirror_normalize_roi_and_srclocs_hip_compute_mirror(&srcRoi_i4, &dstDimsWH, id_x, id_y, &locSrc_f16);
    else
        resize_mirror_normalize_roi_and_srclocs_hip_compute(&srcRoi_i4, &dstDimsWH, id_x, id_y, &locSrc_f16);
        
    d_float8 dst_f8;
    rpp_hip_interpolate8_bilinear_pln1(srcPtr + srcIdx, srcStridesNCH.z, &locSrc_f16, &srcRoi_i4, &dst_f8, false);
    if(!(meanTensor[id_z] == 0.0 && stdDevTensor[id_z] == 1.0))
        rmn_hip_compute(dstPtr, &dst_f8, &rmnParams_R_f8);
    rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &dst_f8);

    if (channelsDst == 3)
    {
        srcIdx += srcStridesNCH.y;
        dstIdx += dstStridesNCH.y;

        rpp_hip_interpolate8_bilinear_pln1(srcPtr + srcIdx, srcStridesNCH.z, &locSrc_f16, &srcRoi_i4, &dst_f8, false);
        if(!(meanTensor[id_z] == 0.0 && stdDevTensor[id_z] == 1.0))
            rmn_hip_compute(dstPtr, &dst_f8, &rmnParams_G_f8);
        rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &dst_f8);

        srcIdx += srcStridesNCH.y;
        dstIdx += dstStridesNCH.y;

        rpp_hip_interpolate8_bilinear_pln1(srcPtr + srcIdx, srcStridesNCH.z, &locSrc_f16, &srcRoi_i4, &dst_f8, false);
        if(!(meanTensor[id_z] == 0.0 && stdDevTensor[id_z] == 1.0))
            rmn_hip_compute(dstPtr, &dst_f8, &rmnParams_B_f8);
        rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &dst_f8);
    }
}

template <typename T>
__global__ void resize_mirror_normalize_bilinear_pkd3_pln3_tensor(T *srcPtr,
                                                                  uint2 srcStridesNH,
                                                                  T *dstPtr,
                                                                  uint3 dstStridesNCH,
                                                                  RpptImagePatchPtr dstImgSize,
                                                                  float *meanTensor,
                                                                  float *stdDevTensor,
                                                                  uint *mirrorTensor,
                                                                  RpptROIPtr roiTensorPtrSrc)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    uint2 dstDimsWH;
    dstDimsWH.x = dstImgSize[id_z].width;
    dstDimsWH.y = dstImgSize[id_z].height;

    if ((id_y >= dstDimsWH.y) || (id_x >= dstDimsWH.x))
    {
        return;
    }

    uint srcIdx = (id_z * srcStridesNH.x);
    uint dstIdx = (id_z * dstStridesNCH.x) + (id_y * dstStridesNCH.z) + id_x;

    int4 srcRoi_i4 = *(int4 *)&roiTensorPtrSrc[id_z];
    d_float8 rmnParams_R_f8, rmnParams_G_f8, rmnParams_B_f8;
    //Get Params for R channel
    rmnParams_R_f8.f4[0] = (float4)meanTensor[id_z * 3];
    rmnParams_R_f8.f4[1] = (float4)(1 / stdDevTensor[id_z * 3]);

    //Get Params for G channel
    rmnParams_G_f8.f4[0] = (float4)meanTensor[id_z * 3 + 1];
    rmnParams_G_f8.f4[1] = (float4)(1 / stdDevTensor[id_z * 3 + 1]);

    //Get Params for B channel
    rmnParams_B_f8.f4[0] = (float4)meanTensor[id_z * 3 + 2];
    rmnParams_B_f8.f4[1] = (float4)(1 / stdDevTensor[id_z * 3 + 2]);

    d_float16 locSrc_f16;
    if(mirrorTensor[id_z] == 1)
        resize_mirror_normalize_roi_and_srclocs_hip_compute_mirror(&srcRoi_i4, &dstDimsWH, id_x, id_y, &locSrc_f16);
    else
        resize_mirror_normalize_roi_and_srclocs_hip_compute(&srcRoi_i4, &dstDimsWH, id_x, id_y, &locSrc_f16);

    d_float24 dst_f24;
    rpp_hip_interpolate24_bilinear_pkd3(srcPtr + srcIdx, srcStridesNH.y, &locSrc_f16, &srcRoi_i4, &dst_f24, false);
    
    d_float24 dst_f24_pln;
    rpp_hip_pack_float24_pkd3_to_pln3(&dst_f24, &dst_f24_pln);
    rmn_hip_compute(dstPtr, &dst_f24_pln.f8[0], &rmnParams_R_f8);
    rmn_hip_compute(dstPtr, &dst_f24_pln.f8[1], &rmnParams_G_f8);
    rmn_hip_compute(dstPtr, &dst_f24_pln.f8[2], &rmnParams_B_f8);
    rpp_hip_pack_float24_pln3_and_store24_pln3(dstPtr + dstIdx, dstStridesNCH.y, &dst_f24_pln);
}

template <typename T>
__global__ void resize_mirror_normalize_bilinear_pln3_pkd3_tensor(T *srcPtr,
                                                                  uint3 srcStridesNCH,
                                                                  T *dstPtr,
                                                                  uint2 dstStridesNH,
                                                                  RpptImagePatchPtr dstImgSize,
                                                                  float *meanTensor,
                                                                  float *stdDevTensor,
                                                                  uint *mirrorTensor,
                                                                  RpptROIPtr roiTensorPtrSrc)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    uint2 dstDimsWH;
    dstDimsWH.x = dstImgSize[id_z].width;
    dstDimsWH.y = dstImgSize[id_z].height;

    if ((id_y >= dstDimsWH.y) || (id_x >= dstDimsWH.x))
    {
        return;
    }

    uint srcIdx = (id_z * srcStridesNCH.x);
    uint dstIdx = (id_z * dstStridesNH.x) + (id_y * dstStridesNH.y) + id_x * 3;

    d_float8 rmnParams_R_f8, rmnParams_G_f8, rmnParams_B_f8;
    //Get Params for R channel
    rmnParams_R_f8.f4[0] = (float4)meanTensor[id_z * 3];
    rmnParams_R_f8.f4[1] = (float4)(1 / stdDevTensor[id_z * 3]);

    //Get Params for G channel
    rmnParams_G_f8.f4[0] = (float4)meanTensor[id_z * 3 + 1];
    rmnParams_G_f8.f4[1] = (float4)(1 / stdDevTensor[id_z * 3 + 1]);

    //Get Params for B channel
    rmnParams_B_f8.f4[0] = (float4)meanTensor[id_z * 3 + 2];
    rmnParams_B_f8.f4[1] = (float4)(1 / stdDevTensor[id_z * 3 + 2]);
    
    int4 srcRoi_i4 = *(int4 *)&roiTensorPtrSrc[id_z];
    d_float16 locSrc_f16;
    if(mirrorTensor[id_z] == 1)
        resize_mirror_normalize_roi_and_srclocs_hip_compute_mirror(&srcRoi_i4, &dstDimsWH, id_x, id_y, &locSrc_f16);
    else
        resize_mirror_normalize_roi_and_srclocs_hip_compute(&srcRoi_i4, &dstDimsWH, id_x, id_y, &locSrc_f16);
        
    d_float24 dst_f24;
    rpp_hip_interpolate24_bilinear_pln3(srcPtr + srcIdx, &srcStridesNCH, &locSrc_f16, &srcRoi_i4, &dst_f24, false);
    rmn_hip_compute(dstPtr, &dst_f24.f8[0], &rmnParams_R_f8);
    rmn_hip_compute(dstPtr, &dst_f24.f8[1], &rmnParams_G_f8);
    rmn_hip_compute(dstPtr, &dst_f24.f8[2], &rmnParams_B_f8);
    rpp_hip_pack_float24_pln3_and_store24_pkd3(dstPtr + dstIdx, &dst_f24);
}

// -------------------- Set 3 - Kernel Executors --------------------

template <typename T>
RppStatus hip_exec_resize_mirror_normalize_tensor(T *srcPtr,
                                                 RpptDescPtr srcDescPtr,
                                                 T *dstPtr,
                                                 RpptDescPtr dstDescPtr,
                                                 RpptImagePatchPtr dstImgSizes,
                                                 RpptInterpolationType interpolationType,
                                                 RpptROIPtr roiTensorPtrSrc,
                                                 RpptRoiType roiType,
                                                 rpp::Handle& handle)
{
    if (roiType == RpptRoiType::XYWH)
        hip_exec_roi_converison_xywh_to_ltrb(roiTensorPtrSrc, handle);

    //Set non ROI pixels to zero
    int max_dst_size = dstDescPtr->w * dstDescPtr->h * dstDescPtr->c;
    for(int i = 0; i < dstDescPtr->n; i++)
    {
        hipMemset(dstPtr + i * (max_dst_size), (T)0, size_t(max_dst_size));
    }

    int localThreads_x = 16;
    int localThreads_y = 16;
    int localThreads_z = 1;
    int globalThreads_x = (dstDescPtr->strides.hStride + 7) >> 3;
    int globalThreads_y = dstDescPtr->h;
    int globalThreads_z = handle.GetBatchSize();

    if (interpolationType == RpptInterpolationType::BILINEAR)
    {
        if ((srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            hipLaunchKernelGGL(resize_mirror_normalize_bilinear_pkd_tensor,
                               dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
                               dim3(localThreads_x, localThreads_y, localThreads_z),
                               0,
                               handle.GetStream(),
                               srcPtr,
                               make_uint2(srcDescPtr->strides.nStride, srcDescPtr->strides.hStride),
                               dstPtr,
                               make_uint2(dstDescPtr->strides.nStride, dstDescPtr->strides.hStride),
                               dstImgSizes,
                               handle.GetInitHandle()->mem.mgpu.floatArr[0].floatmem,
                               handle.GetInitHandle()->mem.mgpu.floatArr[1].floatmem,
                               handle.GetInitHandle()->mem.mgpu.uintArr[2].uintmem,
                               roiTensorPtrSrc);
        }
        else if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            hipLaunchKernelGGL(resize_mirror_normalize_bilinear_pln_tensor,
                               dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
                               dim3(localThreads_x, localThreads_y, localThreads_z),
                               0,
                               handle.GetStream(),
                               srcPtr,
                               make_uint3(srcDescPtr->strides.nStride, srcDescPtr->strides.cStride, srcDescPtr->strides.hStride),
                               dstPtr,
                               make_uint3(dstDescPtr->strides.nStride, dstDescPtr->strides.cStride, dstDescPtr->strides.hStride),
                               dstImgSizes,
                               dstDescPtr->c,
                               handle.GetInitHandle()->mem.mgpu.floatArr[0].floatmem,
                               handle.GetInitHandle()->mem.mgpu.floatArr[1].floatmem,
                               handle.GetInitHandle()->mem.mgpu.uintArr[2].uintmem,
                               roiTensorPtrSrc);
        }
        else if ((srcDescPtr->c == 3) && (dstDescPtr->c == 3))
        {
            if ((srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
            {
                hipLaunchKernelGGL(resize_mirror_normalize_bilinear_pkd3_pln3_tensor,
                                   dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
                                   dim3(localThreads_x, localThreads_y, localThreads_z),
                                   0,
                                   handle.GetStream(),
                                   srcPtr,
                                   make_uint2(srcDescPtr->strides.nStride, srcDescPtr->strides.hStride),
                                   dstPtr,
                                   make_uint3(dstDescPtr->strides.nStride, dstDescPtr->strides.cStride, dstDescPtr->strides.hStride),
                                   dstImgSizes,
                                   handle.GetInitHandle()->mem.mgpu.floatArr[0].floatmem,
                                   handle.GetInitHandle()->mem.mgpu.floatArr[1].floatmem,
                                   handle.GetInitHandle()->mem.mgpu.uintArr[2].uintmem,
                                   roiTensorPtrSrc);
            }
            else if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
            {
                globalThreads_x = (dstDescPtr->w + 7) >> 3;
                hipLaunchKernelGGL(resize_mirror_normalize_bilinear_pln3_pkd3_tensor,
                                   dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
                                   dim3(localThreads_x, localThreads_y, localThreads_z),
                                   0,
                                   handle.GetStream(),
                                   srcPtr,
                                   make_uint3(srcDescPtr->strides.nStride, srcDescPtr->strides.cStride, srcDescPtr->strides.hStride),
                                   dstPtr,
                                   make_uint2(dstDescPtr->strides.nStride, dstDescPtr->strides.hStride),
                                   dstImgSizes,
                                   handle.GetInitHandle()->mem.mgpu.floatArr[0].floatmem,
                                   handle.GetInitHandle()->mem.mgpu.floatArr[1].floatmem,
                                   handle.GetInitHandle()->mem.mgpu.uintArr[2].uintmem,
                                   roiTensorPtrSrc);
            }
        }
    }

    return RPP_SUCCESS;
}
