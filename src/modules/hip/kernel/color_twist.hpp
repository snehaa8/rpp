#include <hip/hip_runtime.h>
#include "hip/rpp_hip_common.hpp"
#include "cpu/rpp_cpu_common.hpp"

__device__ void color_twist_1RGB_hip_compute(float *pixelR, float *pixelG, float *pixelB, float4 *colorTwistParams_f4)
{
    // RGB to HSV

    float hue, sat, v, add;
    float rf, gf, bf, cmax, cmin, delta;
    rf = *pixelR;
    gf = *pixelG;
    bf = *pixelB;
    cmax = fmaxf(fmaxf(rf, gf), bf);
    cmin = fminf(fminf(rf, gf), bf);
    delta = cmax - cmin;
    hue = 0.0f;
    sat = 0.0f;
    add = 0.0f;
    if ((delta != 0) && (cmax != 0))
    {
        sat = delta / cmax;
        if (cmax == rf)
        {
            hue = gf - bf;
            add = 0.0f;
        }
        else if (cmax == gf)
        {
            hue = bf - rf;
            add = 2.0f;
        }
        else
        {
            hue = rf - gf;
            add = 4.0f;
        }
        hue /= delta;
    }
    v = cmax;

    // Modify Hue and Saturation

    hue += colorTwistParams_f4->z + add;
    if (hue >= 6.0f) hue -= 6.0f;
    if (hue < 0) hue += 6.0f;
    sat *= colorTwistParams_f4->w;
    sat = fmaxf(0.0f, fminf(1.0f, sat));

    // HSV to RGB with brightness/contrast adjustment

    int hueIntegerPart = (int) hue;
    float hueFractionPart = hue - hueIntegerPart;
    float vsat = v * sat;
    float vsatf = vsat * hueFractionPart;
    float p = v - vsat;
    float q = v - vsatf;
    float t = v - vsat + vsatf;
    switch (hueIntegerPart)
    {
        case 0: rf = v; gf = t; bf = p; break;
        case 1: rf = q; gf = v; bf = p; break;
        case 2: rf = p; gf = v; bf = t; break;
        case 3: rf = p; gf = q; bf = v; break;
        case 4: rf = t; gf = p; bf = v; break;
        case 5: rf = v; gf = p; bf = q; break;
    }
    *pixelR = fmaf(rf, colorTwistParams_f4->x, colorTwistParams_f4->y);
    *pixelG = fmaf(gf, colorTwistParams_f4->x, colorTwistParams_f4->y);
    *pixelB = fmaf(bf, colorTwistParams_f4->x, colorTwistParams_f4->y);
}

__device__ void color_twist_8RGB_hip_compute(d_float24 *pix_f24, float4 *colorTwistParams_f4)
{
    color_twist_1RGB_hip_compute(&(pix_f24->x.x.x), &(pix_f24->y.x.x), &(pix_f24->z.x.x), colorTwistParams_f4);
    color_twist_1RGB_hip_compute(&(pix_f24->x.x.y), &(pix_f24->y.x.y), &(pix_f24->z.x.y), colorTwistParams_f4);
    color_twist_1RGB_hip_compute(&(pix_f24->x.x.z), &(pix_f24->y.x.z), &(pix_f24->z.x.z), colorTwistParams_f4);
    color_twist_1RGB_hip_compute(&(pix_f24->x.x.w), &(pix_f24->y.x.w), &(pix_f24->z.x.w), colorTwistParams_f4);
    color_twist_1RGB_hip_compute(&(pix_f24->x.y.x), &(pix_f24->y.y.x), &(pix_f24->z.y.x), colorTwistParams_f4);
    color_twist_1RGB_hip_compute(&(pix_f24->x.y.y), &(pix_f24->y.y.y), &(pix_f24->z.y.y), colorTwistParams_f4);
    color_twist_1RGB_hip_compute(&(pix_f24->x.y.z), &(pix_f24->y.y.z), &(pix_f24->z.y.z), colorTwistParams_f4);
    color_twist_1RGB_hip_compute(&(pix_f24->x.y.w), &(pix_f24->y.y.w), &(pix_f24->z.y.w), colorTwistParams_f4);
}

__device__ void color_twist_hip_compute(uchar *srcPtr, d_float24 *pix_f24, float4 *colorTwistParams_f4)
{
    float4 normalizer_f4 = (float4) ONE_OVER_255;
    rpp_hip_math_multiply24_const(pix_f24, pix_f24, normalizer_f4);
    colorTwistParams_f4->x = colorTwistParams_f4->x * 255.0f;
    colorTwistParams_f4->z = (((int)colorTwistParams_f4->z) % 360) * SIX_OVER_360;
    color_twist_8RGB_hip_compute(pix_f24, colorTwistParams_f4);
    rpp_hip_pixel_check_0to255(pix_f24);
}
__device__ void color_twist_hip_compute(float *srcPtr, d_float24 *pix_f24, float4 *colorTwistParams_f4)
{
    colorTwistParams_f4->y = colorTwistParams_f4->y * ONE_OVER_255;
    colorTwistParams_f4->z = (((int)colorTwistParams_f4->z) % 360) * SIX_OVER_360;
    color_twist_8RGB_hip_compute(pix_f24, colorTwistParams_f4);
    rpp_hip_pixel_check_0to1(pix_f24);
}
__device__ void color_twist_hip_compute(half *srcPtr, d_float24 *pix_f24, float4 *colorTwistParams_f4)
{
    colorTwistParams_f4->y = colorTwistParams_f4->y * ONE_OVER_255;
    colorTwistParams_f4->z = (((int)colorTwistParams_f4->z) % 360) * SIX_OVER_360;
    color_twist_8RGB_hip_compute(pix_f24, colorTwistParams_f4);
    rpp_hip_pixel_check_0to1(pix_f24);
}
__device__ void color_twist_hip_compute(schar *srcPtr, d_float24 *pix_f24, float4 *colorTwistParams_f4)
{
    float4 i8Offset_f4 = (float4) 128.0f;
    float4 normalizer_f4 = (float4) ONE_OVER_255;
    rpp_hip_math_add24_const(pix_f24, pix_f24, i8Offset_f4);
    rpp_hip_math_multiply24_const(pix_f24, pix_f24, normalizer_f4);
    colorTwistParams_f4->x = colorTwistParams_f4->x * 255.0f;
    colorTwistParams_f4->z = (((int)colorTwistParams_f4->z) % 360) * SIX_OVER_360;
    color_twist_8RGB_hip_compute(pix_f24, colorTwistParams_f4);
    rpp_hip_pixel_check_0to255(pix_f24);
    rpp_hip_math_subtract24_const(pix_f24, pix_f24, i8Offset_f4);
}

template <typename T>
__global__ void color_twist_pkd_tensor(T *srcPtr,
                                       uint2 srcStridesNH,
                                       T *dstPtr,
                                       uint2 dstStridesNH,
                                       float *brightnessTensor,
                                       float *contrastTensor,
                                       float *hueTensor,
                                       float *saturationTensor,
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

    float4 colorTwistParams_f4 = make_float4(brightnessTensor[id_z], contrastTensor[id_z], hueTensor[id_z], saturationTensor[id_z]);

    d_float24 pix_f24;

    rpp_hip_load24_pkd3_and_unpack_to_float24_pln3(srcPtr, srcIdx, &pix_f24);
    color_twist_hip_compute(srcPtr, &pix_f24, &colorTwistParams_f4);
    rpp_hip_pack_float24_pln3_and_store24_pkd3(dstPtr, dstIdx, &pix_f24);
}

// template <typename T>
// __global__ void color_twist_pln_tensor(T *srcPtr,
//                                       uint3 srcStridesNCH,
//                                       T *dstPtr,
//                                       uint3 dstStridesNCH,
//                                       RpptRGB *rgbTensor,
//                                       float *alphaTensor,
//                                       RpptROIPtr roiTensorPtrSrc)
// {
//     int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
//     int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
//     int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

//     if ((id_y >= roiTensorPtrSrc[id_z].xywhROI.roiHeight) || (id_x >= roiTensorPtrSrc[id_z].xywhROI.roiWidth))
//     {
//         return;
//     }

//     uint srcIdx = (id_z * srcStridesNCH.x) + ((id_y + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNCH.z) + (id_x + roiTensorPtrSrc[id_z].xywhROI.xy.x);
//     uint dstIdx = (id_z * dstStridesNCH.x) + (id_y * dstStridesNCH.z) + id_x;

//     float4 r_f4 = (float4)((float)rgbTensor[id_z].R);
//     float4 g_f4 = (float4)((float)rgbTensor[id_z].G);
//     float4 b_f4 = (float4)((float)rgbTensor[id_z].B);
//     float4 alpha_f4 = (float4)(alphaTensor[id_z]);

//     d_float24 src_f24, dst_f24;

//     rpp_hip_load24_pln3_and_unpack_to_float24_pln3(srcPtr, srcIdx, srcStridesNCH.y, &src_f24);
//     color_twist_hip_compute(srcPtr, &src_f24.x, &dst_f24.x, &b_f4, &alpha_f4);
//     color_twist_hip_compute(srcPtr, &src_f24.y, &dst_f24.y, &g_f4, &alpha_f4);
//     color_twist_hip_compute(srcPtr, &src_f24.z, &dst_f24.z, &r_f4, &alpha_f4);
//     rpp_hip_pack_float24_pln3_and_store24_pln3(dstPtr, dstIdx, dstStridesNCH.y, &dst_f24);
// }

// template <typename T>
// __global__ void color_twist_pkd3_pln3_tensor(T *srcPtr,
//                                             uint2 srcStridesNH,
//                                             T *dstPtr,
//                                             uint3 dstStridesNCH,
//                                             RpptRGB *rgbTensor,
//                                             float *alphaTensor,
//                                             RpptROIPtr roiTensorPtrSrc)
// {
//     int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
//     int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
//     int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

//     if ((id_y >= roiTensorPtrSrc[id_z].xywhROI.roiHeight) || (id_x >= roiTensorPtrSrc[id_z].xywhROI.roiWidth))
//     {
//         return;
//     }

//     uint srcIdx = (id_z * srcStridesNH.x) + ((id_y + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNH.y) + ((id_x + roiTensorPtrSrc[id_z].xywhROI.xy.x) * 3);
//     uint dstIdx = (id_z * dstStridesNCH.x) + (id_y * dstStridesNCH.z) + id_x;

//     float4 r_f4 = (float4)((float)rgbTensor[id_z].R);
//     float4 g_f4 = (float4)((float)rgbTensor[id_z].G);
//     float4 b_f4 = (float4)((float)rgbTensor[id_z].B);
//     float4 alpha_f4 = (float4)(alphaTensor[id_z]);

//     d_float24 src_f24, dst_f24;

//     rpp_hip_load24_pkd3_and_unpack_to_float24_pln3(srcPtr, srcIdx, &src_f24);
//     color_twist_hip_compute(srcPtr, &src_f24.x, &dst_f24.x, &b_f4, &alpha_f4);
//     color_twist_hip_compute(srcPtr, &src_f24.y, &dst_f24.y, &g_f4, &alpha_f4);
//     color_twist_hip_compute(srcPtr, &src_f24.z, &dst_f24.z, &r_f4, &alpha_f4);
//     rpp_hip_pack_float24_pln3_and_store24_pln3(dstPtr, dstIdx, dstStridesNCH.y, &dst_f24);
// }

// template <typename T>
// __global__ void color_twist_pln3_pkd3_tensor(T *srcPtr,
//                                             uint3 srcStridesNCH,
//                                             T *dstPtr,
//                                             uint2 dstStridesNH,
//                                             RpptRGB *rgbTensor,
//                                             float *alphaTensor,
//                                             RpptROIPtr roiTensorPtrSrc)
// {
//     int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
//     int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
//     int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

//     if ((id_y >= roiTensorPtrSrc[id_z].xywhROI.roiHeight) || (id_x >= roiTensorPtrSrc[id_z].xywhROI.roiWidth))
//     {
//         return;
//     }

//     uint srcIdx = (id_z * srcStridesNCH.x) + ((id_y + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNCH.z) + (id_x + roiTensorPtrSrc[id_z].xywhROI.xy.x);
//     uint dstIdx = (id_z * dstStridesNH.x) + (id_y * dstStridesNH.y) + id_x * 3;

//     float4 r_f4 = (float4)((float)rgbTensor[id_z].R);
//     float4 g_f4 = (float4)((float)rgbTensor[id_z].G);
//     float4 b_f4 = (float4)((float)rgbTensor[id_z].B);
//     float4 alpha_f4 = (float4)(alphaTensor[id_z]);

//     d_float24 src_f24, dst_f24;

//     rpp_hip_load24_pln3_and_unpack_to_float24_pln3(srcPtr, srcIdx, srcStridesNCH.y, &src_f24);
//     color_twist_hip_compute(srcPtr, &src_f24.x, &dst_f24.x, &b_f4, &alpha_f4);
//     color_twist_hip_compute(srcPtr, &src_f24.y, &dst_f24.y, &g_f4, &alpha_f4);
//     color_twist_hip_compute(srcPtr, &src_f24.z, &dst_f24.z, &r_f4, &alpha_f4);
//     rpp_hip_pack_float24_pln3_and_store24_pkd3(dstPtr, dstIdx, &dst_f24);
// }

template <typename T>
RppStatus hip_exec_color_twist_tensor(T *srcPtr,
                                     RpptDescPtr srcDescPtr,
                                     T *dstPtr,
                                     RpptDescPtr dstDescPtr,
                                     RpptROIPtr roiTensorPtrSrc,
                                     rpp::Handle& handle)
{
    if ((srcDescPtr->c == 3) && (dstDescPtr->c == 3))
    {
        int localThreads_x = 16;
        int localThreads_y = 16;
        int localThreads_z = 1;
        int globalThreads_x = (dstDescPtr->strides.hStride + 7) >> 3;
        int globalThreads_y = dstDescPtr->h;
        int globalThreads_z = handle.GetBatchSize();

        if ((srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            globalThreads_x = (dstDescPtr->strides.hStride / 3 + 7) >> 3;
            hipLaunchKernelGGL(color_twist_pkd_tensor,
                               dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
                               dim3(localThreads_x, localThreads_y, localThreads_z),
                               0,
                               handle.GetStream(),
                               srcPtr,
                               make_uint2(srcDescPtr->strides.nStride, srcDescPtr->strides.hStride),
                               dstPtr,
                               make_uint2(dstDescPtr->strides.nStride, dstDescPtr->strides.hStride),
                               handle.GetInitHandle()->mem.mgpu.floatArr[0].floatmem,
                               handle.GetInitHandle()->mem.mgpu.floatArr[1].floatmem,
                               handle.GetInitHandle()->mem.mgpu.floatArr[2].floatmem,
                               handle.GetInitHandle()->mem.mgpu.floatArr[3].floatmem,
                               roiTensorPtrSrc);
        }
        // else if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
        // {
        //     hipLaunchKernelGGL(color_twist_pln_tensor,
        //                        dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
        //                        dim3(localThreads_x, localThreads_y, localThreads_z),
        //                        0,
        //                        handle.GetStream(),
        //                        srcPtr,
        //                        make_uint3(srcDescPtr->strides.nStride, srcDescPtr->strides.cStride, srcDescPtr->strides.hStride),
        //                        dstPtr,
        //                        make_uint3(dstDescPtr->strides.nStride, dstDescPtr->strides.cStride, dstDescPtr->strides.hStride),
        //                        handle.GetInitHandle()->mem.mgpu.floatArr[0].floatmem,
        //                        handle.GetInitHandle()->mem.mgpu.floatArr[1].floatmem,
        //                        handle.GetInitHandle()->mem.mgpu.floatArr[2].floatmem,
        //                        handle.GetInitHandle()->mem.mgpu.floatArr[3].floatmem,
        //                        roiTensorPtrSrc);
        // }
        // else if ((srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
        // {
        //     hipLaunchKernelGGL(color_twist_pkd3_pln3_tensor,
        //                        dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
        //                        dim3(localThreads_x, localThreads_y, localThreads_z),
        //                        0,
        //                        handle.GetStream(),
        //                        srcPtr,
        //                        make_uint2(srcDescPtr->strides.nStride, srcDescPtr->strides.hStride),
        //                        dstPtr,
        //                        make_uint3(dstDescPtr->strides.nStride, dstDescPtr->strides.cStride, dstDescPtr->strides.hStride),
        //                        handle.GetInitHandle()->mem.mgpu.floatArr[0].floatmem,
        //                        handle.GetInitHandle()->mem.mgpu.floatArr[1].floatmem,
        //                        handle.GetInitHandle()->mem.mgpu.floatArr[2].floatmem,
        //                        handle.GetInitHandle()->mem.mgpu.floatArr[3].floatmem,
        //                        roiTensorPtrSrc);
        // }
        // else if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
        // {
        //     globalThreads_x = (srcDescPtr->strides.hStride + 7) >> 3;
        //     hipLaunchKernelGGL(color_twist_pln3_pkd3_tensor,
        //                        dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
        //                        dim3(localThreads_x, localThreads_y, localThreads_z),
        //                        0,
        //                        handle.GetStream(),
        //                        srcPtr,
        //                        make_uint3(srcDescPtr->strides.nStride, srcDescPtr->strides.cStride, srcDescPtr->strides.hStride),
        //                        dstPtr,
        //                        make_uint2(dstDescPtr->strides.nStride, dstDescPtr->strides.hStride),
        //                        handle.GetInitHandle()->mem.mgpu.floatArr[0].floatmem,
        //                        handle.GetInitHandle()->mem.mgpu.floatArr[1].floatmem,
        //                        handle.GetInitHandle()->mem.mgpu.floatArr[2].floatmem,
        //                        handle.GetInitHandle()->mem.mgpu.floatArr[3].floatmem,
        //                        roiTensorPtrSrc);
        // }
    }

    return RPP_SUCCESS;
}
