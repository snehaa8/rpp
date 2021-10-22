/*
Copyright (c) 2019 - 2021 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#ifndef HOST_TENSOR_AUGMENTATIONS_HPP
#define HOST_TENSOR_AUGMENTATIONS_HPP

#include "cpu/rpp_cpu_simd.hpp"
#include <cpu/rpp_cpu_common.hpp>
#include <stdlib.h>
#include <time.h>
#include <algorithm>

/************ brightness ************/

RppStatus brightness_u8_u8_host_tensor(Rpp8u *srcPtr,
                                       RpptDescPtr srcDescPtr,
                                       Rpp8u *dstPtr,
                                       RpptDescPtr dstDescPtr,
                                       Rpp32f *alphaTensor,
                                       Rpp32f *betaTensor,
                                       RpptROIPtr roiTensorPtrSrc,
                                       RpptRoiType roiType,
                                       RppLayoutParams layoutParams)
{
    RpptROI roiDefault;
    RpptROIPtr roiPtrDefault;
    roiPtrDefault = &roiDefault;
    roiPtrDefault->xywhROI.xy.x = 0;
    roiPtrDefault->xywhROI.xy.y = 0;
    roiPtrDefault->xywhROI.roiWidth = srcDescPtr->w;
    roiPtrDefault->xywhROI.roiHeight = srcDescPtr->h;

    omp_set_dynamic(0);
#pragma omp parallel for num_threads(dstDescPtr->n)
    for(int batchCount = 0; batchCount < dstDescPtr->n; batchCount++)
    {
        RpptROI roi;
        RpptROIPtr roiPtr;

        if (&roiTensorPtrSrc[batchCount] == NULL)
        {
            roiPtr = roiPtrDefault;
        }
        else
        {
            RpptROIPtr roiPtrInput = &roiTensorPtrSrc[batchCount];

            RpptROI roiImage;
            RpptROIPtr roiPtrImage;

            if (roiType == RpptRoiType::LTRB)
            {
                roiPtrImage = &roiImage;
                compute_xywh_from_ltrb_host(roiPtrInput, roiPtrImage);
            }
            else if (roiType == RpptRoiType::XYWH)
            {
                roiPtrImage = roiPtrInput;
            }

            roiPtr = &roi;
            compute_roi_boundary_check_host(roiPtrImage, roiPtr, roiPtrDefault);
        }

        Rpp32f alpha = alphaTensor[batchCount];
        Rpp32f beta = betaTensor[batchCount];

        Rpp8u *srcPtrImage, *dstPtrImage;
        srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;

        Rpp32u bufferLength = roiPtr->xywhROI.roiWidth * layoutParams.bufferMultiplier;

        __m128 pMul = _mm_set1_ps(alpha);
        __m128 pAdd = _mm_set1_ps(beta);

        Rpp8u *srcPtrChannel, *dstPtrChannel;
        srcPtrChannel = srcPtrImage + (roiPtr->xywhROI.xy.y * srcDescPtr->strides.hStride) + (roiPtr->xywhROI.xy.x * layoutParams.bufferMultiplier);
        dstPtrChannel = dstPtrImage;

        // Brightness with fused output-layout toggle (NHWC -> NCHW)
        if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp32u alignedLength = bufferLength & ~47;

            Rpp8u *srcPtrRow, *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
            srcPtrRow = srcPtrChannel;
            dstPtrRowR = dstPtrChannel;
            dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
            dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;

            for(int i = 0; i < roiPtr->xywhROI.roiHeight; i++)
            {
                Rpp8u *srcPtrTemp, *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
                srcPtrTemp = srcPtrRow;
                dstPtrTempR = dstPtrRowR;
                dstPtrTempG = dstPtrRowG;
                dstPtrTempB = dstPtrRowB;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount+=48)
                {
                    __m128 p[12];

                    rpp_simd_load(rpp_load48_u8pkd3_to_f32pln3, srcPtrTemp, p);    // simd loads
                    p[0] = _mm_fmadd_ps(p[0], pMul, pAdd);    // brightness adjustment Rs
                    p[1] = _mm_fmadd_ps(p[1], pMul, pAdd);    // brightness adjustment Rs
                    p[2] = _mm_fmadd_ps(p[2], pMul, pAdd);    // brightness adjustment Rs
                    p[3] = _mm_fmadd_ps(p[3], pMul, pAdd);    // brightness adjustment Rs
                    p[4] = _mm_fmadd_ps(p[4], pMul, pAdd);    // brightness adjustment Gs
                    p[5] = _mm_fmadd_ps(p[5], pMul, pAdd);    // brightness adjustment Gs
                    p[6] = _mm_fmadd_ps(p[6], pMul, pAdd);    // brightness adjustment Gs
                    p[7] = _mm_fmadd_ps(p[7], pMul, pAdd);    // brightness adjustment Gs
                    p[8] = _mm_fmadd_ps(p[8], pMul, pAdd);    // brightness adjustment Bs
                    p[9] = _mm_fmadd_ps(p[9], pMul, pAdd);    // brightness adjustment Bs
                    p[10] = _mm_fmadd_ps(p[10], pMul, pAdd);    // brightness adjustment Bs
                    p[11] = _mm_fmadd_ps(p[11], pMul, pAdd);    // brightness adjustment Bs
                    rpp_simd_store(rpp_store48_f32pln3_to_u8pln3, dstPtrTempR, dstPtrTempG, dstPtrTempB, p);    // simd stores

                    srcPtrTemp += 48;
                    dstPtrTempR += 16;
                    dstPtrTempG += 16;
                    dstPtrTempB += 16;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount+=3)
                {
                    *dstPtrTempR = (Rpp8u) RPPPIXELCHECK((((Rpp32f) (srcPtrTemp[0])) * alpha) + beta);
                    *dstPtrTempG = (Rpp8u) RPPPIXELCHECK((((Rpp32f) (srcPtrTemp[1])) * alpha) + beta);
                    *dstPtrTempB = (Rpp8u) RPPPIXELCHECK((((Rpp32f) (srcPtrTemp[2])) * alpha) + beta);

                    srcPtrTemp += 3;
                    dstPtrTempR++;
                    dstPtrTempG++;
                    dstPtrTempB++;
                }

                srcPtrRow += srcDescPtr->strides.hStride;
                dstPtrRowR += dstDescPtr->strides.hStride;
                dstPtrRowG += dstDescPtr->strides.hStride;
                dstPtrRowB += dstDescPtr->strides.hStride;
            }
        }

        // Brightness with fused output-layout toggle (NCHW -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp32u alignedLength = bufferLength & ~47;

            Rpp8u *srcPtrRowR, *srcPtrRowG, *srcPtrRowB, *dstPtrRow;
            srcPtrRowR = srcPtrChannel;
            srcPtrRowG = srcPtrRowR + srcDescPtr->strides.cStride;
            srcPtrRowB = srcPtrRowG + srcDescPtr->strides.cStride;
            dstPtrRow = dstPtrChannel;

            for(int i = 0; i < roiPtr->xywhROI.roiHeight; i++)
            {
                Rpp8u *srcPtrTempR, *srcPtrTempG, *srcPtrTempB, *dstPtrTemp;
                srcPtrTempR = srcPtrRowR;
                srcPtrTempG = srcPtrRowG;
                srcPtrTempB = srcPtrRowB;
                dstPtrTemp = dstPtrRow;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount+=16)
                {
                    __m128 p[12];

                    rpp_simd_load(rpp_load48_u8pln3_to_f32pln3, srcPtrTempR, srcPtrTempG, srcPtrTempB, p);    // simd loads
                    p[0] = _mm_fmadd_ps(p[0], pMul, pAdd);    // brightness adjustment Rs
                    p[1] = _mm_fmadd_ps(p[1], pMul, pAdd);    // brightness adjustment Rs
                    p[2] = _mm_fmadd_ps(p[2], pMul, pAdd);    // brightness adjustment Rs
                    p[3] = _mm_fmadd_ps(p[3], pMul, pAdd);    // brightness adjustment Rs
                    p[4] = _mm_fmadd_ps(p[4], pMul, pAdd);    // brightness adjustment Gs
                    p[5] = _mm_fmadd_ps(p[5], pMul, pAdd);    // brightness adjustment Gs
                    p[6] = _mm_fmadd_ps(p[6], pMul, pAdd);    // brightness adjustment Gs
                    p[7] = _mm_fmadd_ps(p[7], pMul, pAdd);    // brightness adjustment Gs
                    p[8] = _mm_fmadd_ps(p[8], pMul, pAdd);    // brightness adjustment Bs
                    p[9] = _mm_fmadd_ps(p[9], pMul, pAdd);    // brightness adjustment Bs
                    p[10] = _mm_fmadd_ps(p[10], pMul, pAdd);    // brightness adjustment Bs
                    p[11] = _mm_fmadd_ps(p[11], pMul, pAdd);    // brightness adjustment Bs
                    rpp_simd_store(rpp_store48_f32pln3_to_u8pkd3, dstPtrTemp, p);    // simd stores

                    srcPtrTempR += 16;
                    srcPtrTempG += 16;
                    srcPtrTempB += 16;
                    dstPtrTemp += 48;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    dstPtrTemp[0] = (Rpp8u) RPPPIXELCHECK((((Rpp32f) (*srcPtrTempR)) * alpha) + beta);
                    dstPtrTemp[1] = (Rpp8u) RPPPIXELCHECK((((Rpp32f) (*srcPtrTempG)) * alpha) + beta);
                    dstPtrTemp[2] = (Rpp8u) RPPPIXELCHECK((((Rpp32f) (*srcPtrTempB)) * alpha) + beta);

                    srcPtrTempR++;
                    srcPtrTempG++;
                    srcPtrTempB++;
                    dstPtrTemp += 3;
                }

                srcPtrRowR += srcDescPtr->strides.hStride;
                srcPtrRowG += srcDescPtr->strides.hStride;
                srcPtrRowB += srcDescPtr->strides.hStride;
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }

        // Brightness without fused output-layout toggle (NHWC -> NHWC or NCHW -> NCHW)
        else
        {
            Rpp32u alignedLength = bufferLength & ~15;

            for(int c = 0; c < layoutParams.channelParam; c++)
            {
                Rpp8u *srcPtrRow, *dstPtrRow;
                srcPtrRow = srcPtrChannel;
                dstPtrRow = dstPtrChannel;

                for(int i = 0; i < roiPtr->xywhROI.roiHeight; i++)
                {
                    Rpp8u *srcPtrTemp, *dstPtrTemp;
                    srcPtrTemp = srcPtrRow;
                    dstPtrTemp = dstPtrRow;

                    int vectorLoopCount = 0;
                    for (; vectorLoopCount < alignedLength; vectorLoopCount+=16)
                    {
                        __m128 p[4];

                        rpp_simd_load(rpp_load16_u8_to_f32, srcPtrTemp, p);    // simd loads
                        p[0] = _mm_fmadd_ps(p[0], pMul, pAdd);    // brightness adjustment
                        p[1] = _mm_fmadd_ps(p[1], pMul, pAdd);    // brightness adjustment
                        p[2] = _mm_fmadd_ps(p[2], pMul, pAdd);    // brightness adjustment
                        p[3] = _mm_fmadd_ps(p[3], pMul, pAdd);    // brightness adjustment
                        rpp_simd_store(rpp_store16_f32_to_u8, dstPtrTemp, p);    // simd stores

                        srcPtrTemp +=16;
                        dstPtrTemp +=16;
                    }
                    for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                    {
                        *dstPtrTemp = (Rpp8u) RPPPIXELCHECK((((Rpp32f) (*srcPtrTemp)) * alpha) + beta);

                        srcPtrTemp++;
                        dstPtrTemp++;
                    }

                    srcPtrRow += srcDescPtr->strides.hStride;
                    dstPtrRow += dstDescPtr->strides.hStride;
                }

                srcPtrChannel += srcDescPtr->strides.cStride;
                dstPtrChannel += dstDescPtr->strides.cStride;
            }
        }
    }

    return RPP_SUCCESS;
}

RppStatus brightness_f32_f32_host_tensor(Rpp32f *srcPtr,
                                         RpptDescPtr srcDescPtr,
                                         Rpp32f *dstPtr,
                                         RpptDescPtr dstDescPtr,
                                         Rpp32f *alphaTensor,
                                         Rpp32f *betaTensor,
                                         RpptROIPtr roiTensorPtrSrc,
                                         RpptRoiType roiType,
                                         RppLayoutParams layoutParams)
{
    RpptROI roiDefault;
    RpptROIPtr roiPtrDefault;
    roiPtrDefault = &roiDefault;
    roiPtrDefault->xywhROI.xy.x = 0;
    roiPtrDefault->xywhROI.xy.y = 0;
    roiPtrDefault->xywhROI.roiWidth = srcDescPtr->w;
    roiPtrDefault->xywhROI.roiHeight = srcDescPtr->h;

    omp_set_dynamic(0);
#pragma omp parallel for num_threads(dstDescPtr->n)
    for(int batchCount = 0; batchCount < dstDescPtr->n; batchCount++)
    {
        RpptROI roi;
        RpptROIPtr roiPtr;

        if (&roiTensorPtrSrc[batchCount] == NULL)
        {
            roiPtr = roiPtrDefault;
        }
        else
        {
            RpptROIPtr roiPtrInput = &roiTensorPtrSrc[batchCount];

            RpptROI roiImage;
            RpptROIPtr roiPtrImage;

            if (roiType == RpptRoiType::LTRB)
            {
                roiPtrImage = &roiImage;
                compute_xywh_from_ltrb_host(roiPtrInput, roiPtrImage);
            }
            else if (roiType == RpptRoiType::XYWH)
            {
                roiPtrImage = roiPtrInput;
            }

            roiPtr = &roi;
            compute_roi_boundary_check_host(roiPtrImage, roiPtr, roiPtrDefault);
        }

        Rpp32f alpha = alphaTensor[batchCount];
        Rpp32f beta = betaTensor[batchCount] * 0.0039216; // 1/255

        Rpp32f *srcPtrImage, *dstPtrImage;
        srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;

        Rpp32u bufferLength = roiPtr->xywhROI.roiWidth * layoutParams.bufferMultiplier;

        __m128 pMul = _mm_set1_ps(alpha);
        __m128 pAdd = _mm_set1_ps(beta);

        Rpp32f *srcPtrChannel, *dstPtrChannel;
        srcPtrChannel = srcPtrImage + (roiPtr->xywhROI.xy.y * srcDescPtr->strides.hStride) + (roiPtr->xywhROI.xy.x * layoutParams.bufferMultiplier);
        dstPtrChannel = dstPtrImage;

        // Brightness with fused output-layout toggle (NHWC -> NCHW)
        if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp32u alignedLength = bufferLength & ~11;

            Rpp32f *srcPtrRow, *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
            srcPtrRow = srcPtrChannel;
            dstPtrRowR = dstPtrChannel;
            dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
            dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;

            for(int i = 0; i < roiPtr->xywhROI.roiHeight; i++)
            {
                Rpp32f *srcPtrTemp, *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
                srcPtrTemp = srcPtrRow;
                dstPtrTempR = dstPtrRowR;
                dstPtrTempG = dstPtrRowG;
                dstPtrTempB = dstPtrRowB;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount+=12)
                {
                    __m128 p[4];

                    rpp_simd_load(rpp_load12_f32pkd3_to_f32pln3, srcPtrTemp, p);    // simd loads
                    p[0] = _mm_fmadd_ps(p[0], pMul, pAdd);    // brightness adjustment Rs
                    p[1] = _mm_fmadd_ps(p[1], pMul, pAdd);    // brightness adjustment Gs
                    p[2] = _mm_fmadd_ps(p[2], pMul, pAdd);    // brightness adjustment Bs
                    rpp_simd_store(rpp_store12_f32pln3_to_f32pln3, dstPtrTempR, dstPtrTempG, dstPtrTempB, p);    // simd stores

                    srcPtrTemp += 12;
                    dstPtrTempR += 4;
                    dstPtrTempG += 4;
                    dstPtrTempB += 4;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount+=3)
                {
                    *dstPtrTempR = RPPPIXELCHECKF32(srcPtrTemp[0] * alpha + beta);
                    *dstPtrTempG = RPPPIXELCHECKF32(srcPtrTemp[1] * alpha + beta);
                    *dstPtrTempB = RPPPIXELCHECKF32(srcPtrTemp[2] * alpha + beta);

                    srcPtrTemp += 3;
                    dstPtrTempR++;
                    dstPtrTempG++;
                    dstPtrTempB++;
                }

                srcPtrRow += srcDescPtr->strides.hStride;
                dstPtrRowR += dstDescPtr->strides.hStride;
                dstPtrRowG += dstDescPtr->strides.hStride;
                dstPtrRowB += dstDescPtr->strides.hStride;
            }
        }

        // Brightness with fused output-layout toggle (NCHW -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp32u alignedLength = bufferLength & ~11;

            Rpp32f *srcPtrRowR, *srcPtrRowG, *srcPtrRowB, *dstPtrRow;
            srcPtrRowR = srcPtrChannel;
            srcPtrRowG = srcPtrRowR + srcDescPtr->strides.cStride;
            srcPtrRowB = srcPtrRowG + srcDescPtr->strides.cStride;
            dstPtrRow = dstPtrChannel;

            for(int i = 0; i < roiPtr->xywhROI.roiHeight; i++)
            {
                Rpp32f *srcPtrTempR, *srcPtrTempG, *srcPtrTempB, *dstPtrTemp;
                srcPtrTempR = srcPtrRowR;
                srcPtrTempG = srcPtrRowG;
                srcPtrTempB = srcPtrRowB;
                dstPtrTemp = dstPtrRow;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount+=4)
                {
                    __m128 p[4];

                    rpp_simd_load(rpp_load12_f32pln3_to_f32pln3, srcPtrTempR, srcPtrTempG, srcPtrTempB, p);    // simd loads
                    p[0] = _mm_fmadd_ps(p[0], pMul, pAdd);    // brightness adjustment Rs
                    p[1] = _mm_fmadd_ps(p[1], pMul, pAdd);    // brightness adjustment Gs
                    p[2] = _mm_fmadd_ps(p[2], pMul, pAdd);    // brightness adjustment Bs
                    rpp_simd_store(rpp_store12_f32pln3_to_f32pkd3, dstPtrTemp, p);    // simd stores

                    srcPtrTempR += 4;
                    srcPtrTempG += 4;
                    srcPtrTempB += 4;
                    dstPtrTemp += 12;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    dstPtrTemp[0] = RPPPIXELCHECKF32(*srcPtrTempR * alpha + beta);
                    dstPtrTemp[1] = RPPPIXELCHECKF32(*srcPtrTempG * alpha + beta);
                    dstPtrTemp[2] = RPPPIXELCHECKF32(*srcPtrTempB * alpha + beta);

                    srcPtrTempR++;
                    srcPtrTempG++;
                    srcPtrTempB++;
                    dstPtrTemp += 3;
                }

                srcPtrRowR += srcDescPtr->strides.hStride;
                srcPtrRowG += srcDescPtr->strides.hStride;
                srcPtrRowB += srcDescPtr->strides.hStride;
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }

        // Brightness without fused output-layout toggle (NHWC -> NHWC or NCHW -> NCHW)
        else
        {
            Rpp32u alignedLength = bufferLength & ~3;

            for(int c = 0; c < layoutParams.channelParam; c++)
            {
                Rpp32f *srcPtrRow, *dstPtrRow;
                srcPtrRow = srcPtrChannel;
                dstPtrRow = dstPtrChannel;

                for(int i = 0; i < roiPtr->xywhROI.roiHeight; i++)
                {
                    Rpp32f *srcPtrTemp, *dstPtrTemp;
                    srcPtrTemp = srcPtrRow;
                    dstPtrTemp = dstPtrRow;

                    int vectorLoopCount = 0;
                    for (; vectorLoopCount < alignedLength; vectorLoopCount+=4)
                    {
                        __m128 p[1];

                        rpp_simd_load(rpp_load4_f32_to_f32, srcPtrTemp, p);    // simd loads
                        p[0] = _mm_fmadd_ps(p[0], pMul, pAdd);    // brightness adjustment
                        rpp_simd_store(rpp_store4_f32_to_f32, dstPtrTemp, p);    // simd stores

                        srcPtrTemp += 4;
                        dstPtrTemp += 4;
                    }
                    for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                    {
                        *dstPtrTemp = RPPPIXELCHECKF32(*srcPtrTemp * alpha + beta);

                        srcPtrTemp++;
                        dstPtrTemp++;
                    }

                    srcPtrRow += srcDescPtr->strides.hStride;
                    dstPtrRow += dstDescPtr->strides.hStride;
                }

                srcPtrChannel += srcDescPtr->strides.cStride;
                dstPtrChannel += dstDescPtr->strides.cStride;
            }
        }
    }

    return RPP_SUCCESS;
}

RppStatus brightness_f16_f16_host_tensor(Rpp16f *srcPtr,
                                         RpptDescPtr srcDescPtr,
                                         Rpp16f *dstPtr,
                                         RpptDescPtr dstDescPtr,
                                         Rpp32f *alphaTensor,
                                         Rpp32f *betaTensor,
                                         RpptROIPtr roiTensorPtrSrc,
                                         RpptRoiType roiType,
                                         RppLayoutParams layoutParams)
{
    RpptROI roiDefault;
    RpptROIPtr roiPtrDefault;
    roiPtrDefault = &roiDefault;
    roiPtrDefault->xywhROI.xy.x = 0;
    roiPtrDefault->xywhROI.xy.y = 0;
    roiPtrDefault->xywhROI.roiWidth = srcDescPtr->w;
    roiPtrDefault->xywhROI.roiHeight = srcDescPtr->h;

    omp_set_dynamic(0);
#pragma omp parallel for num_threads(dstDescPtr->n)
    for(int batchCount = 0; batchCount < dstDescPtr->n; batchCount++)
    {
        RpptROI roi;
        RpptROIPtr roiPtr;

        if (&roiTensorPtrSrc[batchCount] == NULL)
        {
            roiPtr = roiPtrDefault;
        }
        else
        {
            RpptROIPtr roiPtrInput = &roiTensorPtrSrc[batchCount];

            RpptROI roiImage;
            RpptROIPtr roiPtrImage;

            if (roiType == RpptRoiType::LTRB)
            {
                roiPtrImage = &roiImage;
                compute_xywh_from_ltrb_host(roiPtrInput, roiPtrImage);
            }
            else if (roiType == RpptRoiType::XYWH)
            {
                roiPtrImage = roiPtrInput;
            }

            roiPtr = &roi;
            compute_roi_boundary_check_host(roiPtrImage, roiPtr, roiPtrDefault);
        }

        Rpp32f alpha = alphaTensor[batchCount];
        Rpp32f beta = betaTensor[batchCount] * 0.0039216; // 1/255

        Rpp16f *srcPtrImage, *dstPtrImage;
        srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;

        Rpp32u bufferLength = roiPtr->xywhROI.roiWidth * layoutParams.bufferMultiplier;

        __m128 pMul = _mm_set1_ps(alpha);
        __m128 pAdd = _mm_set1_ps(beta);

        Rpp16f *srcPtrChannel, *dstPtrChannel;
        srcPtrChannel = srcPtrImage + (roiPtr->xywhROI.xy.y * srcDescPtr->strides.hStride) + (roiPtr->xywhROI.xy.x * layoutParams.bufferMultiplier);
        dstPtrChannel = dstPtrImage;

        // Brightness with fused output-layout toggle (NHWC -> NCHW)
        if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp32u alignedLength = bufferLength & ~11;

            Rpp16f *srcPtrRow, *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
            srcPtrRow = srcPtrChannel;
            dstPtrRowR = dstPtrChannel;
            dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
            dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;

            for(int i = 0; i < roiPtr->xywhROI.roiHeight; i++)
            {
                Rpp16f *srcPtrTemp, *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
                srcPtrTemp = srcPtrRow;
                dstPtrTempR = dstPtrRowR;
                dstPtrTempG = dstPtrRowG;
                dstPtrTempB = dstPtrRowB;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount+=12)
                {
                    Rpp32f srcPtrTemp_ps[12], dstPtrTemp_ps[12];

                    for(int cnt = 0; cnt < 12; cnt++)
                    {
                        *(srcPtrTemp_ps + cnt) = (Rpp32f) *(srcPtrTemp + cnt);
                    }

                    __m128 p[4];

                    rpp_simd_load(rpp_load12_f32pkd3_to_f32pln3, srcPtrTemp_ps, p);    // simd loads
                    p[0] = _mm_fmadd_ps(p[0], pMul, pAdd);    // brightness adjustment Rs
                    p[1] = _mm_fmadd_ps(p[1], pMul, pAdd);    // brightness adjustment Gs
                    p[2] = _mm_fmadd_ps(p[2], pMul, pAdd);    // brightness adjustment Bs
                    rpp_simd_store(rpp_store12_f32pln3_to_f32pln3, dstPtrTemp_ps, dstPtrTemp_ps + 4, dstPtrTemp_ps + 8, p);    // simd stores

                    for(int cnt = 0; cnt < 4; cnt++)
                    {
                        *(dstPtrTempR + cnt) = (Rpp16f) *(dstPtrTemp_ps + cnt);
                        *(dstPtrTempG + cnt) = (Rpp16f) *(dstPtrTemp_ps + 4 + cnt);
                        *(dstPtrTempB + cnt) = (Rpp16f) *(dstPtrTemp_ps + 8 + cnt);
                    }

                    srcPtrTemp += 12;
                    dstPtrTempR += 4;
                    dstPtrTempG += 4;
                    dstPtrTempB += 4;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount+=3)
                {
                    *dstPtrTempR = (Rpp16f) RPPPIXELCHECKF32((Rpp32f)srcPtrTemp[0] * alpha + beta);
                    *dstPtrTempG = (Rpp16f) RPPPIXELCHECKF32((Rpp32f)srcPtrTemp[1] * alpha + beta);
                    *dstPtrTempB = (Rpp16f) RPPPIXELCHECKF32((Rpp32f)srcPtrTemp[2] * alpha + beta);

                    srcPtrTemp += 3;
                    dstPtrTempR++;
                    dstPtrTempG++;
                    dstPtrTempB++;
                }

                srcPtrRow += srcDescPtr->strides.hStride;
                dstPtrRowR += dstDescPtr->strides.hStride;
                dstPtrRowG += dstDescPtr->strides.hStride;
                dstPtrRowB += dstDescPtr->strides.hStride;
            }
        }

        // Brightness with fused output-layout toggle (NCHW -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp32u alignedLength = bufferLength & ~11;

            Rpp16f *srcPtrRowR, *srcPtrRowG, *srcPtrRowB, *dstPtrRow;
            srcPtrRowR = srcPtrChannel;
            srcPtrRowG = srcPtrRowR + srcDescPtr->strides.cStride;
            srcPtrRowB = srcPtrRowG + srcDescPtr->strides.cStride;
            dstPtrRow = dstPtrChannel;

            for(int i = 0; i < roiPtr->xywhROI.roiHeight; i++)
            {
                Rpp16f *srcPtrTempR, *srcPtrTempG, *srcPtrTempB, *dstPtrTemp;
                srcPtrTempR = srcPtrRowR;
                srcPtrTempG = srcPtrRowG;
                srcPtrTempB = srcPtrRowB;
                dstPtrTemp = dstPtrRow;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount+=4)
                {
                    Rpp32f srcPtrTemp_ps[12], dstPtrTemp_ps[13];

                    for(int cnt = 0; cnt < 4; cnt++)
                    {
                        *(srcPtrTemp_ps + cnt) = (Rpp32f) *(srcPtrTempR + cnt);
                        *(srcPtrTemp_ps + 4 + cnt) = (Rpp32f) *(srcPtrTempG + cnt);
                        *(srcPtrTemp_ps + 8 + cnt) = (Rpp32f) *(srcPtrTempB + cnt);
                    }

                    __m128 p[4];

                    rpp_simd_load(rpp_load12_f32pln3_to_f32pln3, srcPtrTemp_ps, srcPtrTemp_ps + 4, srcPtrTemp_ps + 8, p);    // simd loads
                    p[0] = _mm_fmadd_ps(p[0], pMul, pAdd);    // brightness adjustment Rs
                    p[1] = _mm_fmadd_ps(p[1], pMul, pAdd);    // brightness adjustment Gs
                    p[2] = _mm_fmadd_ps(p[2], pMul, pAdd);    // brightness adjustment Bs
                    rpp_simd_store(rpp_store12_f32pln3_to_f32pkd3, dstPtrTemp_ps, p);    // simd stores

                    for(int cnt = 0; cnt < 12; cnt++)
                    {
                        *(dstPtrTemp + cnt) = (Rpp16f) *(dstPtrTemp_ps + cnt);
                    }

                    srcPtrTempR += 4;
                    srcPtrTempG += 4;
                    srcPtrTempB += 4;
                    dstPtrTemp += 12;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    dstPtrTemp[0] = (Rpp16f) RPPPIXELCHECKF32((Rpp32f)*srcPtrTempR * alpha + beta);
                    dstPtrTemp[1] = (Rpp16f) RPPPIXELCHECKF32((Rpp32f)*srcPtrTempG * alpha + beta);
                    dstPtrTemp[2] = (Rpp16f) RPPPIXELCHECKF32((Rpp32f)*srcPtrTempB * alpha + beta);

                    srcPtrTempR++;
                    srcPtrTempG++;
                    srcPtrTempB++;
                    dstPtrTemp += 3;
                }

                srcPtrRowR += srcDescPtr->strides.hStride;
                srcPtrRowG += srcDescPtr->strides.hStride;
                srcPtrRowB += srcDescPtr->strides.hStride;
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }

        // Brightness without fused output-layout toggle (NHWC -> NHWC or NCHW -> NCHW)
        else
        {
            Rpp32u alignedLength = bufferLength & ~3;

            for(int c = 0; c < layoutParams.channelParam; c++)
            {
                Rpp16f *srcPtrRow, *dstPtrRow;
                srcPtrRow = srcPtrChannel;
                dstPtrRow = dstPtrChannel;

                for(int i = 0; i < roiPtr->xywhROI.roiHeight; i++)
                {
                    Rpp16f *srcPtrTemp, *dstPtrTemp;
                    srcPtrTemp = srcPtrRow;
                    dstPtrTemp = dstPtrRow;

                    int vectorLoopCount = 0;
                    for (; vectorLoopCount < alignedLength; vectorLoopCount+=4)
                    {
                        Rpp32f srcPtrTemp_ps[4], dstPtrTemp_ps[4];

                        for(int cnt = 0; cnt < 4; cnt++)
                        {
                            *(srcPtrTemp_ps + cnt) = (Rpp16f) *(srcPtrTemp + cnt);
                        }

                        __m128 p[1];

                        rpp_simd_load(rpp_load4_f32_to_f32, srcPtrTemp_ps, p);    // simd loads
                        p[0] = _mm_fmadd_ps(p[0], pMul, pAdd);    // brightness adjustment
                        rpp_simd_store(rpp_store4_f32_to_f32, dstPtrTemp_ps, p);    // simd stores

                        for(int cnt = 0; cnt < 4; cnt++)
                        {
                            *(dstPtrTemp + cnt) = (Rpp16f) *(dstPtrTemp_ps + cnt);
                        }

                        srcPtrTemp += 4;
                        dstPtrTemp += 4;
                    }
                    for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                    {
                        *dstPtrTemp = (Rpp16f) RPPPIXELCHECKF32((Rpp32f)*srcPtrTemp * alpha + beta);

                        srcPtrTemp++;
                        dstPtrTemp++;
                    }

                    srcPtrRow += srcDescPtr->strides.hStride;
                    dstPtrRow += dstDescPtr->strides.hStride;
                }

                srcPtrChannel += srcDescPtr->strides.cStride;
                dstPtrChannel += dstDescPtr->strides.cStride;
            }
        }
    }

    return RPP_SUCCESS;
}

RppStatus brightness_i8_i8_host_tensor(Rpp8s *srcPtr,
                                       RpptDescPtr srcDescPtr,
                                       Rpp8s *dstPtr,
                                       RpptDescPtr dstDescPtr,
                                       Rpp32f *alphaTensor,
                                       Rpp32f *betaTensor,
                                       RpptROIPtr roiTensorPtrSrc,
                                       RpptRoiType roiType,
                                       RppLayoutParams layoutParams)
{
    RpptROI roiDefault;
    RpptROIPtr roiPtrDefault;
    roiPtrDefault = &roiDefault;
    roiPtrDefault->xywhROI.xy.x = 0;
    roiPtrDefault->xywhROI.xy.y = 0;
    roiPtrDefault->xywhROI.roiWidth = srcDescPtr->w;
    roiPtrDefault->xywhROI.roiHeight = srcDescPtr->h;

    omp_set_dynamic(0);
#pragma omp parallel for num_threads(dstDescPtr->n)
    for(int batchCount = 0; batchCount < dstDescPtr->n; batchCount++)
    {
        RpptROI roi;
        RpptROIPtr roiPtr;

        if (&roiTensorPtrSrc[batchCount] == NULL)
        {
            roiPtr = roiPtrDefault;
        }
        else
        {
            RpptROIPtr roiPtrInput = &roiTensorPtrSrc[batchCount];

            RpptROI roiImage;
            RpptROIPtr roiPtrImage;

            if (roiType == RpptRoiType::LTRB)
            {
                roiPtrImage = &roiImage;
                compute_xywh_from_ltrb_host(roiPtrInput, roiPtrImage);
            }
            else if (roiType == RpptRoiType::XYWH)
            {
                roiPtrImage = roiPtrInput;
            }

            roiPtr = &roi;
            compute_roi_boundary_check_host(roiPtrImage, roiPtr, roiPtrDefault);
        }

        Rpp32f alpha = alphaTensor[batchCount];
        Rpp32f beta = betaTensor[batchCount];

        Rpp8s *srcPtrImage, *dstPtrImage;
        srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;

        Rpp32u bufferLength = roiPtr->xywhROI.roiWidth * layoutParams.bufferMultiplier;

        __m128 pMul = _mm_set1_ps(alpha);
        __m128 pAdd = _mm_set1_ps(beta);

        Rpp8s *srcPtrChannel, *dstPtrChannel;
        srcPtrChannel = srcPtrImage + (roiPtr->xywhROI.xy.y * srcDescPtr->strides.hStride) + (roiPtr->xywhROI.xy.x * layoutParams.bufferMultiplier);
        dstPtrChannel = dstPtrImage;

        // Brightness with fused output-layout toggle (NHWC -> NCHW)
        if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp32u alignedLength = bufferLength & ~47;

            Rpp8s *srcPtrRow, *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
            srcPtrRow = srcPtrChannel;
            dstPtrRowR = dstPtrChannel;
            dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
            dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;

            for(int i = 0; i < roiPtr->xywhROI.roiHeight; i++)
            {
                Rpp8s *srcPtrTemp, *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
                srcPtrTemp = srcPtrRow;
                dstPtrTempR = dstPtrRowR;
                dstPtrTempG = dstPtrRowG;
                dstPtrTempB = dstPtrRowB;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount+=48)
                {
                    __m128 p[12];

                    rpp_simd_load(rpp_load48_i8pkd3_to_f32pln3, srcPtrTemp, p);    // simd loads
                    p[0] = _mm_fmadd_ps(p[0], pMul, pAdd);    // brightness adjustment Rs
                    p[1] = _mm_fmadd_ps(p[1], pMul, pAdd);    // brightness adjustment Rs
                    p[2] = _mm_fmadd_ps(p[2], pMul, pAdd);    // brightness adjustment Rs
                    p[3] = _mm_fmadd_ps(p[3], pMul, pAdd);    // brightness adjustment Rs
                    p[4] = _mm_fmadd_ps(p[4], pMul, pAdd);    // brightness adjustment Gs
                    p[5] = _mm_fmadd_ps(p[5], pMul, pAdd);    // brightness adjustment Gs
                    p[6] = _mm_fmadd_ps(p[6], pMul, pAdd);    // brightness adjustment Gs
                    p[7] = _mm_fmadd_ps(p[7], pMul, pAdd);    // brightness adjustment Gs
                    p[8] = _mm_fmadd_ps(p[8], pMul, pAdd);    // brightness adjustment Bs
                    p[9] = _mm_fmadd_ps(p[9], pMul, pAdd);    // brightness adjustment Bs
                    p[10] = _mm_fmadd_ps(p[10], pMul, pAdd);    // brightness adjustment Bs
                    p[11] = _mm_fmadd_ps(p[11], pMul, pAdd);    // brightness adjustment Bs
                    rpp_simd_store(rpp_store48_f32pln3_to_i8pln3, dstPtrTempR, dstPtrTempG, dstPtrTempB, p);    // simd stores

                    srcPtrTemp += 48;
                    dstPtrTempR += 16;
                    dstPtrTempG += 16;
                    dstPtrTempB += 16;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount+=3)
                {
                    *dstPtrTempR = (Rpp8s) RPPPIXELCHECKI8((((Rpp32f) (srcPtrTemp[0]) + 128) * alpha) + beta - 128);
                    *dstPtrTempG = (Rpp8s) RPPPIXELCHECKI8((((Rpp32f) (srcPtrTemp[1]) + 128) * alpha) + beta - 128);
                    *dstPtrTempB = (Rpp8s) RPPPIXELCHECKI8((((Rpp32f) (srcPtrTemp[2]) + 128) * alpha) + beta - 128);

                    srcPtrTemp += 3;
                    dstPtrTempR++;
                    dstPtrTempG++;
                    dstPtrTempB++;
                }

                srcPtrRow += srcDescPtr->strides.hStride;
                dstPtrRowR += dstDescPtr->strides.hStride;
                dstPtrRowG += dstDescPtr->strides.hStride;
                dstPtrRowB += dstDescPtr->strides.hStride;
            }
        }

        // Brightness with fused output-layout toggle (NCHW -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp32u alignedLength = bufferLength & ~47;

            Rpp8s *srcPtrRowR, *srcPtrRowG, *srcPtrRowB, *dstPtrRow;
            srcPtrRowR = srcPtrChannel;
            srcPtrRowG = srcPtrRowR + srcDescPtr->strides.cStride;
            srcPtrRowB = srcPtrRowG + srcDescPtr->strides.cStride;
            dstPtrRow = dstPtrChannel;

            for(int i = 0; i < roiPtr->xywhROI.roiHeight; i++)
            {
                Rpp8s *srcPtrTempR, *srcPtrTempG, *srcPtrTempB, *dstPtrTemp;
                srcPtrTempR = srcPtrRowR;
                srcPtrTempG = srcPtrRowG;
                srcPtrTempB = srcPtrRowB;
                dstPtrTemp = dstPtrRow;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount+=16)
                {
                    __m128 p[12];

                    rpp_simd_load(rpp_load48_i8pln3_to_f32pln3, srcPtrTempR, srcPtrTempG, srcPtrTempB, p);    // simd loads
                    p[0] = _mm_fmadd_ps(p[0], pMul, pAdd);    // brightness adjustment Rs
                    p[1] = _mm_fmadd_ps(p[1], pMul, pAdd);    // brightness adjustment Rs
                    p[2] = _mm_fmadd_ps(p[2], pMul, pAdd);    // brightness adjustment Rs
                    p[3] = _mm_fmadd_ps(p[3], pMul, pAdd);    // brightness adjustment Rs
                    p[4] = _mm_fmadd_ps(p[4], pMul, pAdd);    // brightness adjustment Gs
                    p[5] = _mm_fmadd_ps(p[5], pMul, pAdd);    // brightness adjustment Gs
                    p[6] = _mm_fmadd_ps(p[6], pMul, pAdd);    // brightness adjustment Gs
                    p[7] = _mm_fmadd_ps(p[7], pMul, pAdd);    // brightness adjustment Gs
                    p[8] = _mm_fmadd_ps(p[8], pMul, pAdd);    // brightness adjustment Bs
                    p[9] = _mm_fmadd_ps(p[9], pMul, pAdd);    // brightness adjustment Bs
                    p[10] = _mm_fmadd_ps(p[10], pMul, pAdd);    // brightness adjustment Bs
                    p[11] = _mm_fmadd_ps(p[11], pMul, pAdd);    // brightness adjustment Bs
                    rpp_simd_store(rpp_store48_f32pln3_to_i8pkd3, dstPtrTemp, p);    // simd stores

                    srcPtrTempR += 16;
                    srcPtrTempG += 16;
                    srcPtrTempB += 16;
                    dstPtrTemp += 48;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    dstPtrTemp[0] = (Rpp8s) RPPPIXELCHECKI8((((Rpp32f) (*srcPtrTempR) + 128) * alpha) + beta - 128);
                    dstPtrTemp[1] = (Rpp8s) RPPPIXELCHECKI8((((Rpp32f) (*srcPtrTempG) + 128) * alpha) + beta - 128);
                    dstPtrTemp[2] = (Rpp8s) RPPPIXELCHECKI8((((Rpp32f) (*srcPtrTempB) + 128) * alpha) + beta - 128);

                    srcPtrTempR++;
                    srcPtrTempG++;
                    srcPtrTempB++;
                    dstPtrTemp += 3;
                }

                srcPtrRowR += srcDescPtr->strides.hStride;
                srcPtrRowG += srcDescPtr->strides.hStride;
                srcPtrRowB += srcDescPtr->strides.hStride;
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }

        // Brightness without fused output-layout toggle (NHWC -> NHWC or NCHW -> NCHW)
        else
        {
            Rpp32u alignedLength = bufferLength & ~15;

            for(int c = 0; c < layoutParams.channelParam; c++)
            {
                Rpp8s *srcPtrRow, *dstPtrRow;
                srcPtrRow = srcPtrChannel;
                dstPtrRow = dstPtrChannel;

                for(int i = 0; i < roiPtr->xywhROI.roiHeight; i++)
                {
                    Rpp8s *srcPtrTemp, *dstPtrTemp;
                    srcPtrTemp = srcPtrRow;
                    dstPtrTemp = dstPtrRow;

                    int vectorLoopCount = 0;
                    for (; vectorLoopCount < alignedLength; vectorLoopCount+=16)
                    {
                        __m128 p[4];

                        rpp_simd_load(rpp_load16_i8_to_f32, srcPtrTemp, p);    // simd loads
                        p[0] = _mm_fmadd_ps(p[0], pMul, pAdd);    // brightness adjustment
                        p[1] = _mm_fmadd_ps(p[1], pMul, pAdd);    // brightness adjustment
                        p[2] = _mm_fmadd_ps(p[2], pMul, pAdd);    // brightness adjustment
                        p[3] = _mm_fmadd_ps(p[3], pMul, pAdd);    // brightness adjustment
                        rpp_simd_store(rpp_store16_f32_to_i8, dstPtrTemp, p);    // simd stores

                        srcPtrTemp +=16;
                        dstPtrTemp +=16;
                    }
                    for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                    {
                        *dstPtrTemp = (Rpp8s) RPPPIXELCHECKI8((((Rpp32f) (*srcPtrTemp) + 128) * alpha) + beta - 128);

                        srcPtrTemp++;
                        dstPtrTemp++;
                    }

                    srcPtrRow += srcDescPtr->strides.hStride;
                    dstPtrRow += dstDescPtr->strides.hStride;
                }

                srcPtrChannel += srcDescPtr->strides.cStride;
                dstPtrChannel += dstDescPtr->strides.cStride;
            }
        }
    }

    return RPP_SUCCESS;
}

/************ gamma_correction ************/

RppStatus gamma_correction_u8_u8_host_tensor(Rpp8u *srcPtr,
                                             RpptDescPtr srcDescPtr,
                                             Rpp8u *dstPtr,
                                             RpptDescPtr dstDescPtr,
                                             Rpp32f *gammaTensor,
                                             RpptROIPtr roiTensorPtrSrc,
                                             RpptRoiType roiType,
                                             RppLayoutParams layoutParams)
{
    RpptROI roiDefault;
    RpptROIPtr roiPtrDefault;
    roiPtrDefault = &roiDefault;
    roiPtrDefault->xywhROI.xy.x = 0;
    roiPtrDefault->xywhROI.xy.y = 0;
    roiPtrDefault->xywhROI.roiWidth = srcDescPtr->w;
    roiPtrDefault->xywhROI.roiHeight = srcDescPtr->h;

    omp_set_dynamic(0);
#pragma omp parallel for num_threads(dstDescPtr->n)
    for(int batchCount = 0; batchCount < dstDescPtr->n; batchCount++)
    {
        RpptROI roi;
        RpptROIPtr roiPtr;

        if (&roiTensorPtrSrc[batchCount] == NULL)
        {
            roiPtr = roiPtrDefault;
        }
        else
        {
            RpptROIPtr roiPtrInput = &roiTensorPtrSrc[batchCount];

            RpptROI roiImage;
            RpptROIPtr roiPtrImage;

            if (roiType == RpptRoiType::LTRB)
            {
                roiPtrImage = &roiImage;
                compute_xywh_from_ltrb_host(roiPtrInput, roiPtrImage);
            }
            else if (roiType == RpptRoiType::XYWH)
            {
                roiPtrImage = roiPtrInput;
            }

            roiPtr = &roi;
            compute_roi_boundary_check_host(roiPtrImage, roiPtr, roiPtrDefault);
        }

        Rpp32f gamma = gammaTensor[batchCount];

        Rpp8u gammaLUT[256];
        for (int i = 0; i < 256; i++)
        {
            gammaLUT[i] = (Rpp8u) RPPPIXELCHECK(pow((((Rpp32f) i) * 0.003922f), gamma) * 255.0);
        }

        Rpp8u *srcPtrImage, *dstPtrImage;
        srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;

        Rpp32u bufferLength = roiPtr->xywhROI.roiWidth * layoutParams.bufferMultiplier;

        Rpp8u *srcPtrChannel, *dstPtrChannel;
        srcPtrChannel = srcPtrImage + (roiPtr->xywhROI.xy.y * srcDescPtr->strides.hStride) + (roiPtr->xywhROI.xy.x * layoutParams.bufferMultiplier);
        dstPtrChannel = dstPtrImage;

        // Gamma correction with fused output-layout toggle (NHWC -> NCHW)
        if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp8u *srcPtrRow, *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
            srcPtrRow = srcPtrChannel;
            dstPtrRowR = dstPtrChannel;
            dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
            dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;

            for(int i = 0; i < roiPtr->xywhROI.roiHeight; i++)
            {
                Rpp8u *srcPtrTemp, *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
                srcPtrTemp = srcPtrRow;
                dstPtrTempR = dstPtrRowR;
                dstPtrTempG = dstPtrRowG;
                dstPtrTempB = dstPtrRowB;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < bufferLength; vectorLoopCount+=3)
                {
                    *dstPtrTempR = gammaLUT[srcPtrTemp[0]];
                    *dstPtrTempG = gammaLUT[srcPtrTemp[1]];
                    *dstPtrTempB = gammaLUT[srcPtrTemp[2]];

                    srcPtrTemp += 3;
                    dstPtrTempR++;
                    dstPtrTempG++;
                    dstPtrTempB++;
                }

                srcPtrRow += srcDescPtr->strides.hStride;
                dstPtrRowR += dstDescPtr->strides.hStride;
                dstPtrRowG += dstDescPtr->strides.hStride;
                dstPtrRowB += dstDescPtr->strides.hStride;
            }
        }

        // Gamma correction with fused output-layout toggle (NCHW -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp8u *srcPtrRowR, *srcPtrRowG, *srcPtrRowB, *dstPtrRow;
            srcPtrRowR = srcPtrChannel;
            srcPtrRowG = srcPtrRowR + srcDescPtr->strides.cStride;
            srcPtrRowB = srcPtrRowG + srcDescPtr->strides.cStride;
            dstPtrRow = dstPtrChannel;

            for(int i = 0; i < roiPtr->xywhROI.roiHeight; i++)
            {
                Rpp8u *srcPtrTempR, *srcPtrTempG, *srcPtrTempB, *dstPtrTemp;
                srcPtrTempR = srcPtrRowR;
                srcPtrTempG = srcPtrRowG;
                srcPtrTempB = srcPtrRowB;
                dstPtrTemp = dstPtrRow;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    dstPtrTemp[0] = gammaLUT[*srcPtrTempR];
                    dstPtrTemp[1] = gammaLUT[*srcPtrTempG];
                    dstPtrTemp[2] = gammaLUT[*srcPtrTempB];

                    srcPtrTempR++;
                    srcPtrTempG++;
                    srcPtrTempB++;
                    dstPtrTemp += 3;
                }

                srcPtrRowR += srcDescPtr->strides.hStride;
                srcPtrRowG += srcDescPtr->strides.hStride;
                srcPtrRowB += srcDescPtr->strides.hStride;
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }

        // Gamma correction without fused output-layout toggle (NHWC -> NHWC or NCHW -> NCHW)
        else
        {
            for(int c = 0; c < layoutParams.channelParam; c++)
            {
                Rpp8u *srcPtrRow, *dstPtrRow;
                srcPtrRow = srcPtrChannel;
                dstPtrRow = dstPtrChannel;

                for(int i = 0; i < roiPtr->xywhROI.roiHeight; i++)
                {
                    Rpp8u *srcPtrTemp, *dstPtrTemp;
                    srcPtrTemp = srcPtrRow;
                    dstPtrTemp = dstPtrRow;

                    int vectorLoopCount = 0;
                    for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                    {
                        *dstPtrTemp = gammaLUT[*srcPtrTemp];

                        srcPtrTemp++;
                        dstPtrTemp++;
                    }

                    srcPtrRow += srcDescPtr->strides.hStride;
                    dstPtrRow += dstDescPtr->strides.hStride;
                }

                srcPtrChannel += srcDescPtr->strides.cStride;
                dstPtrChannel += dstDescPtr->strides.cStride;
            }
        }
    }

    return RPP_SUCCESS;
}

RppStatus gamma_correction_f32_f32_host_tensor(Rpp32f *srcPtr,
                                               RpptDescPtr srcDescPtr,
                                               Rpp32f *dstPtr,
                                               RpptDescPtr dstDescPtr,
                                               Rpp32f *gammaTensor,
                                               RpptROIPtr roiTensorPtrSrc,
                                               RpptRoiType roiType,
                                               RppLayoutParams layoutParams)
{
    RpptROI roiDefault;
    RpptROIPtr roiPtrDefault;
    roiPtrDefault = &roiDefault;
    roiPtrDefault->xywhROI.xy.x = 0;
    roiPtrDefault->xywhROI.xy.y = 0;
    roiPtrDefault->xywhROI.roiWidth = srcDescPtr->w;
    roiPtrDefault->xywhROI.roiHeight = srcDescPtr->h;

    omp_set_dynamic(0);
#pragma omp parallel for num_threads(dstDescPtr->n)
    for(int batchCount = 0; batchCount < dstDescPtr->n; batchCount++)
    {
        RpptROI roi;
        RpptROIPtr roiPtr;

        if (&roiTensorPtrSrc[batchCount] == NULL)
        {
            roiPtr = roiPtrDefault;
        }
        else
        {
            RpptROIPtr roiPtrInput = &roiTensorPtrSrc[batchCount];

            RpptROI roiImage;
            RpptROIPtr roiPtrImage;

            if (roiType == RpptRoiType::LTRB)
            {
                roiPtrImage = &roiImage;
                compute_xywh_from_ltrb_host(roiPtrInput, roiPtrImage);
            }
            else if (roiType == RpptRoiType::XYWH)
            {
                roiPtrImage = roiPtrInput;
            }

            roiPtr = &roi;
            compute_roi_boundary_check_host(roiPtrImage, roiPtr, roiPtrDefault);
        }

        Rpp32f gamma = gammaTensor[batchCount];

        Rpp32f gammaLUT[256];
        for (int i = 0; i < 256; i++)
        {
            gammaLUT[i] = (Rpp32f) pow((((Rpp32f) i) * 0.003922f), gamma);
        }

        Rpp32f *srcPtrImage, *dstPtrImage;
        srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;

        Rpp32u bufferLength = roiPtr->xywhROI.roiWidth * layoutParams.bufferMultiplier;

        Rpp32f *srcPtrChannel, *dstPtrChannel;
        srcPtrChannel = srcPtrImage + (roiPtr->xywhROI.xy.y * srcDescPtr->strides.hStride) + (roiPtr->xywhROI.xy.x * layoutParams.bufferMultiplier);
        dstPtrChannel = dstPtrImage;

        // Gamma correction with fused output-layout toggle (NHWC -> NCHW)
        if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp32f *srcPtrRow, *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
            srcPtrRow = srcPtrChannel;
            dstPtrRowR = dstPtrChannel;
            dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
            dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;

            for(int i = 0; i < roiPtr->xywhROI.roiHeight; i++)
            {
                Rpp32f *srcPtrTemp, *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
                srcPtrTemp = srcPtrRow;
                dstPtrTempR = dstPtrRowR;
                dstPtrTempG = dstPtrRowG;
                dstPtrTempB = dstPtrRowB;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < bufferLength; vectorLoopCount+=3)
                {
                    *dstPtrTempR = gammaLUT[(int) (RPPPIXELCHECK(srcPtrTemp[0] * 255))];
                    *dstPtrTempG = gammaLUT[(int) (RPPPIXELCHECK(srcPtrTemp[1] * 255))];
                    *dstPtrTempB = gammaLUT[(int) (RPPPIXELCHECK(srcPtrTemp[2] * 255))];

                    srcPtrTemp += 3;
                    dstPtrTempR++;
                    dstPtrTempG++;
                    dstPtrTempB++;
                }

                srcPtrRow += srcDescPtr->strides.hStride;
                dstPtrRowR += dstDescPtr->strides.hStride;
                dstPtrRowG += dstDescPtr->strides.hStride;
                dstPtrRowB += dstDescPtr->strides.hStride;
            }
        }

        // Gamma correction with fused output-layout toggle (NCHW -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp32f *srcPtrRowR, *srcPtrRowG, *srcPtrRowB, *dstPtrRow;
            srcPtrRowR = srcPtrChannel;
            srcPtrRowG = srcPtrRowR + srcDescPtr->strides.cStride;
            srcPtrRowB = srcPtrRowG + srcDescPtr->strides.cStride;
            dstPtrRow = dstPtrChannel;

            for(int i = 0; i < roiPtr->xywhROI.roiHeight; i++)
            {
                Rpp32f *srcPtrTempR, *srcPtrTempG, *srcPtrTempB, *dstPtrTemp;
                srcPtrTempR = srcPtrRowR;
                srcPtrTempG = srcPtrRowG;
                srcPtrTempB = srcPtrRowB;
                dstPtrTemp = dstPtrRow;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    dstPtrTemp[0] = gammaLUT[(int) (RPPPIXELCHECK(*srcPtrTempR * 255))];
                    dstPtrTemp[1] = gammaLUT[(int) (RPPPIXELCHECK(*srcPtrTempG * 255))];
                    dstPtrTemp[2] = gammaLUT[(int) (RPPPIXELCHECK(*srcPtrTempB * 255))];

                    srcPtrTempR++;
                    srcPtrTempG++;
                    srcPtrTempB++;
                    dstPtrTemp += 3;
                }

                srcPtrRowR += srcDescPtr->strides.hStride;
                srcPtrRowG += srcDescPtr->strides.hStride;
                srcPtrRowB += srcDescPtr->strides.hStride;
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }

        // Gamma correction without fused output-layout toggle (NHWC -> NHWC or NCHW -> NCHW)
        else
        {
            for(int c = 0; c < layoutParams.channelParam; c++)
            {
                Rpp32f *srcPtrRow, *dstPtrRow;
                srcPtrRow = srcPtrChannel;
                dstPtrRow = dstPtrChannel;

                for(int i = 0; i < roiPtr->xywhROI.roiHeight; i++)
                {
                    Rpp32f *srcPtrTemp, *dstPtrTemp;
                    srcPtrTemp = srcPtrRow;
                    dstPtrTemp = dstPtrRow;

                    int vectorLoopCount = 0;
                    for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                    {
                        *dstPtrTemp = gammaLUT[(int) (RPPPIXELCHECK(*srcPtrTemp * 255))];

                        srcPtrTemp++;
                        dstPtrTemp++;
                    }

                    srcPtrRow += srcDescPtr->strides.hStride;
                    dstPtrRow += dstDescPtr->strides.hStride;
                }

                srcPtrChannel += srcDescPtr->strides.cStride;
                dstPtrChannel += dstDescPtr->strides.cStride;
            }
        }
    }

    return RPP_SUCCESS;
}

RppStatus gamma_correction_f16_f16_host_tensor(Rpp16f *srcPtr,
                                               RpptDescPtr srcDescPtr,
                                               Rpp16f *dstPtr,
                                               RpptDescPtr dstDescPtr,
                                               Rpp32f *gammaTensor,
                                               RpptROIPtr roiTensorPtrSrc,
                                               RpptRoiType roiType,
                                               RppLayoutParams layoutParams)
{
    RpptROI roiDefault;
    RpptROIPtr roiPtrDefault;
    roiPtrDefault = &roiDefault;
    roiPtrDefault->xywhROI.xy.x = 0;
    roiPtrDefault->xywhROI.xy.y = 0;
    roiPtrDefault->xywhROI.roiWidth = srcDescPtr->w;
    roiPtrDefault->xywhROI.roiHeight = srcDescPtr->h;

    omp_set_dynamic(0);
#pragma omp parallel for num_threads(dstDescPtr->n)
    for(int batchCount = 0; batchCount < dstDescPtr->n; batchCount++)
    {
        RpptROI roi;
        RpptROIPtr roiPtr;

        if (&roiTensorPtrSrc[batchCount] == NULL)
        {
            roiPtr = roiPtrDefault;
        }
        else
        {
            RpptROIPtr roiPtrInput = &roiTensorPtrSrc[batchCount];

            RpptROI roiImage;
            RpptROIPtr roiPtrImage;

            if (roiType == RpptRoiType::LTRB)
            {
                roiPtrImage = &roiImage;
                compute_xywh_from_ltrb_host(roiPtrInput, roiPtrImage);
            }
            else if (roiType == RpptRoiType::XYWH)
            {
                roiPtrImage = roiPtrInput;
            }

            roiPtr = &roi;
            compute_roi_boundary_check_host(roiPtrImage, roiPtr, roiPtrDefault);
        }

        Rpp32f gamma = gammaTensor[batchCount];

        Rpp32f gammaLUT[256];
        for (int i = 0; i < 256; i++)
        {
            gammaLUT[i] = (Rpp32f) pow((((Rpp32f) i) * 0.003922f), gamma);
        }

        Rpp16f *srcPtrImage, *dstPtrImage;
        srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;

        Rpp32u bufferLength = roiPtr->xywhROI.roiWidth * layoutParams.bufferMultiplier;

        Rpp16f *srcPtrChannel, *dstPtrChannel;
        srcPtrChannel = srcPtrImage + (roiPtr->xywhROI.xy.y * srcDescPtr->strides.hStride) + (roiPtr->xywhROI.xy.x * layoutParams.bufferMultiplier);
        dstPtrChannel = dstPtrImage;

        // Gamma correction with fused output-layout toggle (NHWC -> NCHW)
        if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp16f *srcPtrRow, *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
            srcPtrRow = srcPtrChannel;
            dstPtrRowR = dstPtrChannel;
            dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
            dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;

            for(int i = 0; i < roiPtr->xywhROI.roiHeight; i++)
            {
                Rpp16f *srcPtrTemp, *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
                srcPtrTemp = srcPtrRow;
                dstPtrTempR = dstPtrRowR;
                dstPtrTempG = dstPtrRowG;
                dstPtrTempB = dstPtrRowB;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < bufferLength; vectorLoopCount+=3)
                {
                    *dstPtrTempR = (Rpp16f) gammaLUT[(int) (RPPPIXELCHECK(srcPtrTemp[0] * 255))];
                    *dstPtrTempG = (Rpp16f) gammaLUT[(int) (RPPPIXELCHECK(srcPtrTemp[1] * 255))];
                    *dstPtrTempB = (Rpp16f) gammaLUT[(int) (RPPPIXELCHECK(srcPtrTemp[2] * 255))];

                    srcPtrTemp += 3;
                    dstPtrTempR++;
                    dstPtrTempG++;
                    dstPtrTempB++;
                }

                srcPtrRow += srcDescPtr->strides.hStride;
                dstPtrRowR += dstDescPtr->strides.hStride;
                dstPtrRowG += dstDescPtr->strides.hStride;
                dstPtrRowB += dstDescPtr->strides.hStride;
            }
        }

        // Gamma correction with fused output-layout toggle (NCHW -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp16f *srcPtrRowR, *srcPtrRowG, *srcPtrRowB, *dstPtrRow;
            srcPtrRowR = srcPtrChannel;
            srcPtrRowG = srcPtrRowR + srcDescPtr->strides.cStride;
            srcPtrRowB = srcPtrRowG + srcDescPtr->strides.cStride;
            dstPtrRow = dstPtrChannel;

            for(int i = 0; i < roiPtr->xywhROI.roiHeight; i++)
            {
                Rpp16f *srcPtrTempR, *srcPtrTempG, *srcPtrTempB, *dstPtrTemp;
                srcPtrTempR = srcPtrRowR;
                srcPtrTempG = srcPtrRowG;
                srcPtrTempB = srcPtrRowB;
                dstPtrTemp = dstPtrRow;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    dstPtrTemp[0] = (Rpp16f) gammaLUT[(int) (RPPPIXELCHECK(*srcPtrTempR * 255))];
                    dstPtrTemp[1] = (Rpp16f) gammaLUT[(int) (RPPPIXELCHECK(*srcPtrTempG * 255))];
                    dstPtrTemp[2] = (Rpp16f) gammaLUT[(int) (RPPPIXELCHECK(*srcPtrTempB * 255))];

                    srcPtrTempR++;
                    srcPtrTempG++;
                    srcPtrTempB++;
                    dstPtrTemp += 3;
                }

                srcPtrRowR += srcDescPtr->strides.hStride;
                srcPtrRowG += srcDescPtr->strides.hStride;
                srcPtrRowB += srcDescPtr->strides.hStride;
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }

        // Gamma correction without fused output-layout toggle (NHWC -> NHWC or NCHW -> NCHW)
        else
        {
            for(int c = 0; c < layoutParams.channelParam; c++)
            {
                Rpp16f *srcPtrRow, *dstPtrRow;
                srcPtrRow = srcPtrChannel;
                dstPtrRow = dstPtrChannel;

                for(int i = 0; i < roiPtr->xywhROI.roiHeight; i++)
                {
                    Rpp16f *srcPtrTemp, *dstPtrTemp;
                    srcPtrTemp = srcPtrRow;
                    dstPtrTemp = dstPtrRow;

                    int vectorLoopCount = 0;
                    for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                    {
                        *dstPtrTemp = (Rpp16f) gammaLUT[(int) (RPPPIXELCHECK(*srcPtrTemp * 255))];

                        srcPtrTemp++;
                        dstPtrTemp++;
                    }

                    srcPtrRow += srcDescPtr->strides.hStride;
                    dstPtrRow += dstDescPtr->strides.hStride;
                }

                srcPtrChannel += srcDescPtr->strides.cStride;
                dstPtrChannel += dstDescPtr->strides.cStride;
            }
        }
    }

    return RPP_SUCCESS;
}

RppStatus gamma_correction_i8_i8_host_tensor(Rpp8s *srcPtr,
                                             RpptDescPtr srcDescPtr,
                                             Rpp8s *dstPtr,
                                             RpptDescPtr dstDescPtr,
                                             Rpp32f *gammaTensor,
                                             RpptROIPtr roiTensorPtrSrc,
                                             RpptRoiType roiType,
                                             RppLayoutParams layoutParams)
{
    RpptROI roiDefault;
    RpptROIPtr roiPtrDefault;
    roiPtrDefault = &roiDefault;
    roiPtrDefault->xywhROI.xy.x = 0;
    roiPtrDefault->xywhROI.xy.y = 0;
    roiPtrDefault->xywhROI.roiWidth = srcDescPtr->w;
    roiPtrDefault->xywhROI.roiHeight = srcDescPtr->h;

    omp_set_dynamic(0);
#pragma omp parallel for num_threads(dstDescPtr->n)
    for(int batchCount = 0; batchCount < dstDescPtr->n; batchCount++)
    {
        RpptROI roi;
        RpptROIPtr roiPtr;

        if (&roiTensorPtrSrc[batchCount] == NULL)
        {
            roiPtr = roiPtrDefault;
        }
        else
        {
            RpptROIPtr roiPtrInput = &roiTensorPtrSrc[batchCount];

            RpptROI roiImage;
            RpptROIPtr roiPtrImage;

            if (roiType == RpptRoiType::LTRB)
            {
                roiPtrImage = &roiImage;
                compute_xywh_from_ltrb_host(roiPtrInput, roiPtrImage);
            }
            else if (roiType == RpptRoiType::XYWH)
            {
                roiPtrImage = roiPtrInput;
            }

            roiPtr = &roi;
            compute_roi_boundary_check_host(roiPtrImage, roiPtr, roiPtrDefault);
        }

        Rpp32f gamma = gammaTensor[batchCount];

        Rpp8s gammaLUT[256];
        for (int i = 0; i < 256; i++)
        {
            gammaLUT[i] = (Rpp8s) (RPPPIXELCHECK(pow((((Rpp32f) i) * 0.003922f), gamma) * 255.0) - 128);
        }

        Rpp8s *srcPtrImage, *dstPtrImage;
        srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;

        Rpp32u bufferLength = roiPtr->xywhROI.roiWidth * layoutParams.bufferMultiplier;

        Rpp8s *srcPtrChannel, *dstPtrChannel;
        srcPtrChannel = srcPtrImage + (roiPtr->xywhROI.xy.y * srcDescPtr->strides.hStride) + (roiPtr->xywhROI.xy.x * layoutParams.bufferMultiplier);
        dstPtrChannel = dstPtrImage;

        // Gamma correction with fused output-layout toggle (NHWC -> NCHW)
        if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp8s *srcPtrRow, *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
            srcPtrRow = srcPtrChannel;
            dstPtrRowR = dstPtrChannel;
            dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
            dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;

            for(int i = 0; i < roiPtr->xywhROI.roiHeight; i++)
            {
                Rpp8s *srcPtrTemp, *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
                srcPtrTemp = srcPtrRow;
                dstPtrTempR = dstPtrRowR;
                dstPtrTempG = dstPtrRowG;
                dstPtrTempB = dstPtrRowB;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < bufferLength; vectorLoopCount+=3)
                {
                    *dstPtrTempR = gammaLUT[(Rpp32s)(srcPtrTemp[0]) + 128];
                    *dstPtrTempG = gammaLUT[(Rpp32s)(srcPtrTemp[1]) + 128];
                    *dstPtrTempB = gammaLUT[(Rpp32s)(srcPtrTemp[2]) + 128];

                    srcPtrTemp += 3;
                    dstPtrTempR++;
                    dstPtrTempG++;
                    dstPtrTempB++;
                }

                srcPtrRow += srcDescPtr->strides.hStride;
                dstPtrRowR += dstDescPtr->strides.hStride;
                dstPtrRowG += dstDescPtr->strides.hStride;
                dstPtrRowB += dstDescPtr->strides.hStride;
            }
        }

        // Gamma correction with fused output-layout toggle (NCHW -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp8s *srcPtrRowR, *srcPtrRowG, *srcPtrRowB, *dstPtrRow;
            srcPtrRowR = srcPtrChannel;
            srcPtrRowG = srcPtrRowR + srcDescPtr->strides.cStride;
            srcPtrRowB = srcPtrRowG + srcDescPtr->strides.cStride;
            dstPtrRow = dstPtrChannel;

            for(int i = 0; i < roiPtr->xywhROI.roiHeight; i++)
            {
                Rpp8s *srcPtrTempR, *srcPtrTempG, *srcPtrTempB, *dstPtrTemp;
                srcPtrTempR = srcPtrRowR;
                srcPtrTempG = srcPtrRowG;
                srcPtrTempB = srcPtrRowB;
                dstPtrTemp = dstPtrRow;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    dstPtrTemp[0] = gammaLUT[(Rpp32s)(*srcPtrTempR) + 128];
                    dstPtrTemp[1] = gammaLUT[(Rpp32s)(*srcPtrTempG) + 128];
                    dstPtrTemp[2] = gammaLUT[(Rpp32s)(*srcPtrTempB) + 128];

                    srcPtrTempR++;
                    srcPtrTempG++;
                    srcPtrTempB++;
                    dstPtrTemp += 3;
                }

                srcPtrRowR += srcDescPtr->strides.hStride;
                srcPtrRowG += srcDescPtr->strides.hStride;
                srcPtrRowB += srcDescPtr->strides.hStride;
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }

        // Gamma correction without fused output-layout toggle (NHWC -> NHWC or NCHW -> NCHW)
        else
        {
            for(int c = 0; c < layoutParams.channelParam; c++)
            {
                Rpp8s *srcPtrRow, *dstPtrRow;
                srcPtrRow = srcPtrChannel;
                dstPtrRow = dstPtrChannel;

                for(int i = 0; i < roiPtr->xywhROI.roiHeight; i++)
                {
                    Rpp8s *srcPtrTemp, *dstPtrTemp;
                    srcPtrTemp = srcPtrRow;
                    dstPtrTemp = dstPtrRow;

                    int vectorLoopCount = 0;
                    for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                    {
                        *dstPtrTemp = gammaLUT[(Rpp32s)(*srcPtrTemp) + 128];

                        srcPtrTemp++;
                        dstPtrTemp++;
                    }

                    srcPtrRow += srcDescPtr->strides.hStride;
                    dstPtrRow += dstDescPtr->strides.hStride;
                }

                srcPtrChannel += srcDescPtr->strides.cStride;
                dstPtrChannel += dstDescPtr->strides.cStride;
            }
        }
    }

    return RPP_SUCCESS;
}

/************ blend ************/

RppStatus blend_u8_u8_host_tensor(Rpp8u *srcPtr1,
                                  Rpp8u *srcPtr2,
                                  RpptDescPtr srcDescPtr,
                                  Rpp8u *dstPtr,
                                  RpptDescPtr dstDescPtr,
                                  Rpp32f *alphaTensor,
                                  RpptROIPtr roiTensorPtrSrc,
                                  RpptRoiType roiType,
                                  RppLayoutParams layoutParams)
{
    RpptROI roiDefault;
    RpptROIPtr roiPtrDefault;
    roiPtrDefault = &roiDefault;
    roiPtrDefault->xywhROI.xy.x = 0;
    roiPtrDefault->xywhROI.xy.y = 0;
    roiPtrDefault->xywhROI.roiWidth = srcDescPtr->w;
    roiPtrDefault->xywhROI.roiHeight = srcDescPtr->h;

    omp_set_dynamic(0);
#pragma omp parallel for num_threads(dstDescPtr->n)
    for(int batchCount = 0; batchCount < dstDescPtr->n; batchCount++)
    {
        RpptROI roi;
        RpptROIPtr roiPtr;

        if (&roiTensorPtrSrc[batchCount] == NULL)
        {
            roiPtr = roiPtrDefault;
        }
        else
        {
            RpptROIPtr roiPtrInput = &roiTensorPtrSrc[batchCount];

            RpptROI roiImage;
            RpptROIPtr roiPtrImage;

            if (roiType == RpptRoiType::LTRB)
            {
                roiPtrImage = &roiImage;
                compute_xywh_from_ltrb_host(roiPtrInput, roiPtrImage);
            }
            else if (roiType == RpptRoiType::XYWH)
            {
                roiPtrImage = roiPtrInput;
            }

            roiPtr = &roi;
            compute_roi_boundary_check_host(roiPtrImage, roiPtr, roiPtrDefault);
        }

        Rpp32f alpha = alphaTensor[batchCount];

        Rpp8u *srcPtr1Image, *srcPtr2Image, *dstPtrImage;
        srcPtr1Image = srcPtr1 + batchCount * srcDescPtr->strides.nStride;
        srcPtr2Image = srcPtr2 + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;

        Rpp32u bufferLength = roiPtr->xywhROI.roiWidth * layoutParams.bufferMultiplier;

        __m128 pMul = _mm_set1_ps(alpha);

        Rpp8u *srcPtr1Channel, *srcPtr2Channel, *dstPtrChannel;
        srcPtr1Channel = srcPtr1Image + (roiPtr->xywhROI.xy.y * srcDescPtr->strides.hStride) + (roiPtr->xywhROI.xy.x * layoutParams.bufferMultiplier);
        srcPtr2Channel = srcPtr2Image + (roiPtr->xywhROI.xy.y * srcDescPtr->strides.hStride) + (roiPtr->xywhROI.xy.x * layoutParams.bufferMultiplier);
        dstPtrChannel = dstPtrImage;

        // Blend with fused output-layout toggle (NHWC -> NCHW)
        if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp32u alignedLength = bufferLength & ~47;

            Rpp8u *srcPtr1Row, *srcPtr2Row, *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
            srcPtr1Row = srcPtr1Channel;
            srcPtr2Row = srcPtr2Channel;
            dstPtrRowR = dstPtrChannel;
            dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
            dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;

            for(int i = 0; i < roiPtr->xywhROI.roiHeight; i++)
            {
                Rpp8u *srcPtr1Temp, *srcPtr2Temp, *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
                srcPtr1Temp = srcPtr1Row;
                srcPtr2Temp = srcPtr2Row;
                dstPtrTempR = dstPtrRowR;
                dstPtrTempG = dstPtrRowG;
                dstPtrTempB = dstPtrRowB;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount+=48)
                {
                    __m128 p1[12], p2[12];

                    rpp_simd_load(rpp_load48_u8pkd3_to_f32pln3, srcPtr1Temp, p1);    // simd loads
                    rpp_simd_load(rpp_load48_u8pkd3_to_f32pln3, srcPtr2Temp, p2);    // simd loads
                    p1[0] = _mm_fmadd_ps(_mm_sub_ps(p1[0], p2[0]), pMul, p2[0]);    // alpha-blending adjustment
                    p1[1] = _mm_fmadd_ps(_mm_sub_ps(p1[1], p2[1]), pMul, p2[1]);    // alpha-blending adjustment
                    p1[2] = _mm_fmadd_ps(_mm_sub_ps(p1[2], p2[2]), pMul, p2[2]);    // alpha-blending adjustment
                    p1[3] = _mm_fmadd_ps(_mm_sub_ps(p1[3], p2[3]), pMul, p2[3]);    // alpha-blending adjustment
                    p1[4] = _mm_fmadd_ps(_mm_sub_ps(p1[4], p2[4]), pMul, p2[4]);    // alpha-blending adjustment
                    p1[5] = _mm_fmadd_ps(_mm_sub_ps(p1[5], p2[5]), pMul, p2[5]);    // alpha-blending adjustment
                    p1[6] = _mm_fmadd_ps(_mm_sub_ps(p1[6], p2[6]), pMul, p2[6]);    // alpha-blending adjustment
                    p1[7] = _mm_fmadd_ps(_mm_sub_ps(p1[7], p2[7]), pMul, p2[7]);    // alpha-blending adjustment
                    p1[8] = _mm_fmadd_ps(_mm_sub_ps(p1[8], p2[8]), pMul, p2[8]);    // alpha-blending adjustment
                    p1[9] = _mm_fmadd_ps(_mm_sub_ps(p1[9], p2[9]), pMul, p2[9]);    // alpha-blending adjustment
                    p1[10] = _mm_fmadd_ps(_mm_sub_ps(p1[10], p2[10]), pMul, p2[10]);    // alpha-blending adjustment
                    p1[11] = _mm_fmadd_ps(_mm_sub_ps(p1[11], p2[11]), pMul, p2[11]);    // alpha-blending adjustment
                    rpp_simd_store(rpp_store48_f32pln3_to_u8pln3, dstPtrTempR, dstPtrTempG, dstPtrTempB, p1);    // simd stores

                    srcPtr1Temp += 48;
                    srcPtr2Temp += 48;
                    dstPtrTempR += 16;
                    dstPtrTempG += 16;
                    dstPtrTempB += 16;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount+=3)
                {
                    *dstPtrTempR = (Rpp8u) RPPPIXELCHECK((((Rpp32f) (srcPtr1Temp[0]) - (Rpp32f) (srcPtr2Temp[0])) * alpha) + (Rpp32f) (srcPtr2Temp[0]));
                    *dstPtrTempG = (Rpp8u) RPPPIXELCHECK((((Rpp32f) (srcPtr1Temp[1]) - (Rpp32f) (srcPtr2Temp[1])) * alpha) + (Rpp32f) (srcPtr2Temp[1]));
                    *dstPtrTempB = (Rpp8u) RPPPIXELCHECK((((Rpp32f) (srcPtr1Temp[2]) - (Rpp32f) (srcPtr2Temp[2])) * alpha) + (Rpp32f) (srcPtr2Temp[2]));

                    srcPtr1Temp += 3;
                    srcPtr2Temp += 3;
                    dstPtrTempR++;
                    dstPtrTempG++;
                    dstPtrTempB++;
                }

                srcPtr1Row += srcDescPtr->strides.hStride;
                srcPtr2Row += srcDescPtr->strides.hStride;
                dstPtrRowR += dstDescPtr->strides.hStride;
                dstPtrRowG += dstDescPtr->strides.hStride;
                dstPtrRowB += dstDescPtr->strides.hStride;
            }
        }

        // Blend with fused output-layout toggle (NCHW -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp32u alignedLength = bufferLength & ~47;

            Rpp8u *srcPtr1RowR, *srcPtr1RowG, *srcPtr1RowB, *srcPtr2RowR, *srcPtr2RowG, *srcPtr2RowB, *dstPtrRow;
            srcPtr1RowR = srcPtr1Channel;
            srcPtr1RowG = srcPtr1RowR + srcDescPtr->strides.cStride;
            srcPtr1RowB = srcPtr1RowG + srcDescPtr->strides.cStride;
            srcPtr2RowR = srcPtr2Channel;
            srcPtr2RowG = srcPtr2RowR + srcDescPtr->strides.cStride;
            srcPtr2RowB = srcPtr2RowG + srcDescPtr->strides.cStride;
            dstPtrRow = dstPtrChannel;

            for(int i = 0; i < roiPtr->xywhROI.roiHeight; i++)
            {
                Rpp8u *srcPtr1TempR, *srcPtr1TempG, *srcPtr1TempB, *srcPtr2TempR, *srcPtr2TempG, *srcPtr2TempB, *dstPtrTemp;
                srcPtr1TempR = srcPtr1RowR;
                srcPtr1TempG = srcPtr1RowG;
                srcPtr1TempB = srcPtr1RowB;
                srcPtr2TempR = srcPtr2RowR;
                srcPtr2TempG = srcPtr2RowG;
                srcPtr2TempB = srcPtr2RowB;
                dstPtrTemp = dstPtrRow;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount+=16)
                {
                    __m128 p1[12], p2[12];

                    rpp_simd_load(rpp_load48_u8pln3_to_f32pln3, srcPtr1TempR, srcPtr1TempG, srcPtr1TempB, p1);    // simd loads
                    rpp_simd_load(rpp_load48_u8pln3_to_f32pln3, srcPtr2TempR, srcPtr2TempG, srcPtr2TempB, p2);    // simd loads
                    p1[0] = _mm_fmadd_ps(_mm_sub_ps(p1[0], p2[0]), pMul, p2[0]);    // alpha-blending adjustment
                    p1[1] = _mm_fmadd_ps(_mm_sub_ps(p1[1], p2[1]), pMul, p2[1]);    // alpha-blending adjustment
                    p1[2] = _mm_fmadd_ps(_mm_sub_ps(p1[2], p2[2]), pMul, p2[2]);    // alpha-blending adjustment
                    p1[3] = _mm_fmadd_ps(_mm_sub_ps(p1[3], p2[3]), pMul, p2[3]);    // alpha-blending adjustment
                    p1[4] = _mm_fmadd_ps(_mm_sub_ps(p1[4], p2[4]), pMul, p2[4]);    // alpha-blending adjustment
                    p1[5] = _mm_fmadd_ps(_mm_sub_ps(p1[5], p2[5]), pMul, p2[5]);    // alpha-blending adjustment
                    p1[6] = _mm_fmadd_ps(_mm_sub_ps(p1[6], p2[6]), pMul, p2[6]);    // alpha-blending adjustment
                    p1[7] = _mm_fmadd_ps(_mm_sub_ps(p1[7], p2[7]), pMul, p2[7]);    // alpha-blending adjustment
                    p1[8] = _mm_fmadd_ps(_mm_sub_ps(p1[8], p2[8]), pMul, p2[8]);    // alpha-blending adjustment
                    p1[9] = _mm_fmadd_ps(_mm_sub_ps(p1[9], p2[9]), pMul, p2[9]);    // alpha-blending adjustment
                    p1[10] = _mm_fmadd_ps(_mm_sub_ps(p1[10], p2[10]), pMul, p2[10]);    // alpha-blending adjustment
                    p1[11] = _mm_fmadd_ps(_mm_sub_ps(p1[11], p2[11]), pMul, p2[11]);    // alpha-blending adjustment
                    rpp_simd_store(rpp_store48_f32pln3_to_u8pkd3, dstPtrTemp, p1);    // simd stores

                    srcPtr1TempR += 16;
                    srcPtr1TempG += 16;
                    srcPtr1TempB += 16;
                    srcPtr2TempR += 16;
                    srcPtr2TempG += 16;
                    srcPtr2TempB += 16;
                    dstPtrTemp += 48;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    dstPtrTemp[0] = (Rpp8u) RPPPIXELCHECK((((Rpp32f) (*srcPtr1TempR) - (Rpp32f) (*srcPtr2TempR)) * alpha) + (Rpp32f) (*srcPtr2TempR));
                    dstPtrTemp[1] = (Rpp8u) RPPPIXELCHECK((((Rpp32f) (*srcPtr1TempG) - (Rpp32f) (*srcPtr2TempG)) * alpha) + (Rpp32f) (*srcPtr2TempG));
                    dstPtrTemp[2] = (Rpp8u) RPPPIXELCHECK((((Rpp32f) (*srcPtr1TempB) - (Rpp32f) (*srcPtr2TempB)) * alpha) + (Rpp32f) (*srcPtr2TempB));

                    srcPtr1TempR++;
                    srcPtr2TempR++;
                    srcPtr1TempG++;
                    srcPtr2TempG++;
                    srcPtr1TempB++;
                    srcPtr2TempB++;
                    dstPtrTemp += 3;
                }

                srcPtr1RowR += srcDescPtr->strides.hStride;
                srcPtr1RowG += srcDescPtr->strides.hStride;
                srcPtr1RowB += srcDescPtr->strides.hStride;
                srcPtr2RowR += srcDescPtr->strides.hStride;
                srcPtr2RowG += srcDescPtr->strides.hStride;
                srcPtr2RowB += srcDescPtr->strides.hStride;
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }

        // Blend without fused output-layout toggle (NHWC -> NHWC or NCHW -> NCHW)
        else
        {
            Rpp32u alignedLength = bufferLength & ~15;

            for(int c = 0; c < layoutParams.channelParam; c++)
            {
                Rpp8u *srcPtr1Row, *srcPtr2Row, *dstPtrRow;
                srcPtr1Row = srcPtr1Channel;
                srcPtr2Row = srcPtr2Channel;
                dstPtrRow = dstPtrChannel;

                for(int i = 0; i < roiPtr->xywhROI.roiHeight; i++)
                {
                    Rpp8u *srcPtr1Temp, *srcPtr2Temp, *dstPtrTemp;
                    srcPtr1Temp = srcPtr1Row;
                    srcPtr2Temp = srcPtr2Row;
                    dstPtrTemp = dstPtrRow;

                    int vectorLoopCount = 0;
                    for (; vectorLoopCount < alignedLength; vectorLoopCount+=16)
                    {
                        __m128 p1[4], p2[4];

                        rpp_simd_load(rpp_load16_u8_to_f32, srcPtr1Temp, p1);    // simd loads
                        rpp_simd_load(rpp_load16_u8_to_f32, srcPtr2Temp, p2);    // simd loads
                        p1[0] = _mm_fmadd_ps(_mm_sub_ps(p1[0], p2[0]), pMul, p2[0]);    // alpha-blending adjustment
                        p1[1] = _mm_fmadd_ps(_mm_sub_ps(p1[1], p2[1]), pMul, p2[1]);    // alpha-blending adjustment
                        p1[2] = _mm_fmadd_ps(_mm_sub_ps(p1[2], p2[2]), pMul, p2[2]);    // alpha-blending adjustment
                        p1[3] = _mm_fmadd_ps(_mm_sub_ps(p1[3], p2[3]), pMul, p2[3]);    // alpha-blending adjustment
                        rpp_simd_store(rpp_store16_f32_to_u8, dstPtrTemp, p1);    // simd stores

                        srcPtr1Temp +=16;
                        srcPtr2Temp +=16;
                        dstPtrTemp +=16;
                    }
                    for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                    {
                        *dstPtrTemp = (Rpp8u) RPPPIXELCHECK((((Rpp32f) (*srcPtr1Temp) - (Rpp32f) (*srcPtr2Temp)) * alpha) + (Rpp32f) (*srcPtr2Temp));

                        srcPtr1Temp++;
                        srcPtr2Temp++;
                        dstPtrTemp++;
                    }

                    srcPtr1Row += srcDescPtr->strides.hStride;
                    srcPtr2Row += srcDescPtr->strides.hStride;
                    dstPtrRow += dstDescPtr->strides.hStride;
                }

                srcPtr1Channel += srcDescPtr->strides.cStride;
                srcPtr2Channel += srcDescPtr->strides.cStride;
                dstPtrChannel += dstDescPtr->strides.cStride;
            }
        }
    }

    return RPP_SUCCESS;
}

RppStatus blend_f32_f32_host_tensor(Rpp32f *srcPtr1,
                                    Rpp32f *srcPtr2,
                                    RpptDescPtr srcDescPtr,
                                    Rpp32f *dstPtr,
                                    RpptDescPtr dstDescPtr,
                                    Rpp32f *alphaTensor,
                                    RpptROIPtr roiTensorPtrSrc,
                                    RpptRoiType roiType,
                                    RppLayoutParams layoutParams)
{
    RpptROI roiDefault;
    RpptROIPtr roiPtrDefault;
    roiPtrDefault = &roiDefault;
    roiPtrDefault->xywhROI.xy.x = 0;
    roiPtrDefault->xywhROI.xy.y = 0;
    roiPtrDefault->xywhROI.roiWidth = srcDescPtr->w;
    roiPtrDefault->xywhROI.roiHeight = srcDescPtr->h;

    omp_set_dynamic(0);
#pragma omp parallel for num_threads(dstDescPtr->n)
    for(int batchCount = 0; batchCount < dstDescPtr->n; batchCount++)
    {
        RpptROI roi;
        RpptROIPtr roiPtr;

        if (&roiTensorPtrSrc[batchCount] == NULL)
        {
            roiPtr = roiPtrDefault;
        }
        else
        {
            RpptROIPtr roiPtrInput = &roiTensorPtrSrc[batchCount];

            RpptROI roiImage;
            RpptROIPtr roiPtrImage;

            if (roiType == RpptRoiType::LTRB)
            {
                roiPtrImage = &roiImage;
                compute_xywh_from_ltrb_host(roiPtrInput, roiPtrImage);
            }
            else if (roiType == RpptRoiType::XYWH)
            {
                roiPtrImage = roiPtrInput;
            }

            roiPtr = &roi;
            compute_roi_boundary_check_host(roiPtrImage, roiPtr, roiPtrDefault);
        }

        Rpp32f alpha = alphaTensor[batchCount];

        Rpp32f *srcPtr1Image, *srcPtr2Image, *dstPtrImage;
        srcPtr1Image = srcPtr1 + batchCount * srcDescPtr->strides.nStride;
        srcPtr2Image = srcPtr2 + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;

        Rpp32u bufferLength = roiPtr->xywhROI.roiWidth * layoutParams.bufferMultiplier;

        __m128 pMul = _mm_set1_ps(alpha);

        Rpp32f *srcPtr1Channel, *srcPtr2Channel, *dstPtrChannel;
        srcPtr1Channel = srcPtr1Image + (roiPtr->xywhROI.xy.y * srcDescPtr->strides.hStride) + (roiPtr->xywhROI.xy.x * layoutParams.bufferMultiplier);
        srcPtr2Channel = srcPtr2Image + (roiPtr->xywhROI.xy.y * srcDescPtr->strides.hStride) + (roiPtr->xywhROI.xy.x * layoutParams.bufferMultiplier);
        dstPtrChannel = dstPtrImage;

        // Blend with fused output-layout toggle (NHWC -> NCHW)
        if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp32u alignedLength = bufferLength & ~11;

            Rpp32f *srcPtr1Row, *srcPtr2Row, *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
            srcPtr1Row = srcPtr1Channel;
            srcPtr2Row = srcPtr2Channel;
            dstPtrRowR = dstPtrChannel;
            dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
            dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;

            for(int i = 0; i < roiPtr->xywhROI.roiHeight; i++)
            {
                Rpp32f *srcPtr1Temp, *srcPtr2Temp, *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
                srcPtr1Temp = srcPtr1Row;
                srcPtr2Temp = srcPtr2Row;
                dstPtrTempR = dstPtrRowR;
                dstPtrTempG = dstPtrRowG;
                dstPtrTempB = dstPtrRowB;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount+=12)
                {
                    __m128 p1[4], p2[4];

                    rpp_simd_load(rpp_load12_f32pkd3_to_f32pln3, srcPtr1Temp, p1);    // simd loads
                    rpp_simd_load(rpp_load12_f32pkd3_to_f32pln3, srcPtr2Temp, p2);    // simd loads
                    p1[0] = _mm_fmadd_ps(_mm_sub_ps(p1[0], p2[0]), pMul, p2[0]);    // alpha-blending adjustment
                    p1[1] = _mm_fmadd_ps(_mm_sub_ps(p1[1], p2[1]), pMul, p2[1]);    // alpha-blending adjustment
                    p1[2] = _mm_fmadd_ps(_mm_sub_ps(p1[2], p2[2]), pMul, p2[2]);    // alpha-blending adjustment
                    rpp_simd_store(rpp_store12_f32pln3_to_f32pln3, dstPtrTempR, dstPtrTempG, dstPtrTempB, p1);    // simd stores

                    srcPtr1Temp += 12;
                    srcPtr2Temp += 12;
                    dstPtrTempR += 4;
                    dstPtrTempG += 4;
                    dstPtrTempB += 4;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount+=3)
                {
                    *dstPtrTempR = RPPPIXELCHECKF32((srcPtr1Temp[0] - srcPtr2Temp[0]) * alpha + srcPtr2Temp[0]);
                    *dstPtrTempG = RPPPIXELCHECKF32((srcPtr1Temp[1] - srcPtr2Temp[1]) * alpha + srcPtr2Temp[1]);
                    *dstPtrTempB = RPPPIXELCHECKF32((srcPtr1Temp[2] - srcPtr2Temp[2]) * alpha + srcPtr2Temp[2]);

                    dstPtrTempR++;
                    dstPtrTempG++;
                    dstPtrTempB++;
                    srcPtr1Temp += 3;
                    srcPtr2Temp += 3;
                }

                srcPtr1Row += srcDescPtr->strides.hStride;
                srcPtr2Row += srcDescPtr->strides.hStride;
                dstPtrRowR += dstDescPtr->strides.hStride;
                dstPtrRowG += dstDescPtr->strides.hStride;
                dstPtrRowB += dstDescPtr->strides.hStride;
            }
        }

        // Blend with fused output-layout toggle (NCHW -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp32u alignedLength = bufferLength & ~11;

            Rpp32f *srcPtr1RowR, *srcPtr1RowG, *srcPtr1RowB, *srcPtr2RowR, *srcPtr2RowG, *srcPtr2RowB, *dstPtrRow;
            srcPtr1RowR = srcPtr1Channel;
            srcPtr1RowG = srcPtr1RowR + srcDescPtr->strides.cStride;
            srcPtr1RowB = srcPtr1RowG + srcDescPtr->strides.cStride;
            srcPtr2RowR = srcPtr2Channel;
            srcPtr2RowG = srcPtr2RowR + srcDescPtr->strides.cStride;
            srcPtr2RowB = srcPtr2RowG + srcDescPtr->strides.cStride;
            dstPtrRow = dstPtrChannel;

            for(int i = 0; i < roiPtr->xywhROI.roiHeight; i++)
            {
                Rpp32f *srcPtr1TempR, *srcPtr1TempG, *srcPtr1TempB, *srcPtr2TempR, *srcPtr2TempG, *srcPtr2TempB, *dstPtrTemp;
                srcPtr1TempR = srcPtr1RowR;
                srcPtr1TempG = srcPtr1RowG;
                srcPtr1TempB = srcPtr1RowB;
                srcPtr2TempR = srcPtr2RowR;
                srcPtr2TempG = srcPtr2RowG;
                srcPtr2TempB = srcPtr2RowB;
                dstPtrTemp = dstPtrRow;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount+=4)
                {
                    __m128 p1[4], p2[4];

                    rpp_simd_load(rpp_load12_f32pln3_to_f32pln3, srcPtr1TempR, srcPtr1TempG, srcPtr1TempB, p1);    // simd loads
                    rpp_simd_load(rpp_load12_f32pln3_to_f32pln3, srcPtr2TempR, srcPtr2TempG, srcPtr2TempB, p2);    // simd loads
                    p1[0] = _mm_fmadd_ps(_mm_sub_ps(p1[0], p2[0]), pMul, p2[0]);    // alpha-blending adjustment
                    p1[1] = _mm_fmadd_ps(_mm_sub_ps(p1[1], p2[1]), pMul, p2[1]);    // alpha-blending adjustment
                    p1[2] = _mm_fmadd_ps(_mm_sub_ps(p1[2], p2[2]), pMul, p2[2]);    // alpha-blending adjustment
                    rpp_simd_store(rpp_store12_f32pln3_to_f32pkd3, dstPtrTemp, p1);    // simd stores

                    srcPtr1TempR += 4;
                    srcPtr1TempG += 4;
                    srcPtr1TempB += 4;
                    srcPtr2TempR += 4;
                    srcPtr2TempG += 4;
                    srcPtr2TempB += 4;
                    dstPtrTemp += 12;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    dstPtrTemp[0] = RPPPIXELCHECKF32((*srcPtr1TempR - *srcPtr2TempR) * alpha + *srcPtr2TempR);
                    dstPtrTemp[1] = RPPPIXELCHECKF32((*srcPtr1TempG - *srcPtr2TempG) * alpha + *srcPtr2TempG);
                    dstPtrTemp[2] = RPPPIXELCHECKF32((*srcPtr1TempB - *srcPtr2TempB) * alpha + *srcPtr2TempB);

                    srcPtr1TempR++;
                    srcPtr2TempR++;
                    srcPtr1TempG++;
                    srcPtr2TempG++;
                    srcPtr1TempB++;
                    srcPtr2TempB++;
                    dstPtrTemp += 3;
                }

                srcPtr1RowR += srcDescPtr->strides.hStride;
                srcPtr1RowG += srcDescPtr->strides.hStride;
                srcPtr1RowB += srcDescPtr->strides.hStride;
                srcPtr2RowR += srcDescPtr->strides.hStride;
                srcPtr2RowG += srcDescPtr->strides.hStride;
                srcPtr2RowB += srcDescPtr->strides.hStride;
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }

        // Blend without fused output-layout toggle (NHWC -> NHWC or NCHW -> NCHW)
        else
        {
            Rpp32u alignedLength = bufferLength & ~3;

            for(int c = 0; c < layoutParams.channelParam; c++)
            {
                Rpp32f *srcPtr1Row, *srcPtr2Row, *dstPtrRow;
                srcPtr1Row = srcPtr1Channel;
                srcPtr2Row = srcPtr2Channel;
                dstPtrRow = dstPtrChannel;

                for(int i = 0; i < roiPtr->xywhROI.roiHeight; i++)
                {
                    Rpp32f *srcPtr1Temp, *srcPtr2Temp, *dstPtrTemp;
                    srcPtr1Temp = srcPtr1Row;
                    srcPtr2Temp = srcPtr2Row;
                    dstPtrTemp = dstPtrRow;

                    int vectorLoopCount = 0;
                    for (; vectorLoopCount < alignedLength; vectorLoopCount+=4)
                    {
                        __m128 p1[1], p2[1];

                        rpp_simd_load(rpp_load4_f32_to_f32, srcPtr1Temp, p1);    // simd loads
                        rpp_simd_load(rpp_load4_f32_to_f32, srcPtr2Temp, p2);    // simd loads
                        p1[0] = _mm_fmadd_ps(_mm_sub_ps(p1[0], p2[0]), pMul, p2[0]);    // alpha-blending adjustment
                        rpp_simd_store(rpp_store4_f32_to_f32, dstPtrTemp, p1);    // simd stores

                        srcPtr1Temp += 4;
                        srcPtr2Temp += 4;
                        dstPtrTemp += 4;
                    }
                    for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                    {
                        *dstPtrTemp = RPPPIXELCHECKF32((*srcPtr1Temp - *srcPtr2Temp) * alpha + *srcPtr2Temp);

                        srcPtr1Temp++;
                        srcPtr2Temp++;
                        dstPtrTemp++;
                    }

                    srcPtr1Row += srcDescPtr->strides.hStride;
                    srcPtr2Row += srcDescPtr->strides.hStride;
                    dstPtrRow += dstDescPtr->strides.hStride;
                }

                srcPtr1Channel += srcDescPtr->strides.cStride;
                srcPtr2Channel += srcDescPtr->strides.cStride;
                dstPtrChannel += dstDescPtr->strides.cStride;
            }
        }
    }

    return RPP_SUCCESS;
}

RppStatus blend_f16_f16_host_tensor(Rpp16f *srcPtr1,
                                    Rpp16f *srcPtr2,
                                    RpptDescPtr srcDescPtr,
                                    Rpp16f *dstPtr,
                                    RpptDescPtr dstDescPtr,
                                    Rpp32f *alphaTensor,
                                    RpptROIPtr roiTensorPtrSrc,
                                    RpptRoiType roiType,
                                    RppLayoutParams layoutParams)
{
    RpptROI roiDefault;
    RpptROIPtr roiPtrDefault;
    roiPtrDefault = &roiDefault;
    roiPtrDefault->xywhROI.xy.x = 0;
    roiPtrDefault->xywhROI.xy.y = 0;
    roiPtrDefault->xywhROI.roiWidth = srcDescPtr->w;
    roiPtrDefault->xywhROI.roiHeight = srcDescPtr->h;

    omp_set_dynamic(0);
#pragma omp parallel for num_threads(dstDescPtr->n)
    for(int batchCount = 0; batchCount < dstDescPtr->n; batchCount++)
    {
        RpptROI roi;
        RpptROIPtr roiPtr;

        if (&roiTensorPtrSrc[batchCount] == NULL)
        {
            roiPtr = roiPtrDefault;
        }
        else
        {
            RpptROIPtr roiPtrInput = &roiTensorPtrSrc[batchCount];

            RpptROI roiImage;
            RpptROIPtr roiPtrImage;

            if (roiType == RpptRoiType::LTRB)
            {
                roiPtrImage = &roiImage;
                compute_xywh_from_ltrb_host(roiPtrInput, roiPtrImage);
            }
            else if (roiType == RpptRoiType::XYWH)
            {
                roiPtrImage = roiPtrInput;
            }

            roiPtr = &roi;
            compute_roi_boundary_check_host(roiPtrImage, roiPtr, roiPtrDefault);
        }

        Rpp32f alpha = alphaTensor[batchCount];

        Rpp16f *srcPtr1Image, *srcPtr2Image, *dstPtrImage;
        srcPtr1Image = srcPtr1 + batchCount * srcDescPtr->strides.nStride;
        srcPtr2Image = srcPtr2 + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;

        Rpp32u bufferLength = roiPtr->xywhROI.roiWidth * layoutParams.bufferMultiplier;

        __m128 pMul = _mm_set1_ps(alpha);

        Rpp16f *srcPtr1Channel, *srcPtr2Channel, *dstPtrChannel;
        srcPtr1Channel = srcPtr1Image + (roiPtr->xywhROI.xy.y * srcDescPtr->strides.hStride) + (roiPtr->xywhROI.xy.x * layoutParams.bufferMultiplier);
        srcPtr2Channel = srcPtr2Image + (roiPtr->xywhROI.xy.y * srcDescPtr->strides.hStride) + (roiPtr->xywhROI.xy.x * layoutParams.bufferMultiplier);
        dstPtrChannel = dstPtrImage;

        // Blend with fused output-layout toggle (NHWC -> NCHW)
        if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp32u alignedLength = bufferLength & ~11;

            Rpp16f *srcPtr1Row, *srcPtr2Row, *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
            srcPtr1Row = srcPtr1Channel;
            srcPtr2Row = srcPtr2Channel;
            dstPtrRowR = dstPtrChannel;
            dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
            dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;

            for(int i = 0; i < roiPtr->xywhROI.roiHeight; i++)
            {
                Rpp16f *srcPtr1Temp, *srcPtr2Temp, *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
                srcPtr1Temp = srcPtr1Row;
                srcPtr2Temp = srcPtr2Row;
                dstPtrTempR = dstPtrRowR;
                dstPtrTempG = dstPtrRowG;
                dstPtrTempB = dstPtrRowB;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount+=12)
                {
                    Rpp32f srcPtr1Temp_ps[12], srcPtr2Temp_ps[12], dstPtrTemp_ps[12];

                    for(int cnt = 0; cnt < 12; cnt++)
                    {
                        *(srcPtr1Temp_ps + cnt) = (Rpp32f) *(srcPtr1Temp + cnt);
                        *(srcPtr2Temp_ps + cnt) = (Rpp32f) *(srcPtr2Temp + cnt);
                    }

                    __m128 p1[4], p2[4];

                    rpp_simd_load(rpp_load12_f32pkd3_to_f32pln3, srcPtr1Temp_ps, p1);    // simd loads
                    rpp_simd_load(rpp_load12_f32pkd3_to_f32pln3, srcPtr2Temp_ps, p2);    // simd loads
                    p1[0] = _mm_fmadd_ps(_mm_sub_ps(p1[0], p2[0]), pMul, p2[0]);    // alpha-blending adjustment
                    p1[1] = _mm_fmadd_ps(_mm_sub_ps(p1[1], p2[1]), pMul, p2[1]);    // alpha-blending adjustment
                    p1[2] = _mm_fmadd_ps(_mm_sub_ps(p1[2], p2[2]), pMul, p2[2]);    // alpha-blending adjustment
                    rpp_simd_store(rpp_store12_f32pln3_to_f32pln3, dstPtrTemp_ps, dstPtrTemp_ps + 4, dstPtrTemp_ps + 8, p1);    // simd stores

                    for(int cnt = 0; cnt < 4; cnt++)
                    {
                        *(dstPtrTempR + cnt) = (Rpp16f) *(dstPtrTemp_ps + cnt);
                        *(dstPtrTempG + cnt) = (Rpp16f) *(dstPtrTemp_ps + 4 + cnt);
                        *(dstPtrTempB + cnt) = (Rpp16f) *(dstPtrTemp_ps + 8 + cnt);
                    }

                    srcPtr1Temp += 12;
                    srcPtr2Temp += 12;
                    dstPtrTempR += 4;
                    dstPtrTempG += 4;
                    dstPtrTempB += 4;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount+=3)
                {
                    *dstPtrTempR = (Rpp16f) RPPPIXELCHECKF32((srcPtr1Temp[0] - srcPtr2Temp[0]) * alpha + srcPtr2Temp[0]);
                    *dstPtrTempG = (Rpp16f) RPPPIXELCHECKF32((srcPtr1Temp[1] - srcPtr2Temp[1]) * alpha + srcPtr2Temp[1]);
                    *dstPtrTempB = (Rpp16f) RPPPIXELCHECKF32((srcPtr1Temp[2] - srcPtr2Temp[2]) * alpha + srcPtr2Temp[2]);

                    dstPtrTempR++;
                    dstPtrTempG++;
                    dstPtrTempB++;
                    srcPtr1Temp += 3;
                    srcPtr2Temp += 3;
                }

                srcPtr1Row += srcDescPtr->strides.hStride;
                srcPtr2Row += srcDescPtr->strides.hStride;
                dstPtrRowR += dstDescPtr->strides.hStride;
                dstPtrRowG += dstDescPtr->strides.hStride;
                dstPtrRowB += dstDescPtr->strides.hStride;
            }
        }

        // Blend with fused output-layout toggle (NCHW -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp32u alignedLength = bufferLength & ~11;

            Rpp16f *srcPtr1RowR, *srcPtr1RowG, *srcPtr1RowB, *srcPtr2RowR, *srcPtr2RowG, *srcPtr2RowB, *dstPtrRow;
            srcPtr1RowR = srcPtr1Channel;
            srcPtr1RowG = srcPtr1RowR + srcDescPtr->strides.cStride;
            srcPtr1RowB = srcPtr1RowG + srcDescPtr->strides.cStride;
            srcPtr2RowR = srcPtr2Channel;
            srcPtr2RowG = srcPtr2RowR + srcDescPtr->strides.cStride;
            srcPtr2RowB = srcPtr2RowG + srcDescPtr->strides.cStride;
            dstPtrRow = dstPtrChannel;

            for(int i = 0; i < roiPtr->xywhROI.roiHeight; i++)
            {
                Rpp16f *srcPtr1TempR, *srcPtr1TempG, *srcPtr1TempB, *srcPtr2TempR, *srcPtr2TempG, *srcPtr2TempB, *dstPtrTemp;
                srcPtr1TempR = srcPtr1RowR;
                srcPtr1TempG = srcPtr1RowG;
                srcPtr1TempB = srcPtr1RowB;
                srcPtr2TempR = srcPtr2RowR;
                srcPtr2TempG = srcPtr2RowG;
                srcPtr2TempB = srcPtr2RowB;
                dstPtrTemp = dstPtrRow;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount+=4)
                {
                    Rpp32f srcPtr1Temp_ps[12], srcPtr2Temp_ps[12], dstPtrTemp_ps[13];

                    for(int cnt = 0; cnt < 4; cnt++)
                    {
                        *(srcPtr1Temp_ps + cnt) = (Rpp32f) *(srcPtr1TempR + cnt);
                        *(srcPtr1Temp_ps + 4 + cnt) = (Rpp32f) *(srcPtr1TempG + cnt);
                        *(srcPtr1Temp_ps + 8 + cnt) = (Rpp32f) *(srcPtr1TempB + cnt);

                        *(srcPtr2Temp_ps + cnt) = (Rpp32f) *(srcPtr2TempR + cnt);
                        *(srcPtr2Temp_ps + 4 + cnt) = (Rpp32f) *(srcPtr2TempG + cnt);
                        *(srcPtr2Temp_ps + 8 + cnt) = (Rpp32f) *(srcPtr2TempB + cnt);
                    }

                    __m128 p1[4], p2[4];

                    rpp_simd_load(rpp_load12_f32pln3_to_f32pln3, srcPtr1Temp_ps, srcPtr1Temp_ps + 4, srcPtr1Temp_ps + 8, p1);    // simd loads
                    rpp_simd_load(rpp_load12_f32pln3_to_f32pln3, srcPtr2Temp_ps, srcPtr2Temp_ps + 4, srcPtr2Temp_ps + 8, p2);    // simd loads
                    p1[0] = _mm_fmadd_ps(_mm_sub_ps(p1[0], p2[0]), pMul, p2[0]);    // alpha-blending adjustment
                    p1[1] = _mm_fmadd_ps(_mm_sub_ps(p1[1], p2[1]), pMul, p2[1]);    // alpha-blending adjustment
                    p1[2] = _mm_fmadd_ps(_mm_sub_ps(p1[2], p2[2]), pMul, p2[2]);    // alpha-blending adjustment
                    rpp_simd_store(rpp_store12_f32pln3_to_f32pkd3, dstPtrTemp_ps, p1);    // simd stores

                    for(int cnt = 0; cnt < 12; cnt++)
                    {
                        *(dstPtrTemp + cnt) = (Rpp16f) *(dstPtrTemp_ps + cnt);
                    }

                    srcPtr1TempR += 4;
                    srcPtr1TempG += 4;
                    srcPtr1TempB += 4;
                    srcPtr2TempR += 4;
                    srcPtr2TempG += 4;
                    srcPtr2TempB += 4;
                    dstPtrTemp += 12;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    dstPtrTemp[0] = (Rpp16f) RPPPIXELCHECKF32((*srcPtr1TempR - *srcPtr2TempR) * alpha + *srcPtr2TempR);
                    dstPtrTemp[1] = (Rpp16f) RPPPIXELCHECKF32((*srcPtr1TempG - *srcPtr2TempG) * alpha + *srcPtr2TempG);
                    dstPtrTemp[2] = (Rpp16f) RPPPIXELCHECKF32((*srcPtr1TempB - *srcPtr2TempB) * alpha + *srcPtr2TempB);

                    srcPtr1TempR++;
                    srcPtr2TempR++;
                    srcPtr1TempG++;
                    srcPtr2TempG++;
                    srcPtr1TempB++;
                    srcPtr2TempB++;
                    dstPtrTemp += 3;
                }

                srcPtr1RowR += srcDescPtr->strides.hStride;
                srcPtr1RowG += srcDescPtr->strides.hStride;
                srcPtr1RowB += srcDescPtr->strides.hStride;
                srcPtr2RowR += srcDescPtr->strides.hStride;
                srcPtr2RowG += srcDescPtr->strides.hStride;
                srcPtr2RowB += srcDescPtr->strides.hStride;
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }

        // Blend without fused output-layout toggle (NHWC -> NHWC or NCHW -> NCHW)
        else
        {
            Rpp32u alignedLength = bufferLength & ~3;

            for(int c = 0; c < layoutParams.channelParam; c++)
            {
                Rpp16f *srcPtr1Row, *srcPtr2Row, *dstPtrRow;
                srcPtr1Row = srcPtr1Channel;
                srcPtr2Row = srcPtr2Channel;
                dstPtrRow = dstPtrChannel;

                for(int i = 0; i < roiPtr->xywhROI.roiHeight; i++)
                {
                    Rpp16f *srcPtr1Temp, *srcPtr2Temp, *dstPtrTemp;
                    srcPtr1Temp = srcPtr1Row;
                    srcPtr2Temp = srcPtr2Row;
                    dstPtrTemp = dstPtrRow;

                    int vectorLoopCount = 0;
                    for (; vectorLoopCount < alignedLength; vectorLoopCount+=4)
                    {
                        Rpp32f srcPtr1Temp_ps[4], srcPtr2Temp_ps[4], dstPtrTemp_ps[4];

                        for(int cnt = 0; cnt < 4; cnt++)
                        {
                            *(srcPtr1Temp_ps + cnt) = (Rpp16f) *(srcPtr1Temp + cnt);
                            *(srcPtr2Temp_ps + cnt) = (Rpp16f) *(srcPtr2Temp + cnt);
                        }

                        __m128 p1[1], p2[1];

                        rpp_simd_load(rpp_load4_f32_to_f32, srcPtr1Temp_ps, p1);    // simd loads
                        rpp_simd_load(rpp_load4_f32_to_f32, srcPtr2Temp_ps, p2);    // simd loads
                        p1[0] = _mm_fmadd_ps(_mm_sub_ps(p1[0], p2[0]), pMul, p2[0]);    // alpha-blending adjustment
                        rpp_simd_store(rpp_store4_f32_to_f32, dstPtrTemp_ps, p1);    // simd stores

                        for(int cnt = 0; cnt < 4; cnt++)
                        {
                            *(dstPtrTemp + cnt) = (Rpp16f) *(dstPtrTemp_ps + cnt);
                        }

                        srcPtr1Temp += 4;
                        srcPtr2Temp += 4;
                        dstPtrTemp += 4;
                    }
                    for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                    {
                        *dstPtrTemp = (Rpp16f) RPPPIXELCHECKF32(((Rpp32f)*srcPtr1Temp - (Rpp32f)*srcPtr2Temp) * alpha + (Rpp32f)*srcPtr2Temp);

                        srcPtr1Temp++;
                        srcPtr2Temp++;
                        dstPtrTemp++;
                    }

                    srcPtr1Row += srcDescPtr->strides.hStride;
                    srcPtr2Row += srcDescPtr->strides.hStride;
                    dstPtrRow += dstDescPtr->strides.hStride;
                }

                srcPtr1Channel += srcDescPtr->strides.cStride;
                srcPtr2Channel += srcDescPtr->strides.cStride;
                dstPtrChannel += dstDescPtr->strides.cStride;
            }
        }
    }

    return RPP_SUCCESS;
}

RppStatus blend_i8_i8_host_tensor(Rpp8s *srcPtr1,
                                  Rpp8s *srcPtr2,
                                  RpptDescPtr srcDescPtr,
                                  Rpp8s *dstPtr,
                                  RpptDescPtr dstDescPtr,
                                  Rpp32f *alphaTensor,
                                  RpptROIPtr roiTensorPtrSrc,
                                  RpptRoiType roiType,
                                  RppLayoutParams layoutParams)
{
    RpptROI roiDefault;
    RpptROIPtr roiPtrDefault;
    roiPtrDefault = &roiDefault;
    roiPtrDefault->xywhROI.xy.x = 0;
    roiPtrDefault->xywhROI.xy.y = 0;
    roiPtrDefault->xywhROI.roiWidth = srcDescPtr->w;
    roiPtrDefault->xywhROI.roiHeight = srcDescPtr->h;

    omp_set_dynamic(0);
#pragma omp parallel for num_threads(dstDescPtr->n)
    for(int batchCount = 0; batchCount < dstDescPtr->n; batchCount++)
    {
        RpptROI roi;
        RpptROIPtr roiPtr;

        if (&roiTensorPtrSrc[batchCount] == NULL)
        {
            roiPtr = roiPtrDefault;
        }
        else
        {
            RpptROIPtr roiPtrInput = &roiTensorPtrSrc[batchCount];

            RpptROI roiImage;
            RpptROIPtr roiPtrImage;

            if (roiType == RpptRoiType::LTRB)
            {
                roiPtrImage = &roiImage;
                compute_xywh_from_ltrb_host(roiPtrInput, roiPtrImage);
            }
            else if (roiType == RpptRoiType::XYWH)
            {
                roiPtrImage = roiPtrInput;
            }

            roiPtr = &roi;
            compute_roi_boundary_check_host(roiPtrImage, roiPtr, roiPtrDefault);
        }

        Rpp32f alpha = alphaTensor[batchCount];

        Rpp8s *srcPtr1Image, *srcPtr2Image, *dstPtrImage;
        srcPtr1Image = srcPtr1 + batchCount * srcDescPtr->strides.nStride;
        srcPtr2Image = srcPtr2 + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;

        Rpp32u bufferLength = roiPtr->xywhROI.roiWidth * layoutParams.bufferMultiplier;

        __m128 pMul = _mm_set1_ps(alpha);

        Rpp8s *srcPtr1Channel, *srcPtr2Channel, *dstPtrChannel;
        srcPtr1Channel = srcPtr1Image + (roiPtr->xywhROI.xy.y * srcDescPtr->strides.hStride) + (roiPtr->xywhROI.xy.x * layoutParams.bufferMultiplier);
        srcPtr2Channel = srcPtr2Image + (roiPtr->xywhROI.xy.y * srcDescPtr->strides.hStride) + (roiPtr->xywhROI.xy.x * layoutParams.bufferMultiplier);
        dstPtrChannel = dstPtrImage;

        // Blend with fused output-layout toggle (NHWC -> NCHW)
        if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp32u alignedLength = bufferLength & ~47;

            Rpp8s *srcPtr1Row, *srcPtr2Row, *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
            srcPtr1Row = srcPtr1Channel;
            srcPtr2Row = srcPtr2Channel;
            dstPtrRowR = dstPtrChannel;
            dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
            dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;

            for(int i = 0; i < roiPtr->xywhROI.roiHeight; i++)
            {
                Rpp8s *srcPtr1Temp, *srcPtr2Temp, *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
                srcPtr1Temp = srcPtr1Row;
                srcPtr2Temp = srcPtr2Row;
                dstPtrTempR = dstPtrRowR;
                dstPtrTempG = dstPtrRowG;
                dstPtrTempB = dstPtrRowB;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount+=48)
                {
                    __m128 p1[12], p2[12];

                    rpp_simd_load(rpp_load48_i8pkd3_to_f32pln3, srcPtr1Temp, p1);    // simd loads
                    rpp_simd_load(rpp_load48_i8pkd3_to_f32pln3, srcPtr2Temp, p2);    // simd loads
                    p1[0] = _mm_fmadd_ps(_mm_sub_ps(p1[0], p2[0]), pMul, p2[0]);    // alpha-blending adjustment
                    p1[1] = _mm_fmadd_ps(_mm_sub_ps(p1[1], p2[1]), pMul, p2[1]);    // alpha-blending adjustment
                    p1[2] = _mm_fmadd_ps(_mm_sub_ps(p1[2], p2[2]), pMul, p2[2]);    // alpha-blending adjustment
                    p1[3] = _mm_fmadd_ps(_mm_sub_ps(p1[3], p2[3]), pMul, p2[3]);    // alpha-blending adjustment
                    p1[4] = _mm_fmadd_ps(_mm_sub_ps(p1[4], p2[4]), pMul, p2[4]);    // alpha-blending adjustment
                    p1[5] = _mm_fmadd_ps(_mm_sub_ps(p1[5], p2[5]), pMul, p2[5]);    // alpha-blending adjustment
                    p1[6] = _mm_fmadd_ps(_mm_sub_ps(p1[6], p2[6]), pMul, p2[6]);    // alpha-blending adjustment
                    p1[7] = _mm_fmadd_ps(_mm_sub_ps(p1[7], p2[7]), pMul, p2[7]);    // alpha-blending adjustment
                    p1[8] = _mm_fmadd_ps(_mm_sub_ps(p1[8], p2[8]), pMul, p2[8]);    // alpha-blending adjustment
                    p1[9] = _mm_fmadd_ps(_mm_sub_ps(p1[9], p2[9]), pMul, p2[9]);    // alpha-blending adjustment
                    p1[10] = _mm_fmadd_ps(_mm_sub_ps(p1[10], p2[10]), pMul, p2[10]);    // alpha-blending adjustment
                    p1[11] = _mm_fmadd_ps(_mm_sub_ps(p1[11], p2[11]), pMul, p2[11]);    // alpha-blending adjustment
                    rpp_simd_store(rpp_store48_f32pln3_to_i8pln3, dstPtrTempR, dstPtrTempG, dstPtrTempB, p1);    // simd stores

                    srcPtr1Temp += 48;
                    srcPtr2Temp += 48;
                    dstPtrTempR += 16;
                    dstPtrTempG += 16;
                    dstPtrTempB += 16;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount+=3)
                {
                    *dstPtrTempR = (Rpp8s) RPPPIXELCHECKI8((((Rpp32f) (srcPtr1Temp[0]) - (Rpp32f) (srcPtr2Temp[0])) * alpha) + (Rpp32f) (srcPtr2Temp[0]));
                    *dstPtrTempG = (Rpp8s) RPPPIXELCHECKI8((((Rpp32f) (srcPtr1Temp[1]) - (Rpp32f) (srcPtr2Temp[1])) * alpha) + (Rpp32f) (srcPtr2Temp[1]));
                    *dstPtrTempB = (Rpp8s) RPPPIXELCHECKI8((((Rpp32f) (srcPtr1Temp[2]) - (Rpp32f) (srcPtr2Temp[2])) * alpha) + (Rpp32f) (srcPtr2Temp[2]));

                    srcPtr1Temp += 3;
                    srcPtr2Temp += 3;
                    dstPtrTempR++;
                    dstPtrTempG++;
                    dstPtrTempB++;
                }

                srcPtr1Row += srcDescPtr->strides.hStride;
                srcPtr2Row += srcDescPtr->strides.hStride;
                dstPtrRowR += dstDescPtr->strides.hStride;
                dstPtrRowG += dstDescPtr->strides.hStride;
                dstPtrRowB += dstDescPtr->strides.hStride;
            }
        }

        // Blend with fused output-layout toggle (NCHW -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp32u alignedLength = bufferLength & ~47;

            Rpp8s *srcPtr1RowR, *srcPtr1RowG, *srcPtr1RowB, *srcPtr2RowR, *srcPtr2RowG, *srcPtr2RowB, *dstPtrRow;
            srcPtr1RowR = srcPtr1Channel;
            srcPtr1RowG = srcPtr1RowR + srcDescPtr->strides.cStride;
            srcPtr1RowB = srcPtr1RowG + srcDescPtr->strides.cStride;
            srcPtr2RowR = srcPtr2Channel;
            srcPtr2RowG = srcPtr2RowR + srcDescPtr->strides.cStride;
            srcPtr2RowB = srcPtr2RowG + srcDescPtr->strides.cStride;
            dstPtrRow = dstPtrChannel;

            for(int i = 0; i < roiPtr->xywhROI.roiHeight; i++)
            {
                Rpp8s *srcPtr1TempR, *srcPtr1TempG, *srcPtr1TempB, *srcPtr2TempR, *srcPtr2TempG, *srcPtr2TempB, *dstPtrTemp;
                srcPtr1TempR = srcPtr1RowR;
                srcPtr1TempG = srcPtr1RowG;
                srcPtr1TempB = srcPtr1RowB;
                srcPtr2TempR = srcPtr2RowR;
                srcPtr2TempG = srcPtr2RowG;
                srcPtr2TempB = srcPtr2RowB;
                dstPtrTemp = dstPtrRow;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount+=16)
                {
                    __m128 p1[12], p2[12];

                    rpp_simd_load(rpp_load48_i8pln3_to_f32pln3, srcPtr1TempR, srcPtr1TempG, srcPtr1TempB, p1);    // simd loads
                    rpp_simd_load(rpp_load48_i8pln3_to_f32pln3, srcPtr2TempR, srcPtr2TempG, srcPtr2TempB, p2);    // simd loads
                    p1[0] = _mm_fmadd_ps(_mm_sub_ps(p1[0], p2[0]), pMul, p2[0]);    // alpha-blending adjustment
                    p1[1] = _mm_fmadd_ps(_mm_sub_ps(p1[1], p2[1]), pMul, p2[1]);    // alpha-blending adjustment
                    p1[2] = _mm_fmadd_ps(_mm_sub_ps(p1[2], p2[2]), pMul, p2[2]);    // alpha-blending adjustment
                    p1[3] = _mm_fmadd_ps(_mm_sub_ps(p1[3], p2[3]), pMul, p2[3]);    // alpha-blending adjustment
                    p1[4] = _mm_fmadd_ps(_mm_sub_ps(p1[4], p2[4]), pMul, p2[4]);    // alpha-blending adjustment
                    p1[5] = _mm_fmadd_ps(_mm_sub_ps(p1[5], p2[5]), pMul, p2[5]);    // alpha-blending adjustment
                    p1[6] = _mm_fmadd_ps(_mm_sub_ps(p1[6], p2[6]), pMul, p2[6]);    // alpha-blending adjustment
                    p1[7] = _mm_fmadd_ps(_mm_sub_ps(p1[7], p2[7]), pMul, p2[7]);    // alpha-blending adjustment
                    p1[8] = _mm_fmadd_ps(_mm_sub_ps(p1[8], p2[8]), pMul, p2[8]);    // alpha-blending adjustment
                    p1[9] = _mm_fmadd_ps(_mm_sub_ps(p1[9], p2[9]), pMul, p2[9]);    // alpha-blending adjustment
                    p1[10] = _mm_fmadd_ps(_mm_sub_ps(p1[10], p2[10]), pMul, p2[10]);    // alpha-blending adjustment
                    p1[11] = _mm_fmadd_ps(_mm_sub_ps(p1[11], p2[11]), pMul, p2[11]);    // alpha-blending adjustment
                    rpp_simd_store(rpp_store48_f32pln3_to_i8pkd3, dstPtrTemp, p1);    // simd stores

                    srcPtr1TempR += 16;
                    srcPtr1TempG += 16;
                    srcPtr1TempB += 16;
                    srcPtr2TempR += 16;
                    srcPtr2TempG += 16;
                    srcPtr2TempB += 16;
                    dstPtrTemp += 48;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    dstPtrTemp[0] = (Rpp8s) RPPPIXELCHECKI8((((Rpp32f) (*srcPtr1TempR) - (Rpp32f) (*srcPtr2TempR)) * alpha) + (Rpp32f) (*srcPtr2TempR));
                    dstPtrTemp[1] = (Rpp8s) RPPPIXELCHECKI8((((Rpp32f) (*srcPtr1TempG) - (Rpp32f) (*srcPtr2TempG)) * alpha) + (Rpp32f) (*srcPtr2TempG));
                    dstPtrTemp[2] = (Rpp8s) RPPPIXELCHECKI8((((Rpp32f) (*srcPtr1TempB) - (Rpp32f) (*srcPtr2TempB)) * alpha) + (Rpp32f) (*srcPtr2TempB));

                    srcPtr1TempR++;
                    srcPtr2TempR++;
                    srcPtr1TempG++;
                    srcPtr2TempG++;
                    srcPtr1TempB++;
                    srcPtr2TempB++;
                    dstPtrTemp += 3;
                }

                srcPtr1RowR += srcDescPtr->strides.hStride;
                srcPtr1RowG += srcDescPtr->strides.hStride;
                srcPtr1RowB += srcDescPtr->strides.hStride;
                srcPtr2RowR += srcDescPtr->strides.hStride;
                srcPtr2RowG += srcDescPtr->strides.hStride;
                srcPtr2RowB += srcDescPtr->strides.hStride;
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }

        // Blend without fused output-layout toggle (NHWC -> NHWC or NCHW -> NCHW)
        else
        {
            Rpp32u alignedLength = bufferLength & ~15;

            for(int c = 0; c < layoutParams.channelParam; c++)
            {
                Rpp8s *srcPtr1Row, *srcPtr2Row, *dstPtrRow;
                srcPtr1Row = srcPtr1Channel;
                srcPtr2Row = srcPtr2Channel;
                dstPtrRow = dstPtrChannel;

                for(int i = 0; i < roiPtr->xywhROI.roiHeight; i++)
                {
                    Rpp8s *srcPtr1Temp, *srcPtr2Temp, *dstPtrTemp;
                    srcPtr1Temp = srcPtr1Row;
                    srcPtr2Temp = srcPtr2Row;
                    dstPtrTemp = dstPtrRow;

                    int vectorLoopCount = 0;
                    for (; vectorLoopCount < alignedLength; vectorLoopCount+=16)
                    {
                        __m128 p1[4], p2[4];

                        rpp_simd_load(rpp_load16_i8_to_f32, srcPtr1Temp, p1);    // simd loads
                        rpp_simd_load(rpp_load16_i8_to_f32, srcPtr2Temp, p2);    // simd loads
                        p1[0] = _mm_fmadd_ps(_mm_sub_ps(p1[0], p2[0]), pMul, p2[0]);    // alpha-blending adjustment
                        p1[1] = _mm_fmadd_ps(_mm_sub_ps(p1[1], p2[1]), pMul, p2[1]);    // alpha-blending adjustment
                        p1[2] = _mm_fmadd_ps(_mm_sub_ps(p1[2], p2[2]), pMul, p2[2]);    // alpha-blending adjustment
                        p1[3] = _mm_fmadd_ps(_mm_sub_ps(p1[3], p2[3]), pMul, p2[3]);    // alpha-blending adjustment
                        rpp_simd_store(rpp_store16_f32_to_i8, dstPtrTemp, p1);    // simd stores

                        srcPtr1Temp +=16;
                        srcPtr2Temp +=16;
                        dstPtrTemp +=16;
                    }
                    for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                    {
                        *dstPtrTemp = (Rpp8s) RPPPIXELCHECKI8((((Rpp32f) (*srcPtr1Temp) - (Rpp32f) (*srcPtr2Temp)) * alpha) + (Rpp32f) (*srcPtr2Temp));

                        srcPtr1Temp++;
                        srcPtr2Temp++;
                        dstPtrTemp++;
                    }

                    srcPtr1Row += srcDescPtr->strides.hStride;
                    srcPtr2Row += srcDescPtr->strides.hStride;
                    dstPtrRow += dstDescPtr->strides.hStride;
                }

                srcPtr1Channel += srcDescPtr->strides.cStride;
                srcPtr2Channel += srcDescPtr->strides.cStride;
                dstPtrChannel += dstDescPtr->strides.cStride;
            }
        }
    }

    return RPP_SUCCESS;
}

/************ color_jitter ************/

RppStatus color_jitter_u8_u8_host_tensor(Rpp8u *srcPtr,
                                         RpptDescPtr srcDescPtr,
                                         Rpp8u *dstPtr,
                                         RpptDescPtr dstDescPtr,
                                         Rpp32f *brightnessTensor,
                                         Rpp32f *contrastTensor,
                                         Rpp32f *hueTensor,
                                         Rpp32f *saturationTensor,
                                         RpptROIPtr roiTensorPtrSrc,
                                         RpptRoiType roiType,
                                         RppLayoutParams layoutParams)
{
    RpptROI roiDefault;
    RpptROIPtr roiPtrDefault;
    roiPtrDefault = &roiDefault;
    roiPtrDefault->xywhROI.xy.x = 0;
    roiPtrDefault->xywhROI.xy.y = 0;
    roiPtrDefault->xywhROI.roiWidth = srcDescPtr->w;
    roiPtrDefault->xywhROI.roiHeight = srcDescPtr->h;

    omp_set_dynamic(0);
#pragma omp parallel for num_threads(dstDescPtr->n)
    for(int batchCount = 0; batchCount < dstDescPtr->n; batchCount++)
    {
        RpptROI roi;
        RpptROIPtr roiPtr;

        if (&roiTensorPtrSrc[batchCount] == NULL)
        {
            roiPtr = roiPtrDefault;
        }
        else
        {
            RpptROIPtr roiPtrInput = &roiTensorPtrSrc[batchCount];

            RpptROI roiImage;
            RpptROIPtr roiPtrImage;

            if (roiType == RpptRoiType::LTRB)
            {
                roiPtrImage = &roiImage;
                compute_xywh_from_ltrb_host(roiPtrInput, roiPtrImage);
            }
            else if (roiType == RpptRoiType::XYWH)
            {
                roiPtrImage = roiPtrInput;
            }

            roiPtr = &roi;
            compute_roi_boundary_check_host(roiPtrImage, roiPtr, roiPtrDefault);
        }

        Rpp32f brightnessParam = brightnessTensor[batchCount];
        Rpp32f contrastParam = contrastTensor[batchCount];
        Rpp32f hueParam = hueTensor[batchCount];
        Rpp32f saturationParam = saturationTensor[batchCount];

        Rpp8u *srcPtrImage, *dstPtrImage;
        srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;

        Rpp32u bufferLength = roiPtr->xywhROI.roiWidth * layoutParams.bufferMultiplier;

        alignas(64) Rpp32f ctm[16] = {1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f};
        compute_color_jitter_ctm_host(brightnessParam, contrastParam, hueParam, saturationParam, ctm);

        __m128 pCtm[12];
        for(int i = 0; i < 12; i++)
        {
            pCtm[i] = _mm_set1_ps(ctm[i]);
        }

        Rpp8u *srcPtrChannel, *dstPtrChannel;
        srcPtrChannel = srcPtrImage + (roiPtr->xywhROI.xy.y * srcDescPtr->strides.hStride) + (roiPtr->xywhROI.xy.x * layoutParams.bufferMultiplier);
        dstPtrChannel = dstPtrImage;

        // Color Jitter with fused output-layout toggle (NHWC -> NCHW)
        if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp32u alignedLength = bufferLength & ~47;

            Rpp8u *srcPtrRow, *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
            srcPtrRow = srcPtrChannel;
            dstPtrRowR = dstPtrChannel;
            dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
            dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;

            for(int i = 0; i < roiPtr->xywhROI.roiHeight; i++)
            {
                Rpp8u *srcPtrTemp, *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
                srcPtrTemp = srcPtrRow;
                dstPtrTempR = dstPtrRowR;
                dstPtrTempG = dstPtrRowG;
                dstPtrTempB = dstPtrRowB;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount+=48)
                {
                    __m128 p[12];

                    rpp_simd_load(rpp_load48_u8pkd3_to_f32pln3, srcPtrTemp, p);    // simd loads
                    compute_color_jitter_48_host(p, pCtm);    // color_jitter adjustment
                    rpp_simd_store(rpp_store48_f32pln3_to_u8pln3, dstPtrTempR, dstPtrTempG, dstPtrTempB, p);    // simd stores

                    srcPtrTemp += 48;
                    dstPtrTempR += 16;
                    dstPtrTempG += 16;
                    dstPtrTempB += 16;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount+=3)
                {
                    *dstPtrTempR = (Rpp8u) RPPPIXELCHECK(std::round(ctm[0] * srcPtrTemp[0] + ctm[1] * srcPtrTemp[1] + ctm[2] * srcPtrTemp[2] + ctm[3]));
                    *dstPtrTempG = (Rpp8u) RPPPIXELCHECK(std::round(ctm[4] * srcPtrTemp[0] + ctm[5] * srcPtrTemp[1] + ctm[6] * srcPtrTemp[2] + ctm[7]));
                    *dstPtrTempB = (Rpp8u) RPPPIXELCHECK(std::round(ctm[8] * srcPtrTemp[0] + ctm[9] * srcPtrTemp[1] + ctm[10] * srcPtrTemp[2] + ctm[11]));

                    srcPtrTemp+=3;
                    dstPtrTempR++;
                    dstPtrTempG++;
                    dstPtrTempB++;
                }

                srcPtrRow += srcDescPtr->strides.hStride;
                dstPtrRowR += dstDescPtr->strides.hStride;
                dstPtrRowG += dstDescPtr->strides.hStride;
                dstPtrRowB += dstDescPtr->strides.hStride;
            }
        }

        // Color Jitter with fused output-layout toggle (NCHW -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp32u alignedLength = bufferLength & ~47;

            Rpp8u *srcPtrRowR, *srcPtrRowG, *srcPtrRowB, *dstPtrRow;
            srcPtrRowR = srcPtrChannel;
            srcPtrRowG = srcPtrRowR + srcDescPtr->strides.cStride;
            srcPtrRowB = srcPtrRowG + srcDescPtr->strides.cStride;
            dstPtrRow = dstPtrChannel;

            for(int i = 0; i < roiPtr->xywhROI.roiHeight; i++)
            {
                Rpp8u *srcPtrTempR, *srcPtrTempG, *srcPtrTempB, *dstPtrTemp;
                srcPtrTempR = srcPtrRowR;
                srcPtrTempG = srcPtrRowG;
                srcPtrTempB = srcPtrRowB;
                dstPtrTemp = dstPtrRow;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount+=16)
                {
                    __m128 p[12];

                    rpp_simd_load(rpp_load48_u8pln3_to_f32pln3, srcPtrTempR, srcPtrTempG, srcPtrTempB, p);    // simd loads
                    compute_color_jitter_48_host(p, pCtm);    // color_jitter adjustment
                    rpp_simd_store(rpp_store48_f32pln3_to_u8pkd3, dstPtrTemp, p);    // simd stores

                    srcPtrTempR += 16;
                    srcPtrTempG += 16;
                    srcPtrTempB += 16;
                    dstPtrTemp += 48;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    dstPtrTemp[0] = (Rpp8u) RPPPIXELCHECK(std::round(ctm[0] * *srcPtrTempR + ctm[1] * *srcPtrTempG + ctm[2] * *srcPtrTempB + ctm[3]));
                    dstPtrTemp[1] = (Rpp8u) RPPPIXELCHECK(std::round(ctm[4] * *srcPtrTempR + ctm[5] * *srcPtrTempG + ctm[6] * *srcPtrTempB + ctm[7]));
                    dstPtrTemp[2] = (Rpp8u) RPPPIXELCHECK(std::round(ctm[8] * *srcPtrTempR + ctm[9] * *srcPtrTempG + ctm[10] * *srcPtrTempB + ctm[11]));

                    srcPtrTempR++;
                    srcPtrTempG++;
                    srcPtrTempB++;
                    dstPtrTemp += 3;
                }

                srcPtrRowR += srcDescPtr->strides.hStride;
                srcPtrRowG += srcDescPtr->strides.hStride;
                srcPtrRowB += srcDescPtr->strides.hStride;
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }

        // Color Jitter without fused output-layout toggle (NHWC -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp32u alignedLength = bufferLength & ~47;

            Rpp8u *srcPtrRow, *dstPtrRow;
            srcPtrRow = srcPtrChannel;
            dstPtrRow = dstPtrChannel;

            for(int i = 0; i < roiPtr->xywhROI.roiHeight; i++)
            {
                Rpp8u *srcPtrTemp, *dstPtrTemp;
                srcPtrTemp = srcPtrRow;
                dstPtrTemp = dstPtrRow;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount+=48)
                {
                    __m128 p[12];

                    rpp_simd_load(rpp_load48_u8pkd3_to_f32pln3, srcPtrTemp, p);    // simd loads
                    compute_color_jitter_48_host(p, pCtm);    // color_jitter adjustment
                    rpp_simd_store(rpp_store48_f32pln3_to_u8pkd3, dstPtrTemp, p);    // simd stores

                    srcPtrTemp += 48;
                    dstPtrTemp += 48;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount+=3)
                {
                    dstPtrTemp[0] = (Rpp8u) RPPPIXELCHECK(std::round(ctm[0] * srcPtrTemp[0] + ctm[1] * srcPtrTemp[1] + ctm[2] * srcPtrTemp[2] + ctm[3]));
                    dstPtrTemp[1] = (Rpp8u) RPPPIXELCHECK(std::round(ctm[4] * srcPtrTemp[0] + ctm[5] * srcPtrTemp[1] + ctm[6] * srcPtrTemp[2] + ctm[7]));
                    dstPtrTemp[2] = (Rpp8u) RPPPIXELCHECK(std::round(ctm[8] * srcPtrTemp[0] + ctm[9] * srcPtrTemp[1] + ctm[10] * srcPtrTemp[2] + ctm[11]));

                    srcPtrTemp += 3;
                    dstPtrTemp += 3;
                }

                srcPtrRow += srcDescPtr->strides.hStride;
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }

        // Color Jitter without fused output-layout toggle (NCHW -> NCHW)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp32u alignedLength = bufferLength & ~47;

            Rpp8u *srcPtrRowR, *srcPtrRowG, *srcPtrRowB, *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
            srcPtrRowR = srcPtrChannel;
            srcPtrRowG = srcPtrRowR + srcDescPtr->strides.cStride;
            srcPtrRowB = srcPtrRowG + srcDescPtr->strides.cStride;
            dstPtrRowR = dstPtrChannel;
            dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
            dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;

            for(int i = 0; i < roiPtr->xywhROI.roiHeight; i++)
            {
                Rpp8u *srcPtrTempR, *srcPtrTempG, *srcPtrTempB, *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
                srcPtrTempR = srcPtrRowR;
                srcPtrTempG = srcPtrRowG;
                srcPtrTempB = srcPtrRowB;
                dstPtrTempR = dstPtrRowR;
                dstPtrTempG = dstPtrRowG;
                dstPtrTempB = dstPtrRowB;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount+=16)
                {
                    __m128 p[12];

                    rpp_simd_load(rpp_load48_u8pln3_to_f32pln3, srcPtrTempR, srcPtrTempG, srcPtrTempB, p);    // simd loads
                    compute_color_jitter_48_host(p, pCtm);    // color_jitter adjustment
                    rpp_simd_store(rpp_store48_f32pln3_to_u8pln3, dstPtrTempR, dstPtrTempG, dstPtrTempB, p);    // simd stores

                    srcPtrTempR += 16;
                    srcPtrTempG += 16;
                    srcPtrTempB += 16;
                    dstPtrTempR += 16;
                    dstPtrTempG += 16;
                    dstPtrTempB += 16;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    *dstPtrTempR = (Rpp8u) RPPPIXELCHECK(std::round(ctm[0] * *srcPtrTempR + ctm[1] * *srcPtrTempG + ctm[2] * *srcPtrTempB + ctm[3]));
                    *dstPtrTempG = (Rpp8u) RPPPIXELCHECK(std::round(ctm[4] * *srcPtrTempR + ctm[5] * *srcPtrTempG + ctm[6] * *srcPtrTempB + ctm[7]));
                    *dstPtrTempB = (Rpp8u) RPPPIXELCHECK(std::round(ctm[8] * *srcPtrTempR + ctm[9] * *srcPtrTempG + ctm[10] * *srcPtrTempB + ctm[11]));

                    srcPtrTempR++;
                    srcPtrTempG++;
                    srcPtrTempB++;
                    dstPtrTempR++;
                    dstPtrTempG++;
                    dstPtrTempB++;
                }

                srcPtrRowR += srcDescPtr->strides.hStride;
                srcPtrRowG += srcDescPtr->strides.hStride;
                srcPtrRowB += srcDescPtr->strides.hStride;
                dstPtrRowR += dstDescPtr->strides.hStride;
                dstPtrRowG += dstDescPtr->strides.hStride;
                dstPtrRowB += dstDescPtr->strides.hStride;
            }
        }
    }

    return RPP_SUCCESS;
}

RppStatus color_jitter_f32_f32_host_tensor(Rpp32f *srcPtr,
                                           RpptDescPtr srcDescPtr,
                                           Rpp32f *dstPtr,
                                           RpptDescPtr dstDescPtr,
                                           Rpp32f *brightnessTensor,
                                           Rpp32f *contrastTensor,
                                           Rpp32f *hueTensor,
                                           Rpp32f *saturationTensor,
                                           RpptROIPtr roiTensorPtrSrc,
                                           RpptRoiType roiType,
                                           RppLayoutParams layoutParams)
{
    RpptROI roiDefault;
    RpptROIPtr roiPtrDefault;
    roiPtrDefault = &roiDefault;
    roiPtrDefault->xywhROI.xy.x = 0;
    roiPtrDefault->xywhROI.xy.y = 0;
    roiPtrDefault->xywhROI.roiWidth = srcDescPtr->w;
    roiPtrDefault->xywhROI.roiHeight = srcDescPtr->h;

    omp_set_dynamic(0);
#pragma omp parallel for num_threads(dstDescPtr->n)
    for(int batchCount = 0; batchCount < dstDescPtr->n; batchCount++)
    {
        RpptROI roi;
        RpptROIPtr roiPtr;

        if (&roiTensorPtrSrc[batchCount] == NULL)
        {
            roiPtr = roiPtrDefault;
        }
        else
        {
            RpptROIPtr roiPtrInput = &roiTensorPtrSrc[batchCount];

            RpptROI roiImage;
            RpptROIPtr roiPtrImage;

            if (roiType == RpptRoiType::LTRB)
            {
                roiPtrImage = &roiImage;
                compute_xywh_from_ltrb_host(roiPtrInput, roiPtrImage);
            }
            else if (roiType == RpptRoiType::XYWH)
            {
                roiPtrImage = roiPtrInput;
            }

            roiPtr = &roi;
            compute_roi_boundary_check_host(roiPtrImage, roiPtr, roiPtrDefault);
        }

        Rpp32f brightnessParam = brightnessTensor[batchCount];
        Rpp32f contrastParam = contrastTensor[batchCount];
        Rpp32f hueParam = hueTensor[batchCount];
        Rpp32f saturationParam = saturationTensor[batchCount];

        Rpp32f *srcPtrImage, *dstPtrImage;
        srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;

        Rpp32u bufferLength = roiPtr->xywhROI.roiWidth * layoutParams.bufferMultiplier;

        alignas(64) Rpp32f ctm[16] = {1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f};
        compute_color_jitter_ctm_host(brightnessParam, contrastParam, hueParam, saturationParam, ctm);

        __m128 pCtm[12];
        for(int i = 0; i < 12; i++)
        {
            pCtm[i] = _mm_set1_ps(ctm[i]);
        }

        Rpp32f *srcPtrChannel, *dstPtrChannel;
        srcPtrChannel = srcPtrImage + (roiPtr->xywhROI.xy.y * srcDescPtr->strides.hStride) + (roiPtr->xywhROI.xy.x * layoutParams.bufferMultiplier);
        dstPtrChannel = dstPtrImage;

        // Color Jitter with fused output-layout toggle (NHWC -> NCHW)
        if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp32u alignedLength = bufferLength & ~11;

            Rpp32f *srcPtrRow, *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
            srcPtrRow = srcPtrChannel;
            dstPtrRowR = dstPtrChannel;
            dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
            dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;

            for(int i = 0; i < roiPtr->xywhROI.roiHeight; i++)
            {
                Rpp32f *srcPtrTemp, *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
                srcPtrTemp = srcPtrRow;
                dstPtrTempR = dstPtrRowR;
                dstPtrTempG = dstPtrRowG;
                dstPtrTempB = dstPtrRowB;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount+=12)
                {
                    __m128 p[4];

                    rpp_simd_load(rpp_load12_f32pkd3_to_f32pln3, srcPtrTemp, p);    // simd loads
                    compute_color_jitter_12_host(p, pCtm);    // color_jitter adjustment
                    rpp_simd_store(rpp_store12_f32pln3_to_f32pln3, dstPtrTempR, dstPtrTempG, dstPtrTempB, p);    // simd stores

                    srcPtrTemp += 12;
                    dstPtrTempR += 4;
                    dstPtrTempG += 4;
                    dstPtrTempB += 4;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount+=3)
                {
                    *dstPtrTempR = RPPPIXELCHECKF32(ctm[0] * srcPtrTemp[0] + ctm[1] * srcPtrTemp[1] + ctm[2] * srcPtrTemp[2] + ctm[3]);
                    *dstPtrTempG = RPPPIXELCHECKF32(ctm[4] * srcPtrTemp[0] + ctm[5] * srcPtrTemp[1] + ctm[6] * srcPtrTemp[2] + ctm[7]);
                    *dstPtrTempB = RPPPIXELCHECKF32(ctm[8] * srcPtrTemp[0] + ctm[9] * srcPtrTemp[1] + ctm[10] * srcPtrTemp[2] + ctm[11]);

                    srcPtrTemp += 3;
                    dstPtrTempR++;
                    dstPtrTempG++;
                    dstPtrTempB++;
                }

                srcPtrRow += srcDescPtr->strides.hStride;
                dstPtrRowR += dstDescPtr->strides.hStride;
                dstPtrRowG += dstDescPtr->strides.hStride;
                dstPtrRowB += dstDescPtr->strides.hStride;
            }
        }

        // Color Jitter with fused output-layout toggle (NCHW -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp32u alignedLength = bufferLength & ~11;

            Rpp32f *srcPtrRowR, *srcPtrRowG, *srcPtrRowB, *dstPtrRow;
            srcPtrRowR = srcPtrChannel;
            srcPtrRowG = srcPtrRowR + srcDescPtr->strides.cStride;
            srcPtrRowB = srcPtrRowG + srcDescPtr->strides.cStride;
            dstPtrRow = dstPtrChannel;

            for(int i = 0; i < roiPtr->xywhROI.roiHeight; i++)
            {
                Rpp32f *srcPtrTempR, *srcPtrTempG, *srcPtrTempB, *dstPtrTemp;
                srcPtrTempR = srcPtrRowR;
                srcPtrTempG = srcPtrRowG;
                srcPtrTempB = srcPtrRowB;
                dstPtrTemp = dstPtrRow;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount+=4)
                {
                    __m128 p[4];

                    rpp_simd_load(rpp_load12_f32pln3_to_f32pln3, srcPtrTempR, srcPtrTempG, srcPtrTempB, p);    // simd loads
                    compute_color_jitter_12_host(p, pCtm);    // color_jitter adjustment
                    rpp_simd_store(rpp_store12_f32pln3_to_f32pkd3, dstPtrTemp, p);    // simd stores

                    srcPtrTempR += 4;
                    srcPtrTempG += 4;
                    srcPtrTempB += 4;
                    dstPtrTemp += 12;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    dstPtrTemp[0] = RPPPIXELCHECKF32(ctm[0] * *srcPtrTempR + ctm[1] * *srcPtrTempG + ctm[2] * *srcPtrTempB + ctm[3]);
                    dstPtrTemp[1] = RPPPIXELCHECKF32(ctm[4] * *srcPtrTempR + ctm[5] * *srcPtrTempG + ctm[6] * *srcPtrTempB + ctm[7]);
                    dstPtrTemp[2] = RPPPIXELCHECKF32(ctm[8] * *srcPtrTempR + ctm[9] * *srcPtrTempG + ctm[10] * *srcPtrTempB + ctm[11]);

                    srcPtrTempR++;
                    srcPtrTempG++;
                    srcPtrTempB++;
                    dstPtrTemp += 3;
                }

                srcPtrRowR += srcDescPtr->strides.hStride;
                srcPtrRowG += srcDescPtr->strides.hStride;
                srcPtrRowB += srcDescPtr->strides.hStride;
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }

        // Color Jitter without fused output-layout toggle (NHWC -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp32u alignedLength = bufferLength & ~11;

            Rpp32f *srcPtrRow, *dstPtrRow;
            srcPtrRow = srcPtrChannel;
            dstPtrRow = dstPtrChannel;

            for(int i = 0; i < roiPtr->xywhROI.roiHeight; i++)
            {
                Rpp32f *srcPtrTemp, *dstPtrTemp;
                srcPtrTemp = srcPtrRow;
                dstPtrTemp = dstPtrRow;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount+=12)
                {
                    __m128 p[4];

                    rpp_simd_load(rpp_load12_f32pkd3_to_f32pln3, srcPtrTemp, p);    // simd loads
                    compute_color_jitter_12_host(p, pCtm);    // color_jitter adjustment
                    rpp_simd_store(rpp_store12_f32pln3_to_f32pkd3, dstPtrTemp, p);    // simd stores

                    srcPtrTemp += 12;
                    dstPtrTemp += 12;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount+=3)
                {
                    dstPtrTemp[0] = RPPPIXELCHECKF32(ctm[0] * srcPtrTemp[0] + ctm[1] * srcPtrTemp[1] + ctm[2] * srcPtrTemp[2] + ctm[3]);
                    dstPtrTemp[1] = RPPPIXELCHECKF32(ctm[4] * srcPtrTemp[0] + ctm[5] * srcPtrTemp[1] + ctm[6] * srcPtrTemp[2] + ctm[7]);
                    dstPtrTemp[2] = RPPPIXELCHECKF32(ctm[8] * srcPtrTemp[0] + ctm[9] * srcPtrTemp[1] + ctm[10] * srcPtrTemp[2] + ctm[11]);

                    srcPtrTemp += 3;
                    dstPtrTemp += 3;
                }

                srcPtrRow += srcDescPtr->strides.hStride;
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }

        // Color Jitter without fused output-layout toggle (NCHW -> NCHW)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp32u alignedLength = bufferLength & ~11;

            Rpp32f *srcPtrRowR, *srcPtrRowG, *srcPtrRowB, *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
            srcPtrRowR = srcPtrChannel;
            srcPtrRowG = srcPtrRowR + srcDescPtr->strides.cStride;
            srcPtrRowB = srcPtrRowG + srcDescPtr->strides.cStride;
            dstPtrRowR = dstPtrChannel;
            dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
            dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;

            for(int i = 0; i < roiPtr->xywhROI.roiHeight; i++)
            {
                Rpp32f *srcPtrTempR, *srcPtrTempG, *srcPtrTempB, *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
                srcPtrTempR = srcPtrRowR;
                srcPtrTempG = srcPtrRowG;
                srcPtrTempB = srcPtrRowB;
                dstPtrTempR = dstPtrRowR;
                dstPtrTempG = dstPtrRowG;
                dstPtrTempB = dstPtrRowB;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount+=4)
                {
                    __m128 p[4];

                    rpp_simd_load(rpp_load12_f32pln3_to_f32pln3, srcPtrTempR, srcPtrTempG, srcPtrTempB, p);    // simd loads
                    compute_color_jitter_12_host(p, pCtm);    // color_jitter adjustment
                    rpp_simd_store(rpp_store12_f32pln3_to_f32pln3, dstPtrTempR, dstPtrTempG, dstPtrTempB, p);    // simd stores

                    srcPtrTempR += 4;
                    srcPtrTempG += 4;
                    srcPtrTempB += 4;
                    dstPtrTempR += 4;
                    dstPtrTempG += 4;
                    dstPtrTempB += 4;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    *dstPtrTempR = RPPPIXELCHECKF32(ctm[0] * *srcPtrTempR + ctm[1] * *srcPtrTempG + ctm[2] * *srcPtrTempB + ctm[3]);
                    *dstPtrTempG = RPPPIXELCHECKF32(ctm[4] * *srcPtrTempR + ctm[5] * *srcPtrTempG + ctm[6] * *srcPtrTempB + ctm[7]);
                    *dstPtrTempB = RPPPIXELCHECKF32(ctm[8] * *srcPtrTempR + ctm[9] * *srcPtrTempG + ctm[10] * *srcPtrTempB + ctm[11]);

                    srcPtrTempR++;
                    srcPtrTempG++;
                    srcPtrTempB++;
                    dstPtrTempR++;
                    dstPtrTempG++;
                    dstPtrTempB++;
                }

                srcPtrRowR += srcDescPtr->strides.hStride;
                srcPtrRowG += srcDescPtr->strides.hStride;
                srcPtrRowB += srcDescPtr->strides.hStride;
                dstPtrRowR += srcDescPtr->strides.hStride;
                dstPtrRowG += srcDescPtr->strides.hStride;
                dstPtrRowB += srcDescPtr->strides.hStride;
            }
        }
    }

    return RPP_SUCCESS;
}

RppStatus color_jitter_f16_f16_host_tensor(Rpp16f *srcPtr,
                                           RpptDescPtr srcDescPtr,
                                           Rpp16f *dstPtr,
                                           RpptDescPtr dstDescPtr,
                                           Rpp32f *brightnessTensor,
                                           Rpp32f *contrastTensor,
                                           Rpp32f *hueTensor,
                                           Rpp32f *saturationTensor,
                                           RpptROIPtr roiTensorPtrSrc,
                                           RpptRoiType roiType,
                                           RppLayoutParams layoutParams)
{
    RpptROI roiDefault;
    RpptROIPtr roiPtrDefault;
    roiPtrDefault = &roiDefault;
    roiPtrDefault->xywhROI.xy.x = 0;
    roiPtrDefault->xywhROI.xy.y = 0;
    roiPtrDefault->xywhROI.roiWidth = srcDescPtr->w;
    roiPtrDefault->xywhROI.roiHeight = srcDescPtr->h;

    omp_set_dynamic(0);
#pragma omp parallel for num_threads(dstDescPtr->n)
    for(int batchCount = 0; batchCount < dstDescPtr->n; batchCount++)
    {
        RpptROI roi;
        RpptROIPtr roiPtr;

        if (&roiTensorPtrSrc[batchCount] == NULL)
        {
            roiPtr = roiPtrDefault;
        }
        else
        {
            RpptROIPtr roiPtrInput = &roiTensorPtrSrc[batchCount];

            RpptROI roiImage;
            RpptROIPtr roiPtrImage;

            if (roiType == RpptRoiType::LTRB)
            {
                roiPtrImage = &roiImage;
                compute_xywh_from_ltrb_host(roiPtrInput, roiPtrImage);
            }
            else if (roiType == RpptRoiType::XYWH)
            {
                roiPtrImage = roiPtrInput;
            }

            roiPtr = &roi;
            compute_roi_boundary_check_host(roiPtrImage, roiPtr, roiPtrDefault);
        }

        Rpp32f brightnessParam = brightnessTensor[batchCount];
        Rpp32f contrastParam = contrastTensor[batchCount];
        Rpp32f hueParam = hueTensor[batchCount];
        Rpp32f saturationParam = saturationTensor[batchCount];

        Rpp16f *srcPtrImage, *dstPtrImage;
        srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;

        Rpp32u bufferLength = roiPtr->xywhROI.roiWidth * layoutParams.bufferMultiplier;

        alignas(64) Rpp32f ctm[16] = {1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f};
        compute_color_jitter_ctm_host(brightnessParam, contrastParam, hueParam, saturationParam, ctm);

        __m128 pCtm[12];
        for(int i = 0; i < 12; i++)
        {
            pCtm[i] = _mm_set1_ps(ctm[i]);
        }

        Rpp16f *srcPtrChannel, *dstPtrChannel;
        srcPtrChannel = srcPtrImage + (roiPtr->xywhROI.xy.y * srcDescPtr->strides.hStride) + (roiPtr->xywhROI.xy.x * layoutParams.bufferMultiplier);
        dstPtrChannel = dstPtrImage;

        // Color Jitter with fused output-layout toggle (NHWC -> NCHW)
        if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp32u alignedLength = bufferLength & ~11;

            Rpp16f *srcPtrRow, *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
            srcPtrRow = srcPtrChannel;
            dstPtrRowR = dstPtrChannel;
            dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
            dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;

            for(int i = 0; i < roiPtr->xywhROI.roiHeight; i++)
            {
                Rpp16f *srcPtrTemp, *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
                srcPtrTemp = srcPtrRow;
                dstPtrTempR = dstPtrRowR;
                dstPtrTempG = dstPtrRowG;
                dstPtrTempB = dstPtrRowB;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount+=12)
                {
                    Rpp32f srcPtrTemp_ps[12], dstPtrTemp_ps[12];

                    for(int cnt = 0; cnt < 12; cnt++)
                    {
                        *(srcPtrTemp_ps + cnt) = (Rpp32f) *(srcPtrTemp + cnt);
                    }

                    __m128 p[4];

                    rpp_simd_load(rpp_load12_f32pkd3_to_f32pln3, srcPtrTemp_ps, p);    // simd loads
                    compute_color_jitter_12_host(p, pCtm);    // color_jitter adjustment
                    rpp_simd_store(rpp_store12_f32pln3_to_f32pln3, dstPtrTemp_ps, dstPtrTemp_ps + 4, dstPtrTemp_ps + 8, p);    // simd stores

                    for(int cnt = 0; cnt < 4; cnt++)
                    {
                        *(dstPtrTempR + cnt) = (Rpp16f) *(dstPtrTemp_ps + cnt);
                        *(dstPtrTempG + cnt) = (Rpp16f) *(dstPtrTemp_ps + 4 + cnt);
                        *(dstPtrTempB + cnt) = (Rpp16f) *(dstPtrTemp_ps + 8 + cnt);
                    }

                    srcPtrTemp += 12;
                    dstPtrTempR += 4;
                    dstPtrTempG += 4;
                    dstPtrTempB += 4;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount+=3)
                {
                    *dstPtrTempR = (Rpp16f) RPPPIXELCHECKF32(ctm[0] * srcPtrTemp[0] + ctm[1] * srcPtrTemp[1] + ctm[2] * srcPtrTemp[2] + ctm[3]);
                    *dstPtrTempG = (Rpp16f) RPPPIXELCHECKF32(ctm[4] * srcPtrTemp[0] + ctm[5] * srcPtrTemp[1] + ctm[6] * srcPtrTemp[2] + ctm[7]);
                    *dstPtrTempB = (Rpp16f) RPPPIXELCHECKF32(ctm[8] * srcPtrTemp[0] + ctm[9] * srcPtrTemp[1] + ctm[10] * srcPtrTemp[2] + ctm[11]);

                    srcPtrTemp += 3;
                    dstPtrTempR++;
                    dstPtrTempG++;
                    dstPtrTempB++;
                }

                srcPtrRow += srcDescPtr->strides.hStride;
                dstPtrRowR += dstDescPtr->strides.hStride;
                dstPtrRowG += dstDescPtr->strides.hStride;
                dstPtrRowB += dstDescPtr->strides.hStride;
            }
        }

        // Color Jitter with fused output-layout toggle (NCHW -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp32u alignedLength = bufferLength & ~11;

            Rpp16f *srcPtrRowR, *srcPtrRowG, *srcPtrRowB, *dstPtrRow;
            srcPtrRowR = srcPtrChannel;
            srcPtrRowG = srcPtrRowR + srcDescPtr->strides.cStride;
            srcPtrRowB = srcPtrRowG + srcDescPtr->strides.cStride;
            dstPtrRow = dstPtrChannel;

            for(int i = 0; i < roiPtr->xywhROI.roiHeight; i++)
            {
                Rpp16f *srcPtrTempR, *srcPtrTempG, *srcPtrTempB, *dstPtrTemp;
                srcPtrTempR = srcPtrRowR;
                srcPtrTempG = srcPtrRowG;
                srcPtrTempB = srcPtrRowB;
                dstPtrTemp = dstPtrRow;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount+=4)
                {
                    Rpp32f srcPtrTemp_ps[12], dstPtrTemp_ps[13];

                    for(int cnt = 0; cnt < 4; cnt++)
                    {
                        *(srcPtrTemp_ps + cnt) = (Rpp32f) *(srcPtrTempR + cnt);
                        *(srcPtrTemp_ps + 4 + cnt) = (Rpp32f) *(srcPtrTempG + cnt);
                        *(srcPtrTemp_ps + 8 + cnt) = (Rpp32f) *(srcPtrTempB + cnt);
                    }

                    __m128 p[4];

                    rpp_simd_load(rpp_load12_f32pln3_to_f32pln3, srcPtrTemp_ps, srcPtrTemp_ps + 4, srcPtrTemp_ps + 8, p);    // simd loads
                    compute_color_jitter_12_host(p, pCtm);    // color_jitter adjustment
                    rpp_simd_store(rpp_store12_f32pln3_to_f32pkd3, dstPtrTemp_ps, p);    // simd stores

                    for(int cnt = 0; cnt < 12; cnt++)
                    {
                        *(dstPtrTemp + cnt) = (Rpp16f) *(dstPtrTemp_ps + cnt);
                    }

                    srcPtrTempR += 4;
                    srcPtrTempG += 4;
                    srcPtrTempB += 4;
                    dstPtrTemp += 12;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    dstPtrTemp[0] = (Rpp16f) RPPPIXELCHECKF32(ctm[0] * *srcPtrTempR + ctm[1] * *srcPtrTempG + ctm[2] * *srcPtrTempB + ctm[3]);
                    dstPtrTemp[1] = (Rpp16f) RPPPIXELCHECKF32(ctm[4] * *srcPtrTempR + ctm[5] * *srcPtrTempG + ctm[6] * *srcPtrTempB + ctm[7]);
                    dstPtrTemp[2] = (Rpp16f) RPPPIXELCHECKF32(ctm[8] * *srcPtrTempR + ctm[9] * *srcPtrTempG + ctm[10] * *srcPtrTempB + ctm[11]);

                    srcPtrTempR++;
                    srcPtrTempG++;
                    srcPtrTempB++;
                    dstPtrTemp += 3;
                }

                srcPtrRowR += srcDescPtr->strides.hStride;
                srcPtrRowG += srcDescPtr->strides.hStride;
                srcPtrRowB += srcDescPtr->strides.hStride;
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }

        // Color Jitter without fused output-layout toggle (NHWC -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp32u alignedLength = bufferLength & ~11;

            Rpp16f *srcPtrRow, *dstPtrRow;
            srcPtrRow = srcPtrChannel;
            dstPtrRow = dstPtrChannel;

            for(int i = 0; i < roiPtr->xywhROI.roiHeight; i++)
            {
                Rpp16f *srcPtrTemp, *dstPtrTemp;
                srcPtrTemp = srcPtrRow;
                dstPtrTemp = dstPtrRow;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount+=12)
                {
                    Rpp32f srcPtrTemp_ps[12], dstPtrTemp_ps[13];

                    for(int cnt = 0; cnt < 12; cnt++)
                    {
                        *(srcPtrTemp_ps + cnt) = (Rpp32f) *(srcPtrTemp + cnt);
                    }

                    __m128 p[4];

                    rpp_simd_load(rpp_load12_f32pkd3_to_f32pln3, srcPtrTemp_ps, p);    // simd loads
                    compute_color_jitter_12_host(p, pCtm);    // color_jitter adjustment
                    rpp_simd_store(rpp_store12_f32pln3_to_f32pkd3, dstPtrTemp_ps, p);    // simd stores

                    for(int cnt = 0; cnt < 12; cnt++)
                    {
                        *(dstPtrTemp + cnt) = (Rpp16f) *(dstPtrTemp_ps + cnt);
                    }

                    srcPtrTemp += 12;
                    dstPtrTemp += 12;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount+=3)
                {
                    dstPtrTemp[0] = (Rpp16f) RPPPIXELCHECKF32(ctm[0] * srcPtrTemp[0] + ctm[1] * srcPtrTemp[1] + ctm[2] * srcPtrTemp[2] + ctm[3]);
                    dstPtrTemp[1] = (Rpp16f) RPPPIXELCHECKF32(ctm[4] * srcPtrTemp[0] + ctm[5] * srcPtrTemp[1] + ctm[6] * srcPtrTemp[2] + ctm[7]);
                    dstPtrTemp[2] = (Rpp16f) RPPPIXELCHECKF32(ctm[8] * srcPtrTemp[0] + ctm[9] * srcPtrTemp[1] + ctm[10] * srcPtrTemp[2] + ctm[11]);

                    srcPtrTemp += 3;
                    dstPtrTemp += 3;
                }

                srcPtrRow += srcDescPtr->strides.hStride;
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }

        // Color Jitter without fused output-layout toggle (NCHW -> NCHW)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp32u alignedLength = bufferLength & ~11;

            Rpp16f *srcPtrRowR, *srcPtrRowG, *srcPtrRowB, *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
            srcPtrRowR = srcPtrChannel;
            srcPtrRowG = srcPtrRowR + srcDescPtr->strides.cStride;
            srcPtrRowB = srcPtrRowG + srcDescPtr->strides.cStride;
            dstPtrRowR = dstPtrChannel;
            dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
            dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;

            for(int i = 0; i < roiPtr->xywhROI.roiHeight; i++)
            {
                Rpp16f *srcPtrTempR, *srcPtrTempG, *srcPtrTempB, *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
                srcPtrTempR = srcPtrRowR;
                srcPtrTempG = srcPtrRowG;
                srcPtrTempB = srcPtrRowB;
                dstPtrTempR = dstPtrRowR;
                dstPtrTempG = dstPtrRowG;
                dstPtrTempB = dstPtrRowB;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount+=4)
                {
                    Rpp32f srcPtrTemp_ps[12], dstPtrTemp_ps[13];

                    for(int cnt = 0; cnt < 4; cnt++)
                    {
                        *(srcPtrTemp_ps + cnt) = (Rpp32f) *(srcPtrTempR + cnt);
                        *(srcPtrTemp_ps + 4 + cnt) = (Rpp32f) *(srcPtrTempG + cnt);
                        *(srcPtrTemp_ps + 8 + cnt) = (Rpp32f) *(srcPtrTempB + cnt);
                    }

                    __m128 p[4];

                    rpp_simd_load(rpp_load12_f32pln3_to_f32pln3, srcPtrTemp_ps, srcPtrTemp_ps + 4, srcPtrTemp_ps + 8, p);    // simd loads
                    compute_color_jitter_12_host(p, pCtm);    // color_jitter adjustment
                    rpp_simd_store(rpp_store12_f32pln3_to_f32pln3, dstPtrTemp_ps, dstPtrTemp_ps + 4, dstPtrTemp_ps + 8, p);    // simd stores

                    for(int cnt = 0; cnt < 4; cnt++)
                    {
                        *(dstPtrTempR + cnt) = (Rpp16f) *(dstPtrTemp_ps + cnt);
                        *(dstPtrTempG + cnt) = (Rpp16f) *(dstPtrTemp_ps + 4 + cnt);
                        *(dstPtrTempB + cnt) = (Rpp16f) *(dstPtrTemp_ps + 8 + cnt);
                    }

                    srcPtrTempR += 4;
                    srcPtrTempG += 4;
                    srcPtrTempB += 4;
                    dstPtrTempR += 4;
                    dstPtrTempG += 4;
                    dstPtrTempB += 4;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    *dstPtrTempR = (Rpp16f) RPPPIXELCHECKF32(ctm[0] * *srcPtrTempR + ctm[1] * *srcPtrTempG + ctm[2] * *srcPtrTempB + ctm[3]);
                    *dstPtrTempG = (Rpp16f) RPPPIXELCHECKF32(ctm[4] * *srcPtrTempR + ctm[5] * *srcPtrTempG + ctm[6] * *srcPtrTempB + ctm[7]);
                    *dstPtrTempB = (Rpp16f) RPPPIXELCHECKF32(ctm[8] * *srcPtrTempR + ctm[9] * *srcPtrTempG + ctm[10] * *srcPtrTempB + ctm[11]);

                    srcPtrTempR++;
                    srcPtrTempG++;
                    srcPtrTempB++;
                    dstPtrTempR++;
                    dstPtrTempG++;
                    dstPtrTempB++;
                }

                srcPtrRowR += srcDescPtr->strides.hStride;
                srcPtrRowG += srcDescPtr->strides.hStride;
                srcPtrRowB += srcDescPtr->strides.hStride;
                dstPtrRowR += srcDescPtr->strides.hStride;
                dstPtrRowG += srcDescPtr->strides.hStride;
                dstPtrRowB += srcDescPtr->strides.hStride;
            }
        }
    }

    return RPP_SUCCESS;
}

RppStatus color_jitter_i8_i8_host_tensor(Rpp8s *srcPtr,
                                         RpptDescPtr srcDescPtr,
                                         Rpp8s *dstPtr,
                                         RpptDescPtr dstDescPtr,
                                         Rpp32f *brightnessTensor,
                                         Rpp32f *contrastTensor,
                                         Rpp32f *hueTensor,
                                         Rpp32f *saturationTensor,
                                         RpptROIPtr roiTensorPtrSrc,
                                         RpptRoiType roiType,
                                         RppLayoutParams layoutParams)
{
    RpptROI roiDefault;
    RpptROIPtr roiPtrDefault;
    roiPtrDefault = &roiDefault;
    roiPtrDefault->xywhROI.xy.x = 0;
    roiPtrDefault->xywhROI.xy.y = 0;
    roiPtrDefault->xywhROI.roiWidth = srcDescPtr->w;
    roiPtrDefault->xywhROI.roiHeight = srcDescPtr->h;

    omp_set_dynamic(0);
#pragma omp parallel for num_threads(dstDescPtr->n)
    for(int batchCount = 0; batchCount < dstDescPtr->n; batchCount++)
    {
        RpptROI roi;
        RpptROIPtr roiPtr;

        if (&roiTensorPtrSrc[batchCount] == NULL)
        {
            roiPtr = roiPtrDefault;
        }
        else
        {
            RpptROIPtr roiPtrInput = &roiTensorPtrSrc[batchCount];

            RpptROI roiImage;
            RpptROIPtr roiPtrImage;

            if (roiType == RpptRoiType::LTRB)
            {
                roiPtrImage = &roiImage;
                compute_xywh_from_ltrb_host(roiPtrInput, roiPtrImage);
            }
            else if (roiType == RpptRoiType::XYWH)
            {
                roiPtrImage = roiPtrInput;
            }

            roiPtr = &roi;
            compute_roi_boundary_check_host(roiPtrImage, roiPtr, roiPtrDefault);
        }

        Rpp32f brightnessParam = brightnessTensor[batchCount];
        Rpp32f contrastParam = contrastTensor[batchCount];
        Rpp32f hueParam = hueTensor[batchCount];
        Rpp32f saturationParam = saturationTensor[batchCount];

        Rpp8s *srcPtrImage, *dstPtrImage;
        srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;

        Rpp32u bufferLength = roiPtr->xywhROI.roiWidth * layoutParams.bufferMultiplier;

        alignas(64) Rpp32f ctm[16] = {1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f};
        compute_color_jitter_ctm_host(brightnessParam, contrastParam, hueParam, saturationParam, ctm);

        __m128 pCtm[12];
        for(int i = 0; i < 12; i++)
        {
            pCtm[i] = _mm_set1_ps(ctm[i]);
        }

        Rpp8s *srcPtrChannel, *dstPtrChannel;
        srcPtrChannel = srcPtrImage + (roiPtr->xywhROI.xy.y * srcDescPtr->strides.hStride) + (roiPtr->xywhROI.xy.x * layoutParams.bufferMultiplier);
        dstPtrChannel = dstPtrImage;

        // Color Jitter with fused output-layout toggle (NHWC -> NCHW)
        if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp32u alignedLength = bufferLength & ~47;

            Rpp8s *srcPtrRow, *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
            srcPtrRow = srcPtrChannel;
            dstPtrRowR = dstPtrChannel;
            dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
            dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;

            for(int i = 0; i < roiPtr->xywhROI.roiHeight; i++)
            {
                Rpp8s *srcPtrTemp, *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
                srcPtrTemp = srcPtrRow;
                dstPtrTempR = dstPtrRowR;
                dstPtrTempG = dstPtrRowG;
                dstPtrTempB = dstPtrRowB;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount+=48)
                {
                    __m128 p[12];

                    rpp_simd_load(rpp_load48_i8pkd3_to_f32pln3, srcPtrTemp, p);    // simd loads
                    compute_color_jitter_48_host(p, pCtm);    // color_jitter adjustment
                    rpp_simd_store(rpp_store48_f32pln3_to_i8pln3, dstPtrTempR, dstPtrTempG, dstPtrTempB, p);    // simd stores

                    srcPtrTemp += 48;
                    dstPtrTempR += 16;
                    dstPtrTempG += 16;
                    dstPtrTempB += 16;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount+=3)
                {
                    Rpp32f srcPtrTempI8[3];
                    srcPtrTempI8[0] = (Rpp32f)srcPtrTemp[0] + 128;
                    srcPtrTempI8[1] = (Rpp32f)srcPtrTemp[1] + 128;
                    srcPtrTempI8[2] = (Rpp32f)srcPtrTemp[2] + 128;

                    *dstPtrTempR = (Rpp8s) RPPPIXELCHECKI8(std::round(ctm[0] * srcPtrTempI8[0] + ctm[1] * srcPtrTempI8[1] + ctm[2] * srcPtrTempI8[2] + ctm[3]) - 128);
                    *dstPtrTempG = (Rpp8s) RPPPIXELCHECKI8(std::round(ctm[4] * srcPtrTempI8[0] + ctm[5] * srcPtrTempI8[1] + ctm[6] * srcPtrTempI8[2] + ctm[7]) - 128);
                    *dstPtrTempB = (Rpp8s) RPPPIXELCHECKI8(std::round(ctm[8] * srcPtrTempI8[0] + ctm[9] * srcPtrTempI8[1] + ctm[10] * srcPtrTempI8[2] + ctm[11]) - 128);

                    srcPtrTemp+=3;
                    dstPtrTempR++;
                    dstPtrTempG++;
                    dstPtrTempB++;
                }

                srcPtrRow += srcDescPtr->strides.hStride;
                dstPtrRowR += dstDescPtr->strides.hStride;
                dstPtrRowG += dstDescPtr->strides.hStride;
                dstPtrRowB += dstDescPtr->strides.hStride;
            }
        }

        // Color Jitter with fused output-layout toggle (NCHW -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp32u alignedLength = bufferLength & ~47;

            Rpp8s *srcPtrRowR, *srcPtrRowG, *srcPtrRowB, *dstPtrRow;
            srcPtrRowR = srcPtrChannel;
            srcPtrRowG = srcPtrRowR + srcDescPtr->strides.cStride;
            srcPtrRowB = srcPtrRowG + srcDescPtr->strides.cStride;
            dstPtrRow = dstPtrChannel;

            for(int i = 0; i < roiPtr->xywhROI.roiHeight; i++)
            {
                Rpp8s *srcPtrTempR, *srcPtrTempG, *srcPtrTempB, *dstPtrTemp;
                srcPtrTempR = srcPtrRowR;
                srcPtrTempG = srcPtrRowG;
                srcPtrTempB = srcPtrRowB;
                dstPtrTemp = dstPtrRow;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount+=16)
                {
                    __m128 p[12];

                    rpp_simd_load(rpp_load48_i8pln3_to_f32pln3, srcPtrTempR, srcPtrTempG, srcPtrTempB, p);    // simd loads
                    compute_color_jitter_48_host(p, pCtm);    // color_jitter adjustment
                    rpp_simd_store(rpp_store48_f32pln3_to_i8pkd3, dstPtrTemp, p);    // simd stores

                    srcPtrTempR += 16;
                    srcPtrTempG += 16;
                    srcPtrTempB += 16;
                    dstPtrTemp += 48;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    Rpp32f srcPtrTempI8[3];
                    srcPtrTempI8[0] = (Rpp32f)*srcPtrTempR + 128;
                    srcPtrTempI8[1] = (Rpp32f)*srcPtrTempG + 128;
                    srcPtrTempI8[2] = (Rpp32f)*srcPtrTempB + 128;

                    dstPtrTemp[0] = (Rpp8s) RPPPIXELCHECKI8(std::round(ctm[0] * srcPtrTempI8[0] + ctm[1] * srcPtrTempI8[1] + ctm[2] * srcPtrTempI8[2] + ctm[3]) - 128);
                    dstPtrTemp[1] = (Rpp8s) RPPPIXELCHECKI8(std::round(ctm[4] * srcPtrTempI8[0] + ctm[5] * srcPtrTempI8[1] + ctm[6] * srcPtrTempI8[2] + ctm[7]) - 128);
                    dstPtrTemp[2] = (Rpp8s) RPPPIXELCHECKI8(std::round(ctm[8] * srcPtrTempI8[0] + ctm[9] * srcPtrTempI8[1] + ctm[10] * srcPtrTempI8[2] + ctm[11]) - 128);

                    srcPtrTempR++;
                    srcPtrTempG++;
                    srcPtrTempB++;
                    dstPtrTemp += 3;
                }

                srcPtrRowR += srcDescPtr->strides.hStride;
                srcPtrRowG += srcDescPtr->strides.hStride;
                srcPtrRowB += srcDescPtr->strides.hStride;
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }

        // Color Jitter without fused output-layout toggle (NHWC -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp32u alignedLength = bufferLength & ~47;

            Rpp8s *srcPtrRow, *dstPtrRow;
            srcPtrRow = srcPtrChannel;
            dstPtrRow = dstPtrChannel;

            for(int i = 0; i < roiPtr->xywhROI.roiHeight; i++)
            {
                Rpp8s *srcPtrTemp, *dstPtrTemp;
                srcPtrTemp = srcPtrRow;
                dstPtrTemp = dstPtrRow;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount+=48)
                {
                    __m128 p[12];

                    rpp_simd_load(rpp_load48_i8pkd3_to_f32pln3, srcPtrTemp, p);    // simd loads
                    compute_color_jitter_48_host(p, pCtm);    // color_jitter adjustment
                    rpp_simd_store(rpp_store48_f32pln3_to_i8pkd3, dstPtrTemp, p);    // simd stores

                    srcPtrTemp += 48;
                    dstPtrTemp += 48;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount+=3)
                {
                    Rpp32f srcPtrTempI8[3];
                    srcPtrTempI8[0] = (Rpp32f)srcPtrTemp[0] + 128;
                    srcPtrTempI8[1] = (Rpp32f)srcPtrTemp[1] + 128;
                    srcPtrTempI8[2] = (Rpp32f)srcPtrTemp[2] + 128;

                    dstPtrTemp[0] = (Rpp8s) RPPPIXELCHECKI8(std::round(ctm[0] * srcPtrTempI8[0] + ctm[1] * srcPtrTempI8[1] + ctm[2] * srcPtrTempI8[2] + ctm[3]) - 128);
                    dstPtrTemp[1] = (Rpp8s) RPPPIXELCHECKI8(std::round(ctm[4] * srcPtrTempI8[0] + ctm[5] * srcPtrTempI8[1] + ctm[6] * srcPtrTempI8[2] + ctm[7]) - 128);
                    dstPtrTemp[2] = (Rpp8s) RPPPIXELCHECKI8(std::round(ctm[8] * srcPtrTempI8[0] + ctm[9] * srcPtrTempI8[1] + ctm[10] * srcPtrTempI8[2] + ctm[11]) - 128);

                    srcPtrTemp += 3;
                    dstPtrTemp += 3;
                }

                srcPtrRow += srcDescPtr->strides.hStride;
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }

        // Color Jitter without fused output-layout toggle (NCHW -> NCHW)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp32u alignedLength = bufferLength & ~47;

            Rpp8s *srcPtrRowR, *srcPtrRowG, *srcPtrRowB, *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
            srcPtrRowR = srcPtrChannel;
            srcPtrRowG = srcPtrRowR + srcDescPtr->strides.cStride;
            srcPtrRowB = srcPtrRowG + srcDescPtr->strides.cStride;
            dstPtrRowR = dstPtrChannel;
            dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
            dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;

            for(int i = 0; i < roiPtr->xywhROI.roiHeight; i++)
            {
                Rpp8s *srcPtrTempR, *srcPtrTempG, *srcPtrTempB, *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
                srcPtrTempR = srcPtrRowR;
                srcPtrTempG = srcPtrRowG;
                srcPtrTempB = srcPtrRowB;
                dstPtrTempR = dstPtrRowR;
                dstPtrTempG = dstPtrRowG;
                dstPtrTempB = dstPtrRowB;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount+=16)
                {
                    __m128 p[12];

                    rpp_simd_load(rpp_load48_i8pln3_to_f32pln3, srcPtrTempR, srcPtrTempG, srcPtrTempB, p);    // simd loads
                    compute_color_jitter_48_host(p, pCtm);    // color_jitter adjustment
                    rpp_simd_store(rpp_store48_f32pln3_to_i8pln3, dstPtrTempR, dstPtrTempG, dstPtrTempB, p);    // simd stores

                    srcPtrTempR += 16;
                    srcPtrTempG += 16;
                    srcPtrTempB += 16;
                    dstPtrTempR += 16;
                    dstPtrTempG += 16;
                    dstPtrTempB += 16;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    Rpp32f srcPtrTempI8[3];
                    srcPtrTempI8[0] = (Rpp32f)*srcPtrTempR + 128;
                    srcPtrTempI8[1] = (Rpp32f)*srcPtrTempG + 128;
                    srcPtrTempI8[2] = (Rpp32f)*srcPtrTempB + 128;

                    *dstPtrTempR = (Rpp8s) RPPPIXELCHECKI8(std::round(ctm[0] * srcPtrTempI8[0] + ctm[1] * srcPtrTempI8[1] + ctm[2] * srcPtrTempI8[2] + ctm[3]) - 128);
                    *dstPtrTempG = (Rpp8s) RPPPIXELCHECKI8(std::round(ctm[4] * srcPtrTempI8[0] + ctm[5] * srcPtrTempI8[1] + ctm[6] * srcPtrTempI8[2] + ctm[7]) - 128);
                    *dstPtrTempB = (Rpp8s) RPPPIXELCHECKI8(std::round(ctm[8] * srcPtrTempI8[0] + ctm[9] * srcPtrTempI8[1] + ctm[10] * srcPtrTempI8[2] + ctm[11]) - 128);

                    srcPtrTempR++;
                    srcPtrTempG++;
                    srcPtrTempB++;
                    dstPtrTempR++;
                    dstPtrTempG++;
                    dstPtrTempB++;
                }

                srcPtrRowR += srcDescPtr->strides.hStride;
                srcPtrRowG += srcDescPtr->strides.hStride;
                srcPtrRowB += srcDescPtr->strides.hStride;
                dstPtrRowR += dstDescPtr->strides.hStride;
                dstPtrRowG += dstDescPtr->strides.hStride;
                dstPtrRowB += dstDescPtr->strides.hStride;
            }
        }
    }

    return RPP_SUCCESS;
}

#endif // HOST_TENSOR_AUGMENTATIONS_HPP