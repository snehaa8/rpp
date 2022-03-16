#include "rppdefs.h"
#include "cpu/rpp_cpu_simd.hpp"
#include "cpu/rpp_cpu_common.hpp"

RppStatus flip_u8_u8_host_tensor(Rpp8u *srcPtr,
                                 RpptDescPtr srcDescPtr,
                                 Rpp8u *dstPtr,
                                 RpptDescPtr dstDescPtr,
                                 Rpp32u *horizontalTensor,
                                 Rpp32u *verticalTensor,
                                 RpptROIPtr roiTensorPtrSrc,
                                 RpptRoiType roiType,
                                 RppLayoutParams layoutParams)
{
    RpptROI roiDefault = {0, 0, (Rpp32s)srcDescPtr->w, (Rpp32s)srcDescPtr->h};

    omp_set_dynamic(0);
#pragma omp parallel for num_threads(dstDescPtr->n)
    for(int batchCount = 0; batchCount < dstDescPtr->n; batchCount++)
    {
        RpptROI roi;
        RpptROIPtr roiPtrInput = &roiTensorPtrSrc[batchCount];
        compute_roi_validation_host(roiPtrInput, &roi, &roiDefault, roiType);

        Rpp32u horizontalFlag = horizontalTensor[batchCount];
        Rpp32u verticalFlag = verticalTensor[batchCount];
        
        Rpp8u *srcPtrImage, *dstPtrImage;
        srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;

        Rpp32u bufferLength = roi.xywhROI.roiWidth * layoutParams.bufferMultiplier;
        Rpp8u *srcPtrChannel, *dstPtrChannel;
        dstPtrChannel = dstPtrImage;

        Rpp32u alignedLength = (bufferLength / 48) * 48;
        Rpp32u vectorIncrement = 48;
        Rpp32u vectorIncrementPerChannel = 16;
        int colPtrMirror = 0, rowPtrMirror = 0;
        auto load48FnPkdPln = &rpp_load48_u8pkd3_to_f32pln3_avx;
        auto load48FnPlnPln = &rpp_load48_u8pln3_to_f32pln3_avx;
        auto load16Fn = &rpp_load16_u8_to_f32_avx;

        if(horizontalFlag == 0 && verticalFlag == 0)
        {
            srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi.xywhROI.xy.x  * layoutParams.bufferMultiplier);
            rowPtrMirror = 0;
            colPtrMirror = 0;
        }
        else if(horizontalFlag == 1 && verticalFlag == 1)
        {
            srcPtrChannel = srcPtrImage + ((roi.xywhROI.xy.y + roi.xywhROI.roiHeight - 1) * srcDescPtr->strides.hStride) + ((roi.xywhROI.xy.x + roi.xywhROI.roiWidth) * layoutParams.bufferMultiplier);
            rowPtrMirror = 1;
            colPtrMirror = 1;
            load48FnPkdPln = &rpp_load48_u8pkd3_to_f32pln3_mirror_avx;
            load48FnPlnPln = &rpp_load48_u8pln3_to_f32pln3_mirror_avx;
            load16Fn = &rpp_load16_u8_to_f32_mirror_avx;
        }
        else if(horizontalFlag == 1)
        {
            srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + ((roi.xywhROI.xy.x + roi.xywhROI.roiWidth) * layoutParams.bufferMultiplier);
            colPtrMirror = 1;
            load48FnPkdPln = &rpp_load48_u8pkd3_to_f32pln3_mirror_avx;
            load48FnPlnPln = &rpp_load48_u8pln3_to_f32pln3_mirror_avx;
            load16Fn = &rpp_load16_u8_to_f32_mirror_avx;
        }
        else if(verticalFlag == 1)
        {
            srcPtrChannel = srcPtrImage + ((roi.xywhROI.xy.y + roi.xywhROI.roiHeight - 1) * srcDescPtr->strides.hStride) + (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);
            rowPtrMirror = 1;
        }
        
        // flip without fused output-layout toggle (NHWC -> NHWC or NCHW -> NCHW and  horizontalflag = 0 and verticalflag = 0)
        if ((horizontalFlag == 0) && (verticalFlag == 0) && (srcDescPtr->layout == dstDescPtr->layout))
        {
            for(int c = 0; c < layoutParams.channelParam; c++)
            {
                Rpp8u *srcPtrRow, *dstPtrRow;
                srcPtrRow = srcPtrChannel;
                dstPtrRow = dstPtrChannel;

                for(int i = 0; i < roi.xywhROI.roiHeight; i++)
                {
                    memcpy(dstPtrRow, srcPtrRow, bufferLength);
                    srcPtrRow += srcDescPtr->strides.hStride;
                    dstPtrRow += dstDescPtr->strides.hStride;
                }

                srcPtrChannel += srcDescPtr->strides.cStride;
                dstPtrChannel += dstDescPtr->strides.cStride;
            }
        }

        // flip with fused output-layout toggle (NHWC -> NCHW)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp8u *srcPtrRow, *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
            srcPtrRow = srcPtrChannel;
            dstPtrRowR = dstPtrChannel;
            dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
            dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;

            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp8u *srcPtrTemp, *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
                srcPtrTemp = srcPtrRow;
                dstPtrTempR = dstPtrRowR;
                dstPtrTempG = dstPtrRowG;
                dstPtrTempB = dstPtrRowB;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement)
                {
                    __m256 p[6];
                    srcPtrTemp -=  (vectorIncrement * colPtrMirror);
                    
                    rpp_simd_load(load48FnPkdPln, srcPtrTemp, p);     // simd loads
                    rpp_simd_store(rpp_store48_f32pln3_to_u8pln3_avx, dstPtrTempR, dstPtrTempG, dstPtrTempB, p);    // simd stores
                    
                    srcPtrTemp += (vectorIncrement * (1 - colPtrMirror));
                    dstPtrTempR += vectorIncrementPerChannel;
                    dstPtrTempG += vectorIncrementPerChannel;
                    dstPtrTempB += vectorIncrementPerChannel;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount += 3)
                {
                    srcPtrTemp -=  (3 * colPtrMirror);
                    *dstPtrTempR = (Rpp8u) RPPPIXELCHECK((Rpp32f) (srcPtrTemp[0]));
                    *dstPtrTempG = (Rpp8u) RPPPIXELCHECK((Rpp32f) (srcPtrTemp[1]));
                    *dstPtrTempB = (Rpp8u) RPPPIXELCHECK((Rpp32f) (srcPtrTemp[2]));

                    srcPtrTemp += (3 * (1 - colPtrMirror));
                    dstPtrTempR++;
                    dstPtrTempG++;
                    dstPtrTempB++;
                }

                if(rowPtrMirror)
                    srcPtrRow -= srcDescPtr->strides.hStride;
                else
                    srcPtrRow += srcDescPtr->strides.hStride;
                dstPtrRowR += dstDescPtr->strides.hStride;
                dstPtrRowG += dstDescPtr->strides.hStride;
                dstPtrRowB += dstDescPtr->strides.hStride;
            }
        }

        // flip with fused output-layout toggle (NCHW -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp8u *srcPtrRowR, *srcPtrRowG, *srcPtrRowB, *dstPtrRow;
            srcPtrRowR = srcPtrChannel;
            srcPtrRowG = srcPtrRowR + srcDescPtr->strides.cStride;
            srcPtrRowB = srcPtrRowG + srcDescPtr->strides.cStride;
            dstPtrRow = dstPtrChannel;

            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp8u *srcPtrTempR, *srcPtrTempG, *srcPtrTempB, *dstPtrTemp;
                srcPtrTempR = srcPtrRowR;
                srcPtrTempG = srcPtrRowG;
                srcPtrTempB = srcPtrRowB;
                dstPtrTemp = dstPtrRow;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m256 p[6];
                    srcPtrTempR -= (vectorIncrementPerChannel * colPtrMirror);
                    srcPtrTempG -= (vectorIncrementPerChannel * colPtrMirror);
                    srcPtrTempB -= (vectorIncrementPerChannel * colPtrMirror);

                    rpp_simd_load(load48FnPlnPln, srcPtrTempR, srcPtrTempG, srcPtrTempB, p);    // simd loads
                    rpp_simd_store(rpp_store48_f32pln3_to_u8pkd3_avx, dstPtrTemp, p);    // simd stores

                    srcPtrTempR += (vectorIncrementPerChannel * (1 - colPtrMirror));
                    srcPtrTempG += (vectorIncrementPerChannel * (1 - colPtrMirror));
                    srcPtrTempB += (vectorIncrementPerChannel * (1 - colPtrMirror));
                    dstPtrTemp += vectorIncrement;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    srcPtrTempR -= (colPtrMirror);
                    srcPtrTempG -= (colPtrMirror);
                    srcPtrTempB -= (colPtrMirror);

                    dstPtrTemp[0] = (Rpp8u) RPPPIXELCHECK((Rpp32f) (*srcPtrTempR));
                    dstPtrTemp[1] = (Rpp8u) RPPPIXELCHECK((Rpp32f) (*srcPtrTempG));
                    dstPtrTemp[2] = (Rpp8u) RPPPIXELCHECK((Rpp32f) (*srcPtrTempB));

                    srcPtrTempR += ((1 - colPtrMirror));
                    srcPtrTempG += ((1 - colPtrMirror));
                    srcPtrTempB += ((1 - colPtrMirror));
                    dstPtrTemp += 3;
                }

                if(rowPtrMirror)
                {
                    srcPtrRowR -= srcDescPtr->strides.hStride;
                    srcPtrRowG -= srcDescPtr->strides.hStride;
                    srcPtrRowB -= srcDescPtr->strides.hStride;
                }
                else
                {
                    srcPtrRowR += srcDescPtr->strides.hStride;
                    srcPtrRowG += srcDescPtr->strides.hStride;
                    srcPtrRowB += srcDescPtr->strides.hStride;
                }
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }

        // flip without fused output-layout toggle (NHWC -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp8u *srcPtrRow, *dstPtrRow;
            srcPtrRow = srcPtrChannel;
            dstPtrRow = dstPtrChannel;

            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp8u *srcPtrTemp, *dstPtrTemp;
                srcPtrTemp = srcPtrRow;
                dstPtrTemp = dstPtrRow;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement)
                {
                    __m256 p[6];
                    srcPtrTemp -= (vectorIncrement * colPtrMirror);

                    rpp_simd_load(load48FnPkdPln, srcPtrTemp, p);    // simd loads
                    rpp_simd_store(rpp_store48_f32pln3_to_u8pkd3_avx, dstPtrTemp, p);    // simd stores

                    srcPtrTemp += (vectorIncrement * (1 - colPtrMirror));
                    dstPtrTemp += vectorIncrement;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount+=3)
                {
                    srcPtrTemp -= (3 * colPtrMirror);

                    dstPtrTemp[0] = (Rpp8u) RPPPIXELCHECK((Rpp32f) (srcPtrTemp[0]));
                    dstPtrTemp[1] = (Rpp8u) RPPPIXELCHECK((Rpp32f) (srcPtrTemp[1]));
                    dstPtrTemp[2] = (Rpp8u) RPPPIXELCHECK((Rpp32f) (srcPtrTemp[2]));

                    srcPtrTemp += (3 * (1 - colPtrMirror));
                    dstPtrTemp += 3;
                }

                if(rowPtrMirror)
                {
                    srcPtrRow -= srcDescPtr->strides.hStride;
                }
                else
                {
                    srcPtrRow += srcDescPtr->strides.hStride;
                }
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }

        // flip without fused output-layout toggle (NHWC -> NHWC or NCHW -> NCHW)
        else
        {
            Rpp32u alignedLength = (bufferLength / 16) * 16;
            for(int c = 0; c < layoutParams.channelParam; c++)
            {
                Rpp8u *srcPtrRow, *dstPtrRow;
                srcPtrRow = srcPtrChannel;
                dstPtrRow = dstPtrChannel;

                for(int i = 0; i < roi.xywhROI.roiHeight; i++)
                {
                    Rpp8u *srcPtrTemp, *dstPtrTemp;
                    srcPtrTemp = srcPtrRow;
                    dstPtrTemp = dstPtrRow;

                    int vectorLoopCount = 0;
                    for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                    {
                        __m256 p[2];

                        srcPtrTemp -= (vectorIncrementPerChannel * colPtrMirror);

                        rpp_simd_load(load16Fn, srcPtrTemp, p);    // simd loads
                        rpp_simd_store(rpp_store16_f32_to_u8_avx, dstPtrTemp, p);    // simd stores

                        srcPtrTemp += (vectorIncrementPerChannel * (1 - colPtrMirror));
                        dstPtrTemp += vectorIncrementPerChannel;
                    }
                    for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                    {
                        srcPtrTemp -= colPtrMirror;

                        *dstPtrTemp = (Rpp8u) RPPPIXELCHECK((Rpp32f) (*srcPtrTemp));

                        srcPtrTemp += (1 - colPtrMirror);
                        dstPtrTemp++;
                    }
                    if(rowPtrMirror)
                        srcPtrRow -= srcDescPtr->strides.hStride;
                    else
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


// RppStatus flip_f32_f32_host_tensor(Rpp32f *srcPtr,
//                                    RpptDescPtr srcDescPtr,
//                                    Rpp32f *dstPtr,
//                                    RpptDescPtr dstDescPtr,
//                                    Rpp32u *horizontalTensor,
//                                    Rpp32u *verticalTensor,
//                                    RpptROIPtr roiTensorPtrSrc,
//                                    RpptRoiType roiType,
//                                    RppLayoutParams layoutParams)
// {
//     RpptROI roiDefault = {0, 0, (Rpp32s)srcDescPtr->w, (Rpp32s)srcDescPtr->h};

//     omp_set_dynamic(0);
// #pragma omp parallel for num_threads(dstDescPtr->n)
//     for(int batchCount = 0; batchCount < dstDescPtr->n; batchCount++)
//     {
//         RpptROI roi;
//         RpptROIPtr roiPtrInput = &roiTensorPtrSrc[batchCount];
//         compute_roi_validation_host(roiPtrInput, &roi, &roiDefault, roiType);

//         Rpp32u flipAxis = flipAxisTensor[batchCount];
        
//         Rpp32f *srcPtrImage, *dstPtrImage;
//         srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
//         dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;

//         Rpp32u bufferLength = roi.xywhROI.roiWidth * layoutParams.bufferMultiplier;

//         Rpp32f *srcPtrChannel, *dstPtrChannel;
//         srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);
//         dstPtrChannel = dstPtrImage;

//         Rpp32u alignedLength = (bufferLength / 24) * 24;
//         Rpp32u vectorIncrement = 24;
//         Rpp32u vectorIncrementPerChannel = 8;

//         // flip with fused output-layout toggle (NHWC -> NCHW)
//         if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
//         {
//             Rpp32f *srcPtrRow, *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
//             srcPtrRow = srcPtrChannel;
//             dstPtrRowR = dstPtrChannel;
//             dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
//             dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;

//             for(int i = 0; i < roi.xywhROI.roiHeight; i++)
//             {
//                 Rpp32f *srcPtrTemp, *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
//                 srcPtrTemp = srcPtrRow;
//                 dstPtrTempR = dstPtrRowR;
//                 dstPtrTempG = dstPtrRowG;
//                 dstPtrTempB = dstPtrRowB;

//                 int vectorLoopCount = 0;
//                 for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement)
//                 {
//                     __m256 p[3];
//                     rpp_simd_load(rpp_load24_f32pkd3_to_f32pln3_avx, srcPtrTemp, p);    // simd loads
//                     rpp_simd_store(rpp_store24_f32pln3_to_f32pln3_avx, dstPtrTempR, dstPtrTempG, dstPtrTempB, p);    // simd stores

//                     srcPtrTemp += vectorIncrement;
//                     dstPtrTempR += vectorIncrementPerChannel;
//                     dstPtrTempG += vectorIncrementPerChannel;
//                     dstPtrTempB += vectorIncrementPerChannel;
//                 }
//                 for (; vectorLoopCount < bufferLength; vectorLoopCount += 3)
//                 {
//                     *dstPtrTempR = RPPPIXELCHECKF32(srcPtrTemp[0]);
//                     *dstPtrTempG = RPPPIXELCHECKF32(srcPtrTemp[1]);
//                     *dstPtrTempB = RPPPIXELCHECKF32(srcPtrTemp[2]);

//                     srcPtrTemp += 3;
//                     dstPtrTempR++;
//                     dstPtrTempG++;
//                     dstPtrTempB++;
//                 }

//                 srcPtrRow += srcDescPtr->strides.hStride;
//                 dstPtrRowR += dstDescPtr->strides.hStride;
//                 dstPtrRowG += dstDescPtr->strides.hStride;
//                 dstPtrRowB += dstDescPtr->strides.hStride;
//             }
//         }

//         // flip with fused output-layout toggle (NCHW -> NHWC)
//         else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
//         {
//             Rpp32f *srcPtrRowR, *srcPtrRowG, *srcPtrRowB, *dstPtrRow;
//             srcPtrRowR = srcPtrChannel;
//             srcPtrRowG = srcPtrRowR + srcDescPtr->strides.cStride;
//             srcPtrRowB = srcPtrRowG + srcDescPtr->strides.cStride;
//             dstPtrRow = dstPtrChannel;

//             for(int i = 0; i < roi.xywhROI.roiHeight; i++)
//             {
//                 Rpp32f *srcPtrTempR, *srcPtrTempG, *srcPtrTempB, *dstPtrTemp;
//                 srcPtrTempR = srcPtrRowR;
//                 srcPtrTempG = srcPtrRowG;
//                 srcPtrTempB = srcPtrRowB;
//                 dstPtrTemp = dstPtrRow;

//                 int vectorLoopCount = 0;
//                 for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
//                 {
//                     __m256 p[3];
//                     rpp_simd_load(rpp_load24_f32pln3_to_f32pln3_avx, srcPtrTempR, srcPtrTempG, srcPtrTempB, p);     // simd loads
//                     rpp_simd_store(rpp_store24_f32pln3_to_f32pkd3_avx, dstPtrTemp, p);    // simd stores

//                     srcPtrTempR += vectorIncrementPerChannel;
//                     srcPtrTempG += vectorIncrementPerChannel;
//                     srcPtrTempB += vectorIncrementPerChannel;
//                     dstPtrTemp += vectorIncrement;
//                 }
//                 for (; vectorLoopCount < bufferLength; vectorLoopCount++)
//                 {
//                     dstPtrTemp[0] = RPPPIXELCHECKF32((*srcPtrTempR));
//                     dstPtrTemp[1] = RPPPIXELCHECKF32((*srcPtrTempG));
//                     dstPtrTemp[2] = RPPPIXELCHECKF32((*srcPtrTempB));

//                     srcPtrTempR++;
//                     srcPtrTempG++;
//                     srcPtrTempB++;
//                     dstPtrTemp += 3;
//                 }

//                 srcPtrRowR += srcDescPtr->strides.hStride;
//                 srcPtrRowG += srcDescPtr->strides.hStride;
//                 srcPtrRowB += srcDescPtr->strides.hStride;
//                 dstPtrRow += dstDescPtr->strides.hStride;
//             }
//         }

//         // flip without fused output-layout toggle (NHWC -> NHWC or NCHW -> NCHW)
//         else
//         {
//             Rpp32u alignedLength = (bufferLength / 8) * 8;
//             for(int c = 0; c < layoutParams.channelParam; c++)
//             {
//                 Rpp32f *srcPtrRow, *dstPtrRow;
//                 srcPtrRow = srcPtrChannel;
//                 dstPtrRow = dstPtrChannel;

//                 for(int i = 0; i < roi.xywhROI.roiHeight; i++)
//                 {
//                     Rpp32f *srcPtrTemp, *dstPtrTemp;
//                     srcPtrTemp = srcPtrRow;
//                     dstPtrTemp = dstPtrRow;

//                     int vectorLoopCount = 0;
//                     for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
//                     {
//                         __m256 p[1];

//                         rpp_simd_load(rpp_load8_f32_to_f32_avx, srcPtrTemp, p);    // simd loads
//                         rpp_simd_store(rpp_store8_f32_to_f32_avx, dstPtrTemp, p);    // simd stores

//                         srcPtrTemp += vectorIncrementPerChannel;
//                         dstPtrTemp += vectorIncrementPerChannel;
//                     }
//                     for (; vectorLoopCount < bufferLength; vectorLoopCount++)
//                     {
//                         *dstPtrTemp = RPPPIXELCHECKF32((*srcPtrTemp));

//                         srcPtrTemp++;
//                         dstPtrTemp++;
//                     }
//                     srcPtrRow += srcDescPtr->strides.hStride;
//                     dstPtrRow += dstDescPtr->strides.hStride;
//                 }
//                 srcPtrChannel += srcDescPtr->strides.cStride;
//                 dstPtrChannel += dstDescPtr->strides.cStride;
//             }
//         }
//     }

//     return RPP_SUCCESS;
// }

// RppStatus flip_f16_f16_host_tensor(Rpp16f *srcPtr,
//                                    RpptDescPtr srcDescPtr,
//                                    Rpp16f *dstPtr,
//                                    RpptDescPtr dstDescPtr,
//                                    Rpp32u *horizontalTensor,
//                                    Rpp32u *verticalTensor,
//                                    RpptROIPtr roiTensorPtrSrc,
//                                    RpptRoiType roiType,
//                                    RppLayoutParams layoutParams)
// {
//     RpptROI roiDefault = {0, 0, (Rpp32s)srcDescPtr->w, (Rpp32s)srcDescPtr->h};

//     omp_set_dynamic(0);
// #pragma omp parallel for num_threads(dstDescPtr->n)
//     for(int batchCount = 0; batchCount < dstDescPtr->n; batchCount++)
//     {
//         RpptROI roi;
//         RpptROIPtr roiPtrInput = &roiTensorPtrSrc[batchCount];
//         compute_roi_validation_host(roiPtrInput, &roi, &roiDefault, roiType);

//         Rpp32u flipAxis = flipAxisTensor[batchCount];
        
//         Rpp16f *srcPtrImage, *dstPtrImage;
//         srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
//         dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;

//         Rpp32u bufferLength = roi.xywhROI.roiWidth * layoutParams.bufferMultiplier;

//         Rpp16f *srcPtrChannel, *dstPtrChannel;
//         srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);
//         dstPtrChannel = dstPtrImage;

//         Rpp32u alignedLength = (bufferLength / 24) * 24;
//         Rpp32u vectorIncrement = 24;
//         Rpp32u vectorIncrementPerChannel = 8;

//         // flip with fused output-layout toggle (NHWC -> NCHW)
//         if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
//         {
//             Rpp16f *srcPtrRow, *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
//             srcPtrRow = srcPtrChannel;
//             dstPtrRowR = dstPtrChannel;
//             dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
//             dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;

//             for(int i = 0; i < roi.xywhROI.roiHeight; i++)
//             {
//                 Rpp16f *srcPtrTemp, *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
//                 srcPtrTemp = srcPtrRow;
//                 dstPtrTempR = dstPtrRowR;
//                 dstPtrTempG = dstPtrRowG;
//                 dstPtrTempB = dstPtrRowB;

//                 int vectorLoopCount = 0;
//                 for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement)
//                 {
//                     Rpp32f srcPtrTemp_ps[24];
//                     Rpp32f dstPtrTempR_ps[8], dstPtrTempG_ps[8], dstPtrTempB_ps[8];

//                     for(int cnt = 0; cnt < vectorIncrement; cnt++)
//                         srcPtrTemp_ps[cnt] = (Rpp32f) srcPtrTemp[cnt];

//                     __m256 p[3];

//                     rpp_simd_load(rpp_load24_f32pkd3_to_f32pln3_avx, srcPtrTemp_ps, p);     // simd loads
//                     rpp_simd_store(rpp_store24_f32pln3_to_f32pln3_avx, dstPtrTempR_ps, dstPtrTempG_ps, dstPtrTempB_ps, p);    // simd stores

//                     for(int cnt = 0; cnt < vectorIncrementPerChannel; cnt++)
//                     {
//                         dstPtrTempR[cnt] = (Rpp16f) dstPtrTempR_ps[cnt];
//                         dstPtrTempG[cnt] = (Rpp16f) dstPtrTempG_ps[cnt];
//                         dstPtrTempB[cnt] = (Rpp16f) dstPtrTempB_ps[cnt];
//                     }

//                     srcPtrTemp += vectorIncrement;
//                     dstPtrTempR += vectorIncrementPerChannel;
//                     dstPtrTempG += vectorIncrementPerChannel;
//                     dstPtrTempB += vectorIncrementPerChannel;
//                 }
//                 for (; vectorLoopCount < bufferLength; vectorLoopCount += 3)
//                 {
//                     *dstPtrTempR = (Rpp16f) RPPPIXELCHECKF32((Rpp32f)srcPtrTemp[0]);
//                     *dstPtrTempG = (Rpp16f) RPPPIXELCHECKF32((Rpp32f)srcPtrTemp[1]);
//                     *dstPtrTempB = (Rpp16f) RPPPIXELCHECKF32((Rpp32f)srcPtrTemp[2]);

//                     srcPtrTemp += 3;
//                     dstPtrTempR++;
//                     dstPtrTempG++;
//                     dstPtrTempB++;
//                 }

//                 srcPtrRow += srcDescPtr->strides.hStride;
//                 dstPtrRowR += dstDescPtr->strides.hStride;
//                 dstPtrRowG += dstDescPtr->strides.hStride;
//                 dstPtrRowB += dstDescPtr->strides.hStride;
//             }
//         }

//         // flip with fused output-layout toggle (NCHW -> NHWC)
//         else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
//         {
//             Rpp16f *srcPtrRowR, *srcPtrRowG, *srcPtrRowB, *dstPtrRow;
//             srcPtrRowR = srcPtrChannel;
//             srcPtrRowG = srcPtrRowR + srcDescPtr->strides.cStride;
//             srcPtrRowB = srcPtrRowG + srcDescPtr->strides.cStride;
//             dstPtrRow = dstPtrChannel;

//             for(int i = 0; i < roi.xywhROI.roiHeight; i++)
//             {
//                 Rpp16f *srcPtrTempR, *srcPtrTempG, *srcPtrTempB, *dstPtrTemp;
//                 srcPtrTempR = srcPtrRowR;
//                 srcPtrTempG = srcPtrRowG;
//                 srcPtrTempB = srcPtrRowB;
//                 dstPtrTemp = dstPtrRow;

//                 int vectorLoopCount = 0;
//                 for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
//                 {
//                     Rpp32f srcPtrTempR_ps[8], srcPtrTempG_ps[8], srcPtrTempB_ps[8];
//                     Rpp32f dstPtrTemp_ps[25];
//                     for(int cnt = 0; cnt < vectorIncrementPerChannel; cnt++)
//                     {
//                         srcPtrTempR_ps[cnt] = (Rpp32f) srcPtrTempR[cnt];
//                         srcPtrTempG_ps[cnt] = (Rpp32f) srcPtrTempG[cnt];
//                         srcPtrTempB_ps[cnt] = (Rpp32f) srcPtrTempB[cnt];
//                     }

//                     __m256 p[3];
//                     rpp_simd_load(rpp_load24_f32pln3_to_f32pln3_avx, srcPtrTempR_ps, srcPtrTempG_ps, srcPtrTempB_ps, p);    // simd loads
//                     rpp_simd_store(rpp_store24_f32pln3_to_f32pkd3_avx, dstPtrTemp_ps, p);    // simd stores

//                     for(int cnt = 0; cnt < vectorIncrement; cnt++)
//                         dstPtrTemp[cnt] = (Rpp16f) dstPtrTemp_ps[cnt];

//                     srcPtrTempR += vectorIncrementPerChannel;
//                     srcPtrTempG += vectorIncrementPerChannel;
//                     srcPtrTempB += vectorIncrementPerChannel;
//                     dstPtrTemp += vectorIncrement;
//                 }
//                 for (; vectorLoopCount < bufferLength; vectorLoopCount++)
//                 {
//                     dstPtrTemp[0] = (Rpp16f) RPPPIXELCHECKF32((Rpp32f)(*srcPtrTempR));
//                     dstPtrTemp[1] = (Rpp16f) RPPPIXELCHECKF32((Rpp32f)(*srcPtrTempG));
//                     dstPtrTemp[2] = (Rpp16f) RPPPIXELCHECKF32((Rpp32f)(*srcPtrTempB));

//                     srcPtrTempR++;
//                     srcPtrTempG++;
//                     srcPtrTempB++;
//                     dstPtrTemp += 3;
//                 }

//                 srcPtrRowR += srcDescPtr->strides.hStride;
//                 srcPtrRowG += srcDescPtr->strides.hStride;
//                 srcPtrRowB += srcDescPtr->strides.hStride;
//                 dstPtrRow += dstDescPtr->strides.hStride;
//             }
//         }

//         // flip without fused output-layout toggle (NHWC -> NHWC or NCHW -> NCHW)
//         else
//         {
//             Rpp32u alignedLength = (bufferLength / 8) * 8;
//             for(int c = 0; c < layoutParams.channelParam; c++)
//             {
//                 Rpp16f *srcPtrRow, *dstPtrRow;
//                 srcPtrRow = srcPtrChannel;
//                 dstPtrRow = dstPtrChannel;

//                 for(int i = 0; i < roi.xywhROI.roiHeight; i++)
//                 {
//                     Rpp16f *srcPtrTemp, *dstPtrTemp;
//                     srcPtrTemp = srcPtrRow;
//                     dstPtrTemp = dstPtrRow;

//                     int vectorLoopCount = 0;
//                     for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
//                     {
//                         Rpp32f srcPtrTemp_ps[8], dstPtrTemp_ps[8];

//                         for(int cnt = 0; cnt < vectorIncrementPerChannel; cnt++)
//                             srcPtrTemp_ps[cnt] = (Rpp16f) srcPtrTemp[cnt];

//                         __m256 p[1];

//                         rpp_simd_load(rpp_load8_f32_to_f32_avx, srcPtrTemp_ps, p);    // simd loads
//                         rpp_simd_store(rpp_store8_f32_to_f32_avx, dstPtrTemp_ps, p);    // simd stores

//                         for(int cnt = 0; cnt < vectorIncrementPerChannel; cnt++)
//                             dstPtrTemp[cnt] = (Rpp16f) dstPtrTemp_ps[cnt];

//                         srcPtrTemp += vectorIncrementPerChannel;
//                         dstPtrTemp += vectorIncrementPerChannel;
//                     }
//                     for (; vectorLoopCount < bufferLength; vectorLoopCount++)
//                     {
//                         *dstPtrTemp = (Rpp16f) RPPPIXELCHECKF32((Rpp32f)(*srcPtrTemp));

//                         srcPtrTemp++;
//                         dstPtrTemp++;
//                     }
//                     srcPtrRow += srcDescPtr->strides.hStride;
//                     dstPtrRow += dstDescPtr->strides.hStride;
//                 }
//                 srcPtrChannel += srcDescPtr->strides.cStride;
//                 dstPtrChannel += dstDescPtr->strides.cStride;
//             }
//         }
//     }

//     return RPP_SUCCESS;
// }

RppStatus flip_i8_i8_host_tensor(Rpp8s *srcPtr,
                                 RpptDescPtr srcDescPtr,
                                 Rpp8s *dstPtr,
                                 RpptDescPtr dstDescPtr,
                                 Rpp32u *horizontalTensor,
                                 Rpp32u *verticalTensor,
                                 RpptROIPtr roiTensorPtrSrc,
                                 RpptRoiType roiType,
                                 RppLayoutParams layoutParams)
{
    RpptROI roiDefault = {0, 0, (Rpp32s)srcDescPtr->w, (Rpp32s)srcDescPtr->h};

    omp_set_dynamic(0);
#pragma omp parallel for num_threads(dstDescPtr->n)
    for(int batchCount = 0; batchCount < dstDescPtr->n; batchCount++)
    {
        RpptROI roi;
        RpptROIPtr roiPtrInput = &roiTensorPtrSrc[batchCount];
        compute_roi_validation_host(roiPtrInput, &roi, &roiDefault, roiType);

        Rpp32u horizontalFlag = horizontalTensor[batchCount];
        Rpp32u verticalFlag = verticalTensor[batchCount];
        
        Rpp8s *srcPtrImage, *dstPtrImage;
        srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;

        Rpp32s bufferLength = roi.xywhROI.roiWidth * layoutParams.bufferMultiplier;
        Rpp8s *srcPtrChannel, *dstPtrChannel;
        dstPtrChannel = dstPtrImage;

        Rpp32u alignedLength = (bufferLength / 48) * 48;
        Rpp32u vectorIncrement = 48;
        Rpp32u vectorIncrementPerChannel = 16;
        int colPtrMirror = 0, rowPtrMirror = 0;
        auto load48FnPkdPln = &rpp_load48_i8pkd3_to_f32pln3_avx;
        auto load48FnPlnPln = &rpp_load48_i8pln3_to_f32pln3_avx;
        auto load16Fn = &rpp_load16_i8_to_f32_avx;

        if(horizontalFlag == 0 && verticalFlag == 0)
        {
            srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi.xywhROI.xy.x  * layoutParams.bufferMultiplier);
            rowPtrMirror = 0;
            colPtrMirror = 0;
        }
        else if(horizontalFlag == 1 && verticalFlag == 1)
        {
            srcPtrChannel = srcPtrImage + ((roi.xywhROI.xy.y + roi.xywhROI.roiHeight - 1) * srcDescPtr->strides.hStride) + ((roi.xywhROI.xy.x + roi.xywhROI.roiWidth) * layoutParams.bufferMultiplier);
            rowPtrMirror = 1;
            colPtrMirror = 1;
            load48FnPkdPln = &rpp_load48_i8pkd3_to_f32pln3_mirror_avx;
            load48FnPlnPln = &rpp_load48_i8pln3_to_f32pln3_mirror_avx;
            load16Fn = &rpp_load16_i8_to_f32_mirror_avx;
        }
        else if(horizontalFlag == 1)
        {
            srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + ((roi.xywhROI.xy.x + roi.xywhROI.roiWidth) * layoutParams.bufferMultiplier);
            colPtrMirror = 1;
            load48FnPkdPln = &rpp_load48_i8pkd3_to_f32pln3_mirror_avx;
            load48FnPlnPln = &rpp_load48_i8pln3_to_f32pln3_mirror_avx;
            load16Fn = &rpp_load16_i8_to_f32_mirror_avx;
        }
        else if(verticalFlag == 1)
        {
            srcPtrChannel = srcPtrImage + ((roi.xywhROI.xy.y + roi.xywhROI.roiHeight - 1) * srcDescPtr->strides.hStride) + (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);
            rowPtrMirror = 1;
        }
        
        // flip without fused output-layout toggle (NHWC -> NHWC or NCHW -> NCHW and  horizontalflag = 0 and verticalflag = 0)
        if ((horizontalFlag == 0) && (verticalFlag == 0) && (srcDescPtr->layout == dstDescPtr->layout))
        {
            for(int c = 0; c < layoutParams.channelParam; c++)
            {
                Rpp8s *srcPtrRow, *dstPtrRow;
                srcPtrRow = srcPtrChannel;
                dstPtrRow = dstPtrChannel;

                for(int i = 0; i < roi.xywhROI.roiHeight; i++)
                {
                    memcpy(dstPtrRow, srcPtrRow, bufferLength);
                    srcPtrRow += srcDescPtr->strides.hStride;
                    dstPtrRow += dstDescPtr->strides.hStride;
                }

                srcPtrChannel += srcDescPtr->strides.cStride;
                dstPtrChannel += dstDescPtr->strides.cStride;
            }
        }

        // flip with fused output-layout toggle (NHWC -> NCHW)
        if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp8s *srcPtrRow, *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
            srcPtrRow = srcPtrChannel;
            dstPtrRowR = dstPtrChannel;
            dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
            dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;

            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp8s *srcPtrTemp, *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
                srcPtrTemp = srcPtrRow;
                dstPtrTempR = dstPtrRowR;
                dstPtrTempG = dstPtrRowG;
                dstPtrTempB = dstPtrRowB;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement)
                {
                    __m256 p[6];
                    srcPtrTemp -=  (vectorIncrement * colPtrMirror);
                    
                    rpp_simd_load(load48FnPkdPln, srcPtrTemp, p);     // simd loads
                    rpp_simd_store(rpp_store48_f32pln3_to_i8pln3_avx, dstPtrTempR, dstPtrTempG, dstPtrTempB, p);    // simd stores
                    
                    srcPtrTemp += (vectorIncrement * (1 - colPtrMirror));
                    dstPtrTempR += vectorIncrementPerChannel;
                    dstPtrTempG += vectorIncrementPerChannel;
                    dstPtrTempB += vectorIncrementPerChannel;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount += 3)
                {
                    srcPtrTemp -=  (3 * colPtrMirror);
                    *dstPtrTempR = (Rpp8s) RPPPIXELCHECKI8((Rpp32f) (srcPtrTemp[0]));
                    *dstPtrTempG = (Rpp8s) RPPPIXELCHECKI8((Rpp32f) (srcPtrTemp[1]));
                    *dstPtrTempB = (Rpp8s) RPPPIXELCHECKI8((Rpp32f) (srcPtrTemp[2]));

                    srcPtrTemp += (3 * (1 - colPtrMirror));
                    dstPtrTempR++;
                    dstPtrTempG++;
                    dstPtrTempB++;
                }

                if(rowPtrMirror)
                    srcPtrRow -= srcDescPtr->strides.hStride;
                else
                    srcPtrRow += srcDescPtr->strides.hStride;
                dstPtrRowR += dstDescPtr->strides.hStride;
                dstPtrRowG += dstDescPtr->strides.hStride;
                dstPtrRowB += dstDescPtr->strides.hStride;
            }
        }

        // flip with fused output-layout toggle (NCHW -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp8s *srcPtrRowR, *srcPtrRowG, *srcPtrRowB, *dstPtrRow;
            srcPtrRowR = srcPtrChannel;
            srcPtrRowG = srcPtrRowR + srcDescPtr->strides.cStride;
            srcPtrRowB = srcPtrRowG + srcDescPtr->strides.cStride;
            dstPtrRow = dstPtrChannel;

            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp8s *srcPtrTempR, *srcPtrTempG, *srcPtrTempB, *dstPtrTemp;
                srcPtrTempR = srcPtrRowR;
                srcPtrTempG = srcPtrRowG;
                srcPtrTempB = srcPtrRowB;
                dstPtrTemp = dstPtrRow;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m256 p[6];
                    srcPtrTempR -= (vectorIncrementPerChannel * colPtrMirror);
                    srcPtrTempG -= (vectorIncrementPerChannel * colPtrMirror);
                    srcPtrTempB -= (vectorIncrementPerChannel * colPtrMirror);

                    rpp_simd_load(load48FnPlnPln, srcPtrTempR, srcPtrTempG, srcPtrTempB, p);    // simd loads
                    rpp_simd_store(rpp_store48_f32pln3_to_i8pkd3_avx, dstPtrTemp, p);    // simd stores

                    srcPtrTempR += (vectorIncrementPerChannel * (1 - colPtrMirror));
                    srcPtrTempG += (vectorIncrementPerChannel * (1 - colPtrMirror));
                    srcPtrTempB += (vectorIncrementPerChannel * (1 - colPtrMirror));
                    dstPtrTemp += vectorIncrement;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    srcPtrTempR -= (colPtrMirror);
                    srcPtrTempG -= (colPtrMirror);
                    srcPtrTempB -= (colPtrMirror);

                    dstPtrTemp[0] = (Rpp8s) RPPPIXELCHECKI8((Rpp32f) (*srcPtrTempR));
                    dstPtrTemp[1] = (Rpp8s) RPPPIXELCHECKI8((Rpp32f) (*srcPtrTempG));
                    dstPtrTemp[2] = (Rpp8s) RPPPIXELCHECKI8((Rpp32f) (*srcPtrTempB));

                    srcPtrTempR += ((1 - colPtrMirror));
                    srcPtrTempG += ((1 - colPtrMirror));
                    srcPtrTempB += ((1 - colPtrMirror));
                    dstPtrTemp += 3;
                }

                if(rowPtrMirror)
                {
                    srcPtrRowR -= srcDescPtr->strides.hStride;
                    srcPtrRowG -= srcDescPtr->strides.hStride;
                    srcPtrRowB -= srcDescPtr->strides.hStride;
                }
                else
                {
                    srcPtrRowR += srcDescPtr->strides.hStride;
                    srcPtrRowG += srcDescPtr->strides.hStride;
                    srcPtrRowB += srcDescPtr->strides.hStride;
                }
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }

        // flip without fused output-layout toggle (NHWC -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp8s *srcPtrRow, *dstPtrRow;
            srcPtrRow = srcPtrChannel;
            dstPtrRow = dstPtrChannel;

            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp8s *srcPtrTemp, *dstPtrTemp;
                srcPtrTemp = srcPtrRow;
                dstPtrTemp = dstPtrRow;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement)
                {
                    __m256 p[6];
                    srcPtrTemp -= (vectorIncrement * colPtrMirror);

                    rpp_simd_load(load48FnPkdPln, srcPtrTemp, p);    // simd loads
                    rpp_simd_store(rpp_store48_f32pln3_to_i8pkd3_avx, dstPtrTemp, p);    // simd stores

                    srcPtrTemp += (vectorIncrement * (1 - colPtrMirror));
                    dstPtrTemp += vectorIncrement;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount+=3)
                {
                    srcPtrTemp -= (3 * colPtrMirror);

                    dstPtrTemp[0] = (Rpp8s) RPPPIXELCHECKI8((Rpp32f) (srcPtrTemp[0]));
                    dstPtrTemp[1] = (Rpp8s) RPPPIXELCHECKI8((Rpp32f) (srcPtrTemp[1]));
                    dstPtrTemp[2] = (Rpp8s) RPPPIXELCHECKI8((Rpp32f) (srcPtrTemp[2]));

                    srcPtrTemp += (3 * (1 - colPtrMirror));
                    dstPtrTemp += 3;
                }

                if(rowPtrMirror)
                    srcPtrRow -= srcDescPtr->strides.hStride;
                else
                    srcPtrRow += srcDescPtr->strides.hStride;

                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }

        // flip without fused output-layout toggle (NHWC -> NHWC or NCHW -> NCHW)
        else
        {
            Rpp32u alignedLength = (bufferLength / 16) * 16;
            for(int c = 0; c < layoutParams.channelParam; c++)
            {
                Rpp8s *srcPtrRow, *dstPtrRow;
                srcPtrRow = srcPtrChannel;
                dstPtrRow = dstPtrChannel;

                for(int i = 0; i < roi.xywhROI.roiHeight; i++)
                {
                    Rpp8s *srcPtrTemp, *dstPtrTemp;
                    srcPtrTemp = srcPtrRow;
                    dstPtrTemp = dstPtrRow;

                    int vectorLoopCount = 0;
                    for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                    {
                        __m256 p[2];

                        srcPtrTemp -= (vectorIncrementPerChannel * colPtrMirror);

                        rpp_simd_load(load16Fn, srcPtrTemp, p);    // simd loads
                        rpp_simd_store(rpp_store16_f32_to_i8_avx, dstPtrTemp, p);    // simd stores

                        srcPtrTemp += (vectorIncrementPerChannel * (1 - colPtrMirror));
                        dstPtrTemp += vectorIncrementPerChannel;
                    }
                    for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                    {
                        srcPtrTemp -= colPtrMirror;
                        *dstPtrTemp = (Rpp8s) RPPPIXELCHECKI8((Rpp32f) (*srcPtrTemp));

                        srcPtrTemp += (1 - colPtrMirror);
                        dstPtrTemp++;
                    }
                    if(rowPtrMirror)
                        srcPtrRow -= srcDescPtr->strides.hStride;
                    else
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