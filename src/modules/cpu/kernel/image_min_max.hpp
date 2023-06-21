#include "rppdefs.h"
#include "rpp_cpu_simd.hpp"
#include "rpp_cpu_common.hpp"

RppStatus image_min_max_u8_u8_host_tensor(Rpp8u *srcPtr,
                                          RpptDescPtr srcDescPtr,
                                          Rpp8u *imageMinMaxArr,
                                          Rpp32u imageMinMaxArrLength,
                                          RpptROIPtr roiTensorPtrSrc,
                                          RpptRoiType roiType,
                                          RppLayoutParams layoutParams)
{
    RpptROI roiDefault = {0, 0, (Rpp32s)srcDescPtr->w, (Rpp32s)srcDescPtr->h};

    omp_set_dynamic(0);
#pragma omp parallel for num_threads(srcDescPtr->n)
    for(int batchCount = 0; batchCount < srcDescPtr->n; batchCount++)
    {
        RpptROI roi;
        RpptROIPtr roiPtrInput = &roiTensorPtrSrc[batchCount];
        compute_roi_validation_host(roiPtrInput, &roi, &roiDefault, roiType);

        Rpp8u *srcPtrImage;
        srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;

        Rpp32u bufferLength = roi.xywhROI.roiWidth * layoutParams.bufferMultiplier;

        Rpp8u *srcPtrChannel;
        srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);

        Rpp32u alignedLength = (bufferLength / 192) * 192;
        Rpp32u vectorIncrement = 192;
        Rpp32u vectorIncrementPerChannel = 64;

        // Image Min Max 1 channel (NCHW)
        if ((srcDescPtr->c == 1) && (srcDescPtr->layout == RpptLayout::NCHW))
        {
            alignedLength = (bufferLength / 64) * 64;
            vectorIncrement = 64;
            Rpp8u min = 255;
            Rpp8u max = 0;
            Rpp8u minAvx[32], maxAvx[32];

            for(int c = 0; c < layoutParams.channelParam; c++)
            {
                Rpp8u *srcPtrRow;
                srcPtrRow = srcPtrChannel;
#if __AVX2__
                __m256i pMin = _mm256_set1_epi8((char)255);
                __m256i pMax = _mm256_setzero_si256();
#endif
                for(int i = 0; i < roi.xywhROI.roiHeight; i++)
                {
                    Rpp8u *srcPtrTemp;
                    srcPtrTemp = srcPtrRow;

                    int vectorLoopCount = 0;
#if __AVX2__
                    for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement)
                    {
                        __m256i p1[2];
                        rpp_simd_load(rpp_load64_u8_avx, srcPtrTemp, p1);
                        compute_min_max_64_host(p1, &pMin, &pMax);

                        srcPtrTemp += vectorIncrement;
                    }
#endif
                    for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                    {
                        min = std::min(*srcPtrTemp, min);
                        max = std::max(*srcPtrTemp, max);
                        srcPtrTemp++;
                    }
                    srcPtrRow += srcDescPtr->strides.hStride;
                }
                srcPtrChannel += srcDescPtr->strides.cStride;
#if __AVX2__
                rpp_simd_store(rpp_store32_u8_to_u8_avx, minAvx, &pMin);
                rpp_simd_store(rpp_store32_u8_to_u8_avx, maxAvx, &pMax);

                for(int i=0;i<16;i++)
                {
                    min = std::min(std::min(minAvx[i], minAvx[i + 16]), min);
                    max = std::max(std::max(maxAvx[i], maxAvx[i + 16]), max);
                }
#endif
            }
            imageMinMaxArr[batchCount*2] = min;
            imageMinMaxArr[(batchCount*2) + 1] = max;
        }
        // Image Min Max 3 channel (NCHW)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp8u min = 255, minR = 255, minG = 255, minB = 255;
            Rpp8u max = 0, maxR = 0, maxG = 0, maxB = 0;
            Rpp8u minAvxR[32], maxAvxR[32], minAvxG[32], maxAvxG[32], minAvxB[32], maxAvxB[32];

            for(int c = 0; c < layoutParams.channelParam; c++)
            {
                Rpp8u *srcPtrRowR, *srcPtrRowG, *srcPtrRowB, *dstPtrRow;
                srcPtrRowR = srcPtrChannel;
                srcPtrRowG = srcPtrRowR + srcDescPtr->strides.cStride;
                srcPtrRowB = srcPtrRowG + srcDescPtr->strides.cStride;
#if __AVX2__
                __m256i pMinR = _mm256_set1_epi8((char)255);
                __m256i pMaxR = _mm256_setzero_si256();
				__m256i pMinG = pMinR;
                __m256i pMaxG = pMaxR;
				__m256i pMinB = pMinR;
                __m256i pMaxB = pMaxR;
#endif
                for(int i = 0; i < roi.xywhROI.roiHeight; i++)
                {
                    Rpp8u *srcPtrTempR, *srcPtrTempG, *srcPtrTempB;
                    srcPtrTempR = srcPtrRowR;
                    srcPtrTempG = srcPtrRowG;
                    srcPtrTempB = srcPtrRowB;

                    int vectorLoopCount = 0;
#if __AVX2__
                    for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement)
                    {
                        __m256i p[6];
                        rpp_simd_load(rpp_load192_u8_avx, srcPtrTempR, srcPtrTempG, srcPtrTempB, p);
                        compute_min_max_192_host(p, &pMinR, &pMaxR, &pMinG, &pMaxG, &pMinB, &pMaxB);

                        srcPtrTempR += vectorIncrementPerChannel;
                        srcPtrTempG += vectorIncrementPerChannel;
                        srcPtrTempB += vectorIncrementPerChannel;
                    }
#endif
                    for (; vectorLoopCount < bufferLength; vectorLoopCount += 3)
                    {
                        minR = std::min(*srcPtrTempR, minR);
                        maxR = std::max(*srcPtrTempR, maxR);
						minG = std::min(*srcPtrTempG, minG);
                        maxG = std::max(*srcPtrTempG, maxG);
						minB = std::min(*srcPtrTempB, minB);
                        maxB = std::max(*srcPtrTempB, maxB);
                        srcPtrTempR++;
                        srcPtrTempG++;
                        srcPtrTempB++;
                    }
                    srcPtrRowR += srcDescPtr->strides.hStride;
                    srcPtrRowG += srcDescPtr->strides.hStride;
                    srcPtrRowB += srcDescPtr->strides.hStride;
                }
                srcPtrChannel += srcDescPtr->strides.cStride;
#if __AVX2__
                rpp_simd_store(rpp_store32_u8_to_u8_avx, minAvxR, &pMinR);
                rpp_simd_store(rpp_store32_u8_to_u8_avx, maxAvxR, &pMaxR);
				rpp_simd_store(rpp_store32_u8_to_u8_avx, minAvxG, &pMinG);
                rpp_simd_store(rpp_store32_u8_to_u8_avx, maxAvxG, &pMaxG);
				rpp_simd_store(rpp_store32_u8_to_u8_avx, minAvxB, &pMinB);
                rpp_simd_store(rpp_store32_u8_to_u8_avx, maxAvxB, &pMaxB);

                for(int i=0;i<16;i++)
                {
                    minR = std::min(std::min(minAvxR[i], minAvxR[i + 16]), minR);
                    maxR = std::max(std::max(maxAvxR[i], maxAvxR[i + 16]), maxR);
					minG = std::min(std::min(minAvxG[i], minAvxG[i + 16]), minG);
                    maxG = std::max(std::max(maxAvxG[i], maxAvxG[i + 16]), maxG);
					minB = std::min(std::min(minAvxB[i], minAvxB[i + 16]), minB);
                    maxB = std::max(std::max(maxAvxB[i], maxAvxB[i + 16]), maxB);
                }
#endif
            }
			min = std::min(std::min(minR, minG), minB);
			max = std::max(std::max(maxR, maxG), maxB);
            imageMinMaxArr[batchCount*8] = minR;
            imageMinMaxArr[(batchCount*8) + 1] = maxR;
			imageMinMaxArr[(batchCount*8) + 2] = minG;
            imageMinMaxArr[(batchCount*8) + 3] = maxG;
			imageMinMaxArr[(batchCount*8) + 4] = minB;
            imageMinMaxArr[(batchCount*8) + 5] = maxB;
			imageMinMaxArr[(batchCount*8) + 6] = min;
            imageMinMaxArr[(batchCount*8) + 7] = max;
        }
    }
    printf("\n Min_Max output\n");
    for(int i=0;i<imageMinMaxArrLength;i++)
        printf("imageMinMaxArr[%d]: %d\n", i, (int)imageMinMaxArr[i]);

    return RPP_SUCCESS;
}
