#include "rppdefs.h"
#include "cpu/rpp_cpu_simd.hpp"
#include "cpu/rpp_cpu_common.hpp"

RppStatus resize_bilinear_u8_u8_host_tensor(Rpp8u *srcPtr,
                                            RpptDescPtr srcDescPtr,
                                            Rpp8u *dstPtr,
                                            RpptDescPtr dstDescPtr,
                                            RpptImagePatchPtr dstImgSize,
                                            RpptROIPtr roiTensorPtrSrc,
                                            RpptRoiType roiType,
                                            RppLayoutParams srcLayoutParams)
{
    RpptROI roiDefault = {0, 0, (Rpp32s)srcDescPtr->w, (Rpp32s)srcDescPtr->h};

omp_set_dynamic(0);
#pragma omp parallel for num_threads(dstDescPtr->n)
    for(int batchCount = 0; batchCount < dstDescPtr->n; batchCount++)
    {
        RpptROI roi;
        RpptROIPtr roiPtrInput = &roiTensorPtrSrc[batchCount];
        compute_roi_validation_host(roiPtrInput, &roi, &roiDefault, roiType);

        compute_dst_size_cap_host(&dstImgSize[batchCount], dstDescPtr);     // Check if the dstImgSize exceeds dst buffer size
        Rpp32f wRatio = ((Rpp32f)(roi.xywhROI.roiWidth)) / ((Rpp32f)(dstImgSize[batchCount].width));
        Rpp32f hRatio = ((Rpp32f)(roi.xywhROI.roiHeight)) / ((Rpp32f)(dstImgSize[batchCount].height));
        Rpp32u heightLimit = roi.xywhROI.roiHeight - 1;
        Rpp32u widthLimit = (roi.xywhROI.roiWidth - 1) * srcDescPtr->strides.wStride;
        Rpp32s kernelSize = 2;
        Rpp32f kernelRadius = 1.0f; // kernelSize / 2
        Rpp32s noOfCoeffs = 4; // kernelSize * kernelSize
        Rpp32f hOffset = (hRatio - 1) * 0.5f - kernelRadius;
        Rpp32f wOffset = (wRatio - 1) * 0.5f - kernelRadius;
        Rpp32s vectorIncrementPerChannel = 8;
        Rpp32s vectorIncrementPkd = 24;

        Rpp8u *srcPtrChannel, *dstPtrChannel, *srcPtrImage, *dstPtrImage;
        srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;
        srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi.xywhROI.xy.x * srcLayoutParams.bufferMultiplier);
        dstPtrChannel = dstPtrImage;

        Rpp32u alignedLength = dstImgSize[batchCount].width & ~7;   // Align dst width to process 8 dst pixels per iteration
        __m256 pWRatio = _mm256_set1_ps(wRatio);
        __m256 pWOffset = _mm256_set1_ps(wOffset);
        __m256 pWidthLimit = _mm256_set1_ps((Rpp32f)widthLimit);
        __m256 pWeightParams[noOfCoeffs], pBilinearCoeffs[noOfCoeffs], pDstLoc;
        Rpp32f weightParams[noOfCoeffs], bilinearCoeffs[noOfCoeffs];
        Rpp32s srcLocationColumnArray[8] = {0};     // Since 8 dst pixels are processed per iteration
        Rpp32s srcLocationRow, srcLocationColumn;

        // Resize with fused output-layout toggle (NHWC -> NCHW)
        if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp8u *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
            dstPtrRowR = dstPtrChannel;
            dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
            dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;

            for(int dstLocRow = 0; dstLocRow < dstImgSize[batchCount].height; dstLocRow++)
            {
                Rpp8u *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
                dstPtrTempR = dstPtrRowR;
                dstPtrTempG = dstPtrRowG;
                dstPtrTempB = dstPtrRowB;

                Rpp8u *srcRowPtrsForInterp[2];     // kernelSize(2)
                compute_resize_bilinear_src_loc_and_weights(dstLocRow, hRatio, srcLocationRow, &weightParams[0], hOffset); // Compute the src row location correspoding to the dst row location
                compute_src_row_ptrs_for_bilinear_interpolation(srcRowPtrsForInterp, srcPtrChannel, srcLocationRow, heightLimit, srcDescPtr); // Compute the src row pointers for interpolation
                pWeightParams[0] = _mm256_set1_ps(weightParams[0]);
                pWeightParams[1]  = _mm256_set1_ps(weightParams[1]);
                pDstLoc = avx_pDstLocInit;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m256 pSrc[12], pDst[3];
                    __m256 negativeLocMask;
                    compute_resize_bilinear_src_loc_and_weights_avx(pDstLoc, pWRatio, srcLocationColumnArray, &pWeightParams[2], negativeLocMask, pWOffset, true);   // Compute the src col location correspoding to the dst col location
                    if(srcLocationColumnArray[7] >= widthLimit) break;
                    compute_bilinear_coefficients_avx(pWeightParams, pBilinearCoeffs);  // Compute Bilinear coefficients
                    rpp_simd_load(rpp_bilinear_load_u8pkd3_to_f32pln3_avx, srcRowPtrsForInterp, srcLocationColumnArray, pSrc, negativeLocMask);  // Load input pixels required for bilinear interpolation
                    compute_bilinear_interpolation_3c_avx(pSrc, pBilinearCoeffs, pDst); // Compute Bilinear interpolation
                    rpp_simd_store(rpp_store12_f32pln3_to_u8pln3_avx, dstPtrTempR, dstPtrTempG, dstPtrTempB, pDst); // Store dst pixels
                    dstPtrTempR += vectorIncrementPerChannel;
                    dstPtrTempG += vectorIncrementPerChannel;
                    dstPtrTempB += vectorIncrementPerChannel;
                }
                for (; vectorLoopCount < dstImgSize[batchCount].width; vectorLoopCount++)
                {
                    compute_resize_bilinear_src_loc_and_weights(vectorLoopCount, wRatio, srcLocationColumn, &weightParams[2], wOffset, srcDescPtr->strides.wStride); // Compute the src col location correspoding to the dst col location
                    compute_bilinear_coefficients(weightParams, bilinearCoeffs);    // Compute Bilinear coefficients
                    compute_bilinear_interpolation_3c_pkd(srcRowPtrsForInterp, srcLocationColumn, widthLimit, bilinearCoeffs, dstPtrTempR, dstPtrTempG, dstPtrTempB);   // Compute Bilinear interpolation
                    dstPtrTempR++;
                    dstPtrTempG++;
                    dstPtrTempB++;
                }
                dstPtrRowR += dstDescPtr->strides.hStride;
                dstPtrRowG += dstDescPtr->strides.hStride;
                dstPtrRowB += dstDescPtr->strides.hStride;
            }
        }

        // Resize with fused output-layout toggle (NCHW -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp8u *dstPtrRow;
            dstPtrRow = dstPtrChannel;

            for(int dstLocRow = 0; dstLocRow < dstImgSize[batchCount].height; dstLocRow++)
            {
                Rpp8u *dstPtrTemp;
                dstPtrTemp = dstPtrRow;

                Rpp8u *srcRowPtrsForInterp[6];     // kernelSize(2) * numOfPlanes(3)
                compute_resize_bilinear_src_loc_and_weights(dstLocRow, hRatio, srcLocationRow, &weightParams[0], hOffset);  // Compute the src row location correspoding to the dst row location
                compute_src_row_ptrs_for_bilinear_interpolation_pln(srcRowPtrsForInterp, srcPtrChannel, srcLocationRow, heightLimit, srcDescPtr); // Compute the src row pointers for interpolation
                pWeightParams[0] = _mm256_set1_ps(weightParams[0]);
                pWeightParams[1]  = _mm256_set1_ps(weightParams[1]);
                pDstLoc = avx_pDstLocInit;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m256 pSrc[12], pDst[3];
                    __m256 negativeLocMask;
                    compute_resize_bilinear_src_loc_and_weights_avx(pDstLoc, pWRatio, srcLocationColumnArray, &pWeightParams[2], negativeLocMask, pWOffset); // Compute the src col location correspoding to the dst col location
                    if(srcLocationColumnArray[7] >= widthLimit) break;
                    compute_bilinear_coefficients_avx(pWeightParams, pBilinearCoeffs);      // Compute Bilinear coefficients
                    rpp_simd_load(rpp_bilinear_load_u8pln1_to_f32pln1_avx, &srcRowPtrsForInterp[0], srcLocationColumnArray, pSrc, negativeLocMask);  // Load input pixels required for bilinear interpolation
                    rpp_simd_load(rpp_bilinear_load_u8pln1_to_f32pln1_avx, &srcRowPtrsForInterp[2], srcLocationColumnArray, &pSrc[4], negativeLocMask);
                    rpp_simd_load(rpp_bilinear_load_u8pln1_to_f32pln1_avx, &srcRowPtrsForInterp[4], srcLocationColumnArray, &pSrc[8], negativeLocMask);
                    compute_bilinear_interpolation_3c_avx(pSrc, pBilinearCoeffs, pDst);     // Compute Bilinear interpolation
                    rpp_simd_store(rpp_store12_f32pln3_to_u8pkd3_avx, dstPtrTemp, pDst);    // Store dst pixels
                    dstPtrTemp += vectorIncrementPkd;
                }
                for (; vectorLoopCount < dstImgSize[batchCount].width; vectorLoopCount++)
                {
                    compute_resize_bilinear_src_loc_and_weights(vectorLoopCount, wRatio, srcLocationColumn, &weightParams[2], wOffset);  // Compute the src col location correspoding to the dst col location
                    compute_bilinear_coefficients(weightParams, bilinearCoeffs);    // Compute Bilinear coefficients
                    compute_bilinear_interpolation_3c_pkd(srcRowPtrsForInterp, srcLocationColumn, widthLimit, bilinearCoeffs, &dstPtrTemp[0], &dstPtrTemp[1], &dstPtrTemp[2]);  // Compute Bilinear interpolation
                    dstPtrTemp += dstDescPtr->c;
                }
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }

        // Resize without fused output-layout toggle (NHWC -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp8u *dstPtrRow;
            dstPtrRow = dstPtrChannel;
            for(int dstLocRow = 0; dstLocRow < dstImgSize[batchCount].height; dstLocRow++)
            {
                Rpp8u *dstPtrTemp;
                dstPtrTemp = dstPtrRow;

                Rpp8u *srcRowPtrsForInterp[2];     // kernelSize(2)
                compute_resize_bilinear_src_loc_and_weights(dstLocRow, hRatio, srcLocationRow, &weightParams[0], hOffset);  // Compute the src row location correspoding to the dst row location
                compute_src_row_ptrs_for_bilinear_interpolation(srcRowPtrsForInterp, srcPtrChannel, srcLocationRow, heightLimit, srcDescPtr); // Compute the src row pointers for interpolation
                pWeightParams[0] = _mm256_set1_ps(weightParams[0]);
                pWeightParams[1]  = _mm256_set1_ps(weightParams[1]);
                pDstLoc = avx_pDstLocInit;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m256 pSrc[12], pDst[3];
                    __m256 negativeLocMask;
                    compute_resize_bilinear_src_loc_and_weights_avx(pDstLoc, pWRatio, srcLocationColumnArray, &pWeightParams[2], negativeLocMask, pWOffset, true);   // Compute the src col location correspoding to the dst col location
                    if(srcLocationColumnArray[7] >= widthLimit) break;
                    compute_bilinear_coefficients_avx(pWeightParams, pBilinearCoeffs);      // Compute Bilinear coefficients
                    rpp_simd_load(rpp_bilinear_load_u8pkd3_to_f32pln3_avx, srcRowPtrsForInterp, srcLocationColumnArray, pSrc, negativeLocMask);  // Load input pixels required for bilinear interpolation
                    compute_bilinear_interpolation_3c_avx(pSrc, pBilinearCoeffs, pDst);     // Compute Bilinear interpolation
                    rpp_simd_store(rpp_store12_f32pln3_to_u8pkd3_avx, dstPtrTemp, pDst);   // Store dst pixels
                    dstPtrTemp += vectorIncrementPkd;
                }
                for (; vectorLoopCount < dstImgSize[batchCount].width; vectorLoopCount++)
                {
                    compute_resize_bilinear_src_loc_and_weights(vectorLoopCount, wRatio, srcLocationColumn, &weightParams[2], wOffset, srcDescPtr->strides.wStride); // Compute the src col location correspoding to the dst col location
                    compute_bilinear_coefficients(weightParams, bilinearCoeffs);    // Compute Bilinear coefficients
                    compute_bilinear_interpolation_3c_pkd(srcRowPtrsForInterp, srcLocationColumn, widthLimit, bilinearCoeffs, &dstPtrTemp[0], &dstPtrTemp[1], &dstPtrTemp[2]);  // Compute Bilinear interpolation

                    dstPtrTemp += dstDescPtr->c;
                }
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }

        // Resize without fused output-layout toggle (NCHW -> NCHW for 1 channel and 3 channel)
        else if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp8u *dstPtrRow;
            dstPtrRow = dstPtrChannel;
            for(int dstLocRow = 0; dstLocRow < dstImgSize[batchCount].height; dstLocRow++)
            {
                Rpp8u *dstPtrTemp;
                dstPtrTemp = dstPtrRow;
                Rpp8u *srcRowPtrsForInterp[6];     // kernelSize(2) * numOfPlanes(3)
                compute_resize_bilinear_src_loc_and_weights(dstLocRow, hRatio, srcLocationRow, &weightParams[0], hOffset);  // Compute the src row location correspoding to the dst row location
                compute_src_row_ptrs_for_bilinear_interpolation_pln(srcRowPtrsForInterp, srcPtrChannel, srcLocationRow, heightLimit, srcDescPtr); // Compute the src row pointers for interpolation
                pWeightParams[0] = _mm256_set1_ps(weightParams[0]);
                pWeightParams[1]  = _mm256_set1_ps(weightParams[1]);
                pDstLoc = avx_pDstLocInit;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    Rpp8u *dstPtrTempChn;
                    dstPtrTempChn = dstPtrTemp;
                    __m256 pSrc[4], pDst;
                    __m256 negativeLocMask;
                    compute_resize_bilinear_src_loc_and_weights_avx(pDstLoc, pWRatio, srcLocationColumnArray, &pWeightParams[2], negativeLocMask, pWOffset); // Compute the src col location correspoding to the dst col location
                    if(srcLocationColumnArray[7] >= widthLimit) break;
                    compute_bilinear_coefficients_avx(pWeightParams, pBilinearCoeffs);          // Compute Bilinear coefficients

                    for (int c = 0; c < dstDescPtr->c; c++)
                    {
                        rpp_simd_load(rpp_bilinear_load_u8pln1_to_f32pln1_avx, &srcRowPtrsForInterp[c * kernelSize], srcLocationColumnArray, pSrc, negativeLocMask); // Load input pixels required for bilinear interpolation
                        compute_bilinear_interpolation_1c_avx(pSrc, pBilinearCoeffs, pDst);     // Compute Bilinear interpolation
                        rpp_simd_store(rpp_store4_f32pln1_to_u8pln1_avx, dstPtrTempChn, pDst);  // Store dst pixels
                        dstPtrTempChn += dstDescPtr->strides.cStride;
                    }
                    dstPtrTemp += vectorIncrementPerChannel;
                }
                for (; vectorLoopCount < dstImgSize[batchCount].width; vectorLoopCount++)
                {
                    Rpp8u *dstPtrTempChn;
                    dstPtrTempChn = dstPtrTemp;
                    compute_resize_bilinear_src_loc_and_weights(vectorLoopCount, wRatio, srcLocationColumn, &weightParams[2], wOffset);  // Compute the src col location correspoding to the dst col location
                    compute_bilinear_coefficients(weightParams, bilinearCoeffs);    // Compute Bilinear coefficients
                    for (int c = 0; c < dstDescPtr->c; c++)
                    {
                        compute_bilinear_interpolation_1c(&srcRowPtrsForInterp[c * kernelSize], widthLimit, srcLocationColumn, bilinearCoeffs, dstPtrTempChn);  // Compute Bilinear interpolation
                        dstPtrTempChn += dstDescPtr->strides.cStride;
                    }
                    dstPtrTemp++;
                }
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }
    }

    return RPP_SUCCESS;
}

RppStatus resize_bilinear_f32_f32_host_tensor(Rpp32f *srcPtr,
                                              RpptDescPtr srcDescPtr,
                                              Rpp32f *dstPtr,
                                              RpptDescPtr dstDescPtr,
                                              RpptImagePatchPtr dstImgSize,
                                              RpptROIPtr roiTensorPtrSrc,
                                              RpptRoiType roiType,
                                              RppLayoutParams srcLayoutParams)
{
    RpptROI roiDefault = {0, 0, (Rpp32s)srcDescPtr->w, (Rpp32s)srcDescPtr->h};

omp_set_dynamic(0);
#pragma omp parallel for num_threads(dstDescPtr->n)
    for(int batchCount = 0; batchCount < dstDescPtr->n; batchCount++)
    {
        RpptROI roi;
        RpptROIPtr roiPtrInput = &roiTensorPtrSrc[batchCount];
        compute_roi_validation_host(roiPtrInput, &roi, &roiDefault, roiType);

        compute_dst_size_cap_host(&dstImgSize[batchCount], dstDescPtr);     // Check if the dstImgSize exceeds dst buffer size
        Rpp32f wRatio = ((Rpp32f)(roi.xywhROI.roiWidth)) / ((Rpp32f)(dstImgSize[batchCount].width));
        Rpp32f hRatio = ((Rpp32f)(roi.xywhROI.roiHeight)) / ((Rpp32f)(dstImgSize[batchCount].height));
        Rpp32u heightLimit = roi.xywhROI.roiHeight - 1;
        Rpp32u widthLimit = (roi.xywhROI.roiWidth - 1) * srcDescPtr->strides.wStride;
        Rpp32s kernelSize = 2;
        Rpp32f kernelRadius = 1.0f; // kernelSize / 2
        Rpp32s noOfCoeffs = 4; // kernelSize * kernelSize
        Rpp32f hOffset = (hRatio - 1) * 0.5f - kernelRadius;
        Rpp32f wOffset = (wRatio - 1) * 0.5f - kernelRadius;
        Rpp32s vectorIncrementPerChannel = 8;
        Rpp32s vectorIncrementPkd = 24;

        Rpp32f *srcPtrChannel, *dstPtrChannel, *srcPtrImage, *dstPtrImage;
        srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;
        srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi.xywhROI.xy.x * srcLayoutParams.bufferMultiplier);
        dstPtrChannel = dstPtrImage;

        Rpp32u alignedLength = dstImgSize[batchCount].width & ~7;   // Align dst width to process 8 dst pixels per iteration
        __m256 pWRatio = _mm256_set1_ps(wRatio);
        __m256 pWOffset = _mm256_set1_ps(wOffset);
        __m256 pWidthLimit = _mm256_set1_ps((Rpp32f)widthLimit);
        __m256 pWeightParams[noOfCoeffs], pBilinearCoeffs[noOfCoeffs], pDstLoc;
        Rpp32f weightParams[noOfCoeffs], bilinearCoeffs[noOfCoeffs];
        Rpp32s srcLocationColumnArray[8] = {0};     // Since 8 dst pixels are processed per iteration
        Rpp32s srcLocationRow, srcLocationColumn;

        // Resize with fused output-layout toggle (NHWC -> NCHW)
        if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp32f *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
            dstPtrRowR = dstPtrChannel;
            dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
            dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;

            for(int dstLocRow = 0; dstLocRow < dstImgSize[batchCount].height; dstLocRow++)
            {
                Rpp32f *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
                dstPtrTempR = dstPtrRowR;
                dstPtrTempG = dstPtrRowG;
                dstPtrTempB = dstPtrRowB;

                Rpp32f *srcRowPtrsForInterp[2];     // kernelSize(2)
                compute_resize_bilinear_src_loc_and_weights(dstLocRow, hRatio, srcLocationRow, &weightParams[0], hOffset);  // Compute the src row location correspoding to the dst row location
                compute_src_row_ptrs_for_bilinear_interpolation(srcRowPtrsForInterp, srcPtrChannel, srcLocationRow, heightLimit, srcDescPtr); // Compute the src row pointers for interpolation
                pWeightParams[0] = _mm256_set1_ps(weightParams[0]);
                pWeightParams[1]  = _mm256_set1_ps(weightParams[1]);
                pDstLoc = avx_pDstLocInit;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m256 pSrc[12], pDst[3];
                    __m256 negativeLocMask;
                    compute_resize_bilinear_src_loc_and_weights_avx(pDstLoc, pWRatio, srcLocationColumnArray, &pWeightParams[2], negativeLocMask, pWOffset, true);   // Compute the src col location correspoding to the dst col location
                    if(srcLocationColumnArray[7] >= widthLimit) break;
                    compute_bilinear_coefficients_avx(pWeightParams, pBilinearCoeffs);                              // Compute Bilinear coefficients
                    rpp_simd_load(rpp_bilinear_load_f32pkd3_to_f32pln3_avx, srcRowPtrsForInterp, srcLocationColumnArray, pSrc, negativeLocMask); // Load input pixels required for bilinear interpolation
                    compute_bilinear_interpolation_3c_avx(pSrc, pBilinearCoeffs, pDst);                             // Compute Bilinear interpolation
                    rpp_simd_store(rpp_store24_f32pln3_to_f32pln3_avx, dstPtrTempR, dstPtrTempG, dstPtrTempB, pDst); // Store dst pixels
                    dstPtrTempR += vectorIncrementPerChannel;
                    dstPtrTempG += vectorIncrementPerChannel;
                    dstPtrTempB += vectorIncrementPerChannel;
                }
                for (; vectorLoopCount < dstImgSize[batchCount].width; vectorLoopCount++)
                {
                    compute_resize_bilinear_src_loc_and_weights(vectorLoopCount, wRatio, srcLocationColumn, &weightParams[2], wOffset, srcDescPtr->strides.wStride); // Compute the src col location correspoding to the dst col location
                    compute_bilinear_coefficients(weightParams, bilinearCoeffs);    // Compute Bilinear coefficients
                    compute_bilinear_interpolation_3c_pkd(srcRowPtrsForInterp, srcLocationColumn, widthLimit, bilinearCoeffs, dstPtrTempR, dstPtrTempG, dstPtrTempB);   // Compute Bilinear interpolation
                    dstPtrTempR++;
                    dstPtrTempG++;
                    dstPtrTempB++;
                }
                dstPtrRowR += dstDescPtr->strides.hStride;
                dstPtrRowG += dstDescPtr->strides.hStride;
                dstPtrRowB += dstDescPtr->strides.hStride;
            }
        }

        // Resize with fused output-layout toggle (NCHW -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp32f *dstPtrRow;
            dstPtrRow = dstPtrChannel;

            for(int dstLocRow = 0; dstLocRow < dstImgSize[batchCount].height; dstLocRow++)
            {
                Rpp32f *dstPtrTemp;
                dstPtrTemp = dstPtrRow;

                Rpp32f *srcRowPtrsForInterp[6];     // kernelSize(2) * numOfPlanes(3)
                compute_resize_bilinear_src_loc_and_weights(dstLocRow, hRatio, srcLocationRow, &weightParams[0], hOffset);  // Compute the src row location correspoding to the dst row location
                compute_src_row_ptrs_for_bilinear_interpolation_pln(srcRowPtrsForInterp, srcPtrChannel, srcLocationRow, heightLimit, srcDescPtr); // Compute the src row pointers for interpolation
                pWeightParams[0] = _mm256_set1_ps(weightParams[0]);
                pWeightParams[1]  = _mm256_set1_ps(weightParams[1]);
                pDstLoc = avx_pDstLocInit;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m256 pSrc[12], pDst[4];
                    __m256 negativeLocMask;
                    compute_resize_bilinear_src_loc_and_weights_avx(pDstLoc, pWRatio, srcLocationColumnArray, &pWeightParams[2], negativeLocMask, pWOffset); // Compute the src col location correspoding to the dst col location
                    if(srcLocationColumnArray[7] >= widthLimit) break;
                    compute_bilinear_coefficients_avx(pWeightParams, pBilinearCoeffs);  // Compute Bilinear coefficients
                    rpp_simd_load(rpp_bilinear_load_f32pln1_to_f32pln1_avx, &srcRowPtrsForInterp[0], srcLocationColumnArray, pSrc, negativeLocMask); // Load input pixels required for bilinear interpolation
                    rpp_simd_load(rpp_bilinear_load_f32pln1_to_f32pln1_avx, &srcRowPtrsForInterp[2], srcLocationColumnArray, &pSrc[4], negativeLocMask);
                    rpp_simd_load(rpp_bilinear_load_f32pln1_to_f32pln1_avx, &srcRowPtrsForInterp[4], srcLocationColumnArray, &pSrc[8], negativeLocMask);
                    compute_bilinear_interpolation_3c_avx(pSrc, pBilinearCoeffs, pDst); // Compute Bilinear interpolation
                    rpp_simd_store(rpp_store24_f32pln3_to_f32pkd3_avx, dstPtrTemp, pDst);   // Store dst pixels
                    dstPtrTemp += vectorIncrementPkd;
                }
                for (; vectorLoopCount < dstImgSize[batchCount].width; vectorLoopCount++)
                {
                    compute_resize_bilinear_src_loc_and_weights(vectorLoopCount, wRatio, srcLocationColumn, &weightParams[2], wOffset);  // Compute the src col location correspoding to the dst col location
                    compute_bilinear_coefficients(weightParams, bilinearCoeffs);    // Compute Bilinear coefficients
                    compute_bilinear_interpolation_3c_pkd(srcRowPtrsForInterp, srcLocationColumn, widthLimit, bilinearCoeffs, &dstPtrTemp[0], &dstPtrTemp[1], &dstPtrTemp[2]);  // Compute Bilinear interpolation
                    dstPtrTemp += dstDescPtr->c;
                }
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }

        // Resize without fused output-layout toggle (NHWC -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp32f *dstPtrRow;
            dstPtrRow = dstPtrChannel;

            for(int dstLocRow = 0; dstLocRow < dstImgSize[batchCount].height; dstLocRow++)
            {
                Rpp32f *dstPtrTemp;
                dstPtrTemp = dstPtrRow;

                Rpp32f *srcRowPtrsForInterp[2];     // kernelSize(2)
                compute_resize_bilinear_src_loc_and_weights(dstLocRow, hRatio, srcLocationRow, &weightParams[0], hOffset);  // Compute the src row location correspoding to the dst row location
                compute_src_row_ptrs_for_bilinear_interpolation(srcRowPtrsForInterp, srcPtrChannel, srcLocationRow, heightLimit, srcDescPtr); // Compute the src row pointers for interpolation
                pWeightParams[0] = _mm256_set1_ps(weightParams[0]);
                pWeightParams[1]  = _mm256_set1_ps(weightParams[1]);
                pDstLoc = avx_pDstLocInit;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m256 pSrc[12], pDst[4];
                    __m256 negativeLocMask;
                    compute_resize_bilinear_src_loc_and_weights_avx(pDstLoc, pWRatio, srcLocationColumnArray, &pWeightParams[2], negativeLocMask, pWOffset, true);   // Compute the src col location correspoding to the dst col location
                    if(srcLocationColumnArray[7] >= widthLimit) break;
                    compute_bilinear_coefficients_avx(pWeightParams, pBilinearCoeffs);  // Compute Bilinear coefficients
                    rpp_simd_load(rpp_bilinear_load_f32pkd3_to_f32pln3_avx, srcRowPtrsForInterp, srcLocationColumnArray, pSrc, negativeLocMask); // Load input pixels required for bilinear interpolation
                    compute_bilinear_interpolation_3c_avx(pSrc, pBilinearCoeffs, pDst); // Compute Bilinear interpolation
                    rpp_simd_store(rpp_store24_f32pln3_to_f32pkd3_avx, dstPtrTemp, pDst);   // Store dst pixels
                    dstPtrTemp += vectorIncrementPkd;
                }
                for (; vectorLoopCount < dstImgSize[batchCount].width; vectorLoopCount++)
                {
                    compute_resize_bilinear_src_loc_and_weights(vectorLoopCount, wRatio, srcLocationColumn, &weightParams[2], wOffset, srcDescPtr->strides.wStride); // Compute the src col location correspoding to the col row location
                    compute_bilinear_coefficients(weightParams, bilinearCoeffs);    // Compute Bilinear coefficients
                    compute_bilinear_interpolation_3c_pkd(srcRowPtrsForInterp, srcLocationColumn, widthLimit, bilinearCoeffs, &dstPtrTemp[0], &dstPtrTemp[1], &dstPtrTemp[2]);  // Compute Bilinear interpolation
                    dstPtrTemp += dstDescPtr->c;
                }
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }

        // Resize without fused output-layout toggle (NCHW -> NCHW for 1 channel and 3 channel)
        else if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp32f *dstPtrRow;
            dstPtrRow = dstPtrChannel;

            for(int dstLocRow = 0; dstLocRow < dstImgSize[batchCount].height; dstLocRow++)
            {
                Rpp32f *dstPtrTemp;
                dstPtrTemp = dstPtrRow;

                Rpp32f *srcRowPtrsForInterp[6];     // kernelSize(2) * numOfPlanes(3)
                compute_resize_bilinear_src_loc_and_weights(dstLocRow, hRatio, srcLocationRow, &weightParams[0], hOffset);  // Compute the src row location correspoding to the dst row location
                compute_src_row_ptrs_for_bilinear_interpolation_pln(srcRowPtrsForInterp, srcPtrChannel, srcLocationRow, heightLimit, srcDescPtr); // Compute the src row pointers for interpolation
                pWeightParams[0] = _mm256_set1_ps(weightParams[0]);
                pWeightParams[1]  = _mm256_set1_ps(weightParams[1]);
                pDstLoc = avx_pDstLocInit;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    Rpp32f *dstPtrTempChn;
                    dstPtrTempChn = dstPtrTemp;
                    __m256 pSrc[4], pDst;
                    __m256 negativeLocMask;
                    compute_resize_bilinear_src_loc_and_weights_avx(pDstLoc, pWRatio, srcLocationColumnArray, &pWeightParams[2], negativeLocMask, pWOffset); // Compute the src col location correspoding to the dst col location
                    if(srcLocationColumnArray[7] >= widthLimit) break;
                    compute_bilinear_coefficients_avx(pWeightParams, pBilinearCoeffs);  // Compute Bilinear coefficients

                    for (int c = 0; c < dstDescPtr->c; c++)
                    {
                        rpp_simd_load(rpp_bilinear_load_f32pln1_to_f32pln1_avx, &srcRowPtrsForInterp[c * kernelSize], srcLocationColumnArray, pSrc, negativeLocMask);    // Load input pixels required for bilinear interpolation
                        compute_bilinear_interpolation_1c_avx(pSrc, pBilinearCoeffs, pDst); // Compute Bilinear interpolation
                        rpp_simd_store(rpp_store8_f32pln1_to_f32pln1_avx, dstPtrTempChn, pDst); // Store dst pixels
                        dstPtrTempChn += dstDescPtr->strides.cStride;
                    }
                    dstPtrTemp += vectorIncrementPerChannel;
                }
                for (; vectorLoopCount < dstImgSize[batchCount].width; vectorLoopCount++)
                {
                    Rpp32f *dstPtrTempChn;
                    dstPtrTempChn = dstPtrTemp;
                    compute_resize_bilinear_src_loc_and_weights(vectorLoopCount, wRatio, srcLocationColumn, &weightParams[2], wOffset);  // Compute the src col location correspoding to the dst col location
                    compute_bilinear_coefficients(weightParams, bilinearCoeffs);    // Compute Bilinear coefficients
                    for (int c = 0; c < dstDescPtr->c; c++)
                    {
                        compute_bilinear_interpolation_1c(&srcRowPtrsForInterp[c * kernelSize], widthLimit, srcLocationColumn, bilinearCoeffs, dstPtrTempChn);  // Compute Bilinear interpolation
                        dstPtrTempChn += dstDescPtr->strides.cStride;
                    }
                    dstPtrTemp++;
                }
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }
    }

    return RPP_SUCCESS;
}

RppStatus resize_bilinear_f16_f16_host_tensor(Rpp16f *srcPtr,
                                              RpptDescPtr srcDescPtr,
                                              Rpp16f *dstPtr,
                                              RpptDescPtr dstDescPtr,
                                              RpptImagePatchPtr dstImgSize,
                                              RpptROIPtr roiTensorPtrSrc,
                                              RpptRoiType roiType,
                                              RppLayoutParams srcLayoutParams)
{
    RpptROI roiDefault = {0, 0, (Rpp32s)srcDescPtr->w, (Rpp32s)srcDescPtr->h};

omp_set_dynamic(0);
#pragma omp parallel for num_threads(dstDescPtr->n)
    for(int batchCount = 0; batchCount < dstDescPtr->n; batchCount++)
    {
        RpptROI roi;
        RpptROIPtr roiPtrInput = &roiTensorPtrSrc[batchCount];
        compute_roi_validation_host(roiPtrInput, &roi, &roiDefault, roiType);

        compute_dst_size_cap_host(&dstImgSize[batchCount], dstDescPtr);     // Check if the dstImgSize exceeds dst buffer size
        Rpp32f wRatio = ((Rpp32f)(roi.xywhROI.roiWidth)) / ((Rpp32f)(dstImgSize[batchCount].width));
        Rpp32f hRatio = ((Rpp32f)(roi.xywhROI.roiHeight)) / ((Rpp32f)(dstImgSize[batchCount].height));
        Rpp32u heightLimit = roi.xywhROI.roiHeight - 1;
        Rpp32u widthLimit = (roi.xywhROI.roiWidth - 1) * srcDescPtr->strides.wStride;
        Rpp32s kernelSize = 2;
        Rpp32f kernelRadius = 1.0f; // kernelSize / 2
        Rpp32s noOfCoeffs = 4; // kernelSize * kernelSize
        Rpp32f hOffset = (hRatio - 1) * 0.5f - kernelRadius;
        Rpp32f wOffset = (wRatio - 1) * 0.5f - kernelRadius;
        Rpp32s vectorIncrementPerChannel = 8;
        Rpp32s vectorIncrementPkd = 24;

        Rpp16f *srcPtrChannel, *dstPtrChannel, *srcPtrImage, *dstPtrImage;
        srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;
        srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi.xywhROI.xy.x * srcLayoutParams.bufferMultiplier);
        dstPtrChannel = dstPtrImage;

        Rpp32u alignedLength = dstImgSize[batchCount].width & ~7;   // Align dst width to process 8 dst pixels per iteration
        __m256 pWRatio = _mm256_set1_ps(wRatio);
        __m256 pWOffset = _mm256_set1_ps(wOffset);
        __m256 pWidthLimit = _mm256_set1_ps((Rpp32f)widthLimit);
        __m256 pWeightParams[noOfCoeffs], pBilinearCoeffs[noOfCoeffs], pDstLoc;
        Rpp32f weightParams[noOfCoeffs], bilinearCoeffs[noOfCoeffs];
        Rpp32s srcLocationColumnArray[8] = {0};     // Since 8 dst pixels are processed per iteration
        Rpp32s srcLocationRow, srcLocationColumn;

        // Resize with fused output-layout toggle (NHWC -> NCHW)
        if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp16f *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
            dstPtrRowR = dstPtrChannel;
            dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
            dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;

            for(int dstLocRow = 0; dstLocRow < dstImgSize[batchCount].height; dstLocRow++)
            {
                Rpp16f *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
                dstPtrTempR = dstPtrRowR;
                dstPtrTempG = dstPtrRowG;
                dstPtrTempB = dstPtrRowB;

                Rpp16f *srcRowPtrsForInterp[2];     // kernelSize(2)
                compute_resize_bilinear_src_loc_and_weights(dstLocRow, hRatio, srcLocationRow, &weightParams[0], hOffset);  // Compute the src row location correspoding to the dst row location
                compute_src_row_ptrs_for_bilinear_interpolation(srcRowPtrsForInterp, srcPtrChannel, srcLocationRow, heightLimit, srcDescPtr); // Compute the src row pointers for interpolation
                pWeightParams[0] = _mm256_set1_ps(weightParams[0]);
                pWeightParams[1]  = _mm256_set1_ps(weightParams[1]);
                pDstLoc = avx_pDstLocInit;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m256 pSrc[12], pDst[3];
                    __m256 negativeLocMask;
                    compute_resize_bilinear_src_loc_and_weights_avx(pDstLoc, pWRatio, srcLocationColumnArray, &pWeightParams[2], negativeLocMask, pWOffset, true);   // Compute the src col location correspoding to the dst col location
                    if(srcLocationColumnArray[7] >= widthLimit) break;
                    compute_bilinear_coefficients_avx(pWeightParams, pBilinearCoeffs);  // Compute Bilinear coefficients
                    rpp_simd_load(rpp_bilinear_load_f16pkd3_to_f32pln3_avx, srcRowPtrsForInterp, srcLocationColumnArray, pSrc, negativeLocMask); // Load input pixels required for bilinear interpolation
                    compute_bilinear_interpolation_3c_avx(pSrc, pBilinearCoeffs, pDst); // Compute Bilinear interpolation
                    rpp_simd_store(rpp_store24_f32pln3_to_f16pln3_avx, dstPtrTempR, dstPtrTempG, dstPtrTempB, pDst);    // Store dst pixels
                    dstPtrTempR += vectorIncrementPerChannel;
                    dstPtrTempG += vectorIncrementPerChannel;
                    dstPtrTempB += vectorIncrementPerChannel;
                }
                for (; vectorLoopCount < dstImgSize[batchCount].width; vectorLoopCount++)
                {
                    compute_resize_bilinear_src_loc_and_weights(vectorLoopCount, wRatio, srcLocationColumn, &weightParams[2], wOffset, srcDescPtr->strides.wStride); // Compute the src col location correspoding to the dst col location
                    compute_bilinear_coefficients(weightParams, bilinearCoeffs);    // Compute Bilinear coefficients
                    compute_bilinear_interpolation_3c_pkd(srcRowPtrsForInterp, srcLocationColumn, widthLimit, bilinearCoeffs, dstPtrTempR, dstPtrTempG, dstPtrTempB);   // Compute Bilinear interpolation
                    dstPtrTempR++;
                    dstPtrTempG++;
                    dstPtrTempB++;
                }
                dstPtrRowR += dstDescPtr->strides.hStride;
                dstPtrRowG += dstDescPtr->strides.hStride;
                dstPtrRowB += dstDescPtr->strides.hStride;
            }
        }

        // Resize with fused output-layout toggle (NCHW -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp16f *dstPtrRow;
            dstPtrRow = dstPtrChannel;

            for(int dstLocRow = 0; dstLocRow < dstImgSize[batchCount].height; dstLocRow++)
            {
                Rpp16f *dstPtrTemp;
                dstPtrTemp = dstPtrRow;

                Rpp16f *srcRowPtrsForInterp[6];     // kernelSize(2) * numOfPlanes(3)
                compute_resize_bilinear_src_loc_and_weights(dstLocRow, hRatio, srcLocationRow, &weightParams[0], hOffset);  // Compute the src row location correspoding to the dst row location
                compute_src_row_ptrs_for_bilinear_interpolation_pln(srcRowPtrsForInterp, srcPtrChannel, srcLocationRow, heightLimit, srcDescPtr); // Compute the src row pointers for interpolation
                pWeightParams[0] = _mm256_set1_ps(weightParams[0]);
                pWeightParams[1]  = _mm256_set1_ps(weightParams[1]);
                pDstLoc = avx_pDstLocInit;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m256 pSrc[12], pDst[3];
                    __m256 negativeLocMask;
                    compute_resize_bilinear_src_loc_and_weights_avx(pDstLoc, pWRatio, srcLocationColumnArray, &pWeightParams[2], negativeLocMask, pWOffset); // Compute the src col location correspoding to the dst col location
                    if(srcLocationColumnArray[7] >= widthLimit) break;
                    compute_bilinear_coefficients_avx(pWeightParams, pBilinearCoeffs);      // Compute Bilinear coefficients
                    rpp_simd_load(rpp_bilinear_load_f16pln1_to_f32pln1_avx, &srcRowPtrsForInterp[0], srcLocationColumnArray, pSrc, negativeLocMask); // Load input pixels required for bilinear interpolation
                    rpp_simd_load(rpp_bilinear_load_f16pln1_to_f32pln1_avx, &srcRowPtrsForInterp[2], srcLocationColumnArray, &pSrc[4], negativeLocMask);
                    rpp_simd_load(rpp_bilinear_load_f16pln1_to_f32pln1_avx, &srcRowPtrsForInterp[4], srcLocationColumnArray, &pSrc[8], negativeLocMask);
                    compute_bilinear_interpolation_3c_avx(pSrc, pBilinearCoeffs, pDst);     // Compute Bilinear interpolation
                    rpp_simd_store(rpp_store24_f32pln3_to_f16pkd3_avx, dstPtrTemp, pDst);   // Store dst pixels
                    dstPtrTemp += vectorIncrementPkd;
                }
                for (; vectorLoopCount < dstImgSize[batchCount].width; vectorLoopCount++)
                {
                    compute_resize_bilinear_src_loc_and_weights(vectorLoopCount, wRatio, srcLocationColumn, &weightParams[2], wOffset);  // Compute the src col location correspoding to the dst col location
                    compute_bilinear_coefficients(weightParams, bilinearCoeffs);    // Compute Bilinear coefficients
                    compute_bilinear_interpolation_3c_pkd(srcRowPtrsForInterp, srcLocationColumn, widthLimit, bilinearCoeffs, &dstPtrTemp[0], &dstPtrTemp[1], &dstPtrTemp[2]);  // Compute Bilinear interpolation
                    dstPtrTemp += dstDescPtr->c;
                }
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }

        // Resize without fused output-layout toggle (NHWC -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp16f *dstPtrRow;
            dstPtrRow = dstPtrChannel;

            for(int dstLocRow = 0; dstLocRow < dstImgSize[batchCount].height; dstLocRow++)
            {
                Rpp16f *dstPtrTemp;
                dstPtrTemp = dstPtrRow;

                Rpp16f *srcRowPtrsForInterp[2];     // kernelSize(2)
                compute_resize_bilinear_src_loc_and_weights(dstLocRow, hRatio, srcLocationRow, &weightParams[0], hOffset);  // Compute the src row location correspoding to the dst row location
                compute_src_row_ptrs_for_bilinear_interpolation(srcRowPtrsForInterp, srcPtrChannel, srcLocationRow, heightLimit, srcDescPtr); // Compute the src row pointers for interpolation
                pWeightParams[0] = _mm256_set1_ps(weightParams[0]);
                pWeightParams[1]  = _mm256_set1_ps(weightParams[1]);
                pDstLoc = avx_pDstLocInit;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m256 pSrc[12], pDst[3];
                    __m256 negativeLocMask;
                    compute_resize_bilinear_src_loc_and_weights_avx(pDstLoc, pWRatio, srcLocationColumnArray, &pWeightParams[2], negativeLocMask, pWOffset, true);   // Compute the src col location correspoding to the dst col location
                    if(srcLocationColumnArray[7] >= widthLimit) break;
                    compute_bilinear_coefficients_avx(pWeightParams, pBilinearCoeffs);      // Compute Bilinear coefficients
                    rpp_simd_load(rpp_bilinear_load_f16pkd3_to_f32pln3_avx, srcRowPtrsForInterp, srcLocationColumnArray, pSrc, negativeLocMask); // Load input pixels required for bilinear interpolation
                    compute_bilinear_interpolation_3c_avx(pSrc, pBilinearCoeffs, pDst);     // Compute Bilinear interpolation
                    rpp_simd_store(rpp_store24_f32pln3_to_f16pkd3_avx, dstPtrTemp, pDst);   // Store dst pixels
                    dstPtrTemp += vectorIncrementPkd;
                }
                for (; vectorLoopCount < dstImgSize[batchCount].width; vectorLoopCount++)
                {
                    compute_resize_bilinear_src_loc_and_weights(vectorLoopCount, wRatio, srcLocationColumn, &weightParams[2], wOffset, srcDescPtr->strides.wStride); // Compute the src col location correspoding to the dst col location
                    compute_bilinear_coefficients(weightParams, bilinearCoeffs);    // Compute Bilinear coefficients
                    compute_bilinear_interpolation_3c_pkd(srcRowPtrsForInterp, srcLocationColumn, widthLimit, bilinearCoeffs, &dstPtrTemp[0], &dstPtrTemp[1], &dstPtrTemp[2]);  // Compute Bilinear interpolation
                    dstPtrTemp += dstDescPtr->c;
                }
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }

        // Resize without fused output-layout toggle (NCHW -> NCHW for 1 channel and 3 channel)
        else if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp16f *dstPtrRow;
            dstPtrRow = dstPtrChannel;

            for(int dstLocRow = 0; dstLocRow < dstImgSize[batchCount].height; dstLocRow++)
            {
                Rpp16f *dstPtrTemp;
                dstPtrTemp = dstPtrRow;

                Rpp16f *srcRowPtrsForInterp[6];     // kernelSize(2) * numOfPlanes(3)
                compute_resize_bilinear_src_loc_and_weights(dstLocRow, hRatio, srcLocationRow, &weightParams[0], hOffset);  // Compute the src row location correspoding to the dst row location
                compute_src_row_ptrs_for_bilinear_interpolation_pln(srcRowPtrsForInterp, srcPtrChannel, srcLocationRow, heightLimit, srcDescPtr); // Compute the src row pointers for interpolation
                pWeightParams[0] = _mm256_set1_ps(weightParams[0]);
                pWeightParams[1]  = _mm256_set1_ps(weightParams[1]);
                pDstLoc = avx_pDstLocInit;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    Rpp16f *dstPtrTempChn;
                    dstPtrTempChn = dstPtrTemp;

                    __m256 pSrc[4], pDst;
                    __m256 negativeLocMask;
                    compute_resize_bilinear_src_loc_and_weights_avx(pDstLoc, pWRatio, srcLocationColumnArray, &pWeightParams[2], negativeLocMask, pWOffset); // Compute the src col location correspoding to the dst col location
                    if(srcLocationColumnArray[7] >= widthLimit) break;
                    compute_bilinear_coefficients_avx(pWeightParams, pBilinearCoeffs);          // Compute Bilinear coefficients

                    for (int c = 0; c < dstDescPtr->c; c++)
                    {
                        rpp_simd_load(rpp_bilinear_load_f16pln1_to_f32pln1_avx, &srcRowPtrsForInterp[c * kernelSize], srcLocationColumnArray, pSrc, negativeLocMask);    // Load input pixels required for bilinear interpolation
                        compute_bilinear_interpolation_1c_avx(pSrc, pBilinearCoeffs, pDst);     // Compute Bilinear interpolation
                        rpp_simd_store(rpp_store8_f32pln1_to_f16pln1_avx, dstPtrTempChn, pDst); // Store dst pixels
                        dstPtrTempChn += dstDescPtr->strides.cStride;
                    }
                    dstPtrTemp += vectorIncrementPerChannel;
                }
                for (; vectorLoopCount < dstImgSize[batchCount].width; vectorLoopCount++)
                {
                    Rpp16f *dstPtrTempChn;
                    dstPtrTempChn = dstPtrTemp;
                    compute_resize_bilinear_src_loc_and_weights(vectorLoopCount, wRatio, srcLocationColumn, &weightParams[2], wOffset);  // Compute the src col location correspoding to the dst col location
                    compute_bilinear_coefficients(weightParams, bilinearCoeffs);    // Compute Bilinear coefficients
                    for (int c = 0; c < dstDescPtr->c; c++)
                    {
                        compute_bilinear_interpolation_1c(&srcRowPtrsForInterp[c * kernelSize], widthLimit, srcLocationColumn, bilinearCoeffs, dstPtrTempChn);  // Compute Bilinear interpolation
                        dstPtrTempChn += dstDescPtr->strides.cStride;
                    }
                    dstPtrTemp++;
                }
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }
    }

    return RPP_SUCCESS;
}

RppStatus resize_bilinear_i8_i8_host_tensor(Rpp8s *srcPtr,
                                            RpptDescPtr srcDescPtr,
                                            Rpp8s *dstPtr,
                                            RpptDescPtr dstDescPtr,
                                            RpptImagePatchPtr dstImgSize,
                                            RpptROIPtr roiTensorPtrSrc,
                                            RpptRoiType roiType,
                                            RppLayoutParams srcLayoutParams)
{
    RpptROI roiDefault = {0, 0, (Rpp32s)srcDescPtr->w, (Rpp32s)srcDescPtr->h};

omp_set_dynamic(0);
#pragma omp parallel for num_threads(dstDescPtr->n)
    for(int batchCount = 0; batchCount < dstDescPtr->n; batchCount++)
    {
        RpptROI roi;
        RpptROIPtr roiPtrInput = &roiTensorPtrSrc[batchCount];
        compute_roi_validation_host(roiPtrInput, &roi, &roiDefault, roiType);

        compute_dst_size_cap_host(&dstImgSize[batchCount], dstDescPtr);     // Check if the dstImgSize exceeds dst buffer size
        Rpp32f wRatio = ((Rpp32f)(roi.xywhROI.roiWidth)) / ((Rpp32f)(dstImgSize[batchCount].width));
        Rpp32f hRatio = ((Rpp32f)(roi.xywhROI.roiHeight)) / ((Rpp32f)(dstImgSize[batchCount].height));
        Rpp32u heightLimit = roi.xywhROI.roiHeight - 1;
        Rpp32u widthLimit = (roi.xywhROI.roiWidth - 1) * srcDescPtr->strides.wStride;
        Rpp32s kernelSize = 2;
        Rpp32f kernelRadius = 1.0f; // kernelSize / 2
        Rpp32s noOfCoeffs = 4; // kernelSize * kernelSize
        Rpp32f hOffset = (hRatio - 1) * 0.5f - kernelRadius;
        Rpp32f wOffset = (wRatio - 1) * 0.5f - kernelRadius;
        Rpp32s vectorIncrementPerChannel = 8;
        Rpp32s vectorIncrementPkd = 24;

        Rpp8s *srcPtrChannel, *dstPtrChannel, *srcPtrImage, *dstPtrImage;
        srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;
        srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi.xywhROI.xy.x * srcLayoutParams.bufferMultiplier);
        dstPtrChannel = dstPtrImage;

        Rpp32u alignedLength = dstImgSize[batchCount].width & ~7;   // Align dst width to process 8 dst pixels per iteration
        __m256 pWRatio = _mm256_set1_ps(wRatio);
        __m256 pWOffset = _mm256_set1_ps(wOffset);
        __m256 pWidthLimit = _mm256_set1_ps((Rpp32f)widthLimit);

        __m256 pWeightParams[noOfCoeffs], pBilinearCoeffs[noOfCoeffs], pDstLoc;
        Rpp32f weightParams[noOfCoeffs], bilinearCoeffs[noOfCoeffs];
        Rpp32s srcLocationColumnArray[8] = {0};     // Since 8 dst pixels are processed per iteration
        Rpp32s srcLocationRow, srcLocationColumn;

        // Resize with fused output-layout toggle (NHWC -> NCHW)
        if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp8s *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
            dstPtrRowR = dstPtrChannel;
            dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
            dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;

            for(int dstLocRow = 0; dstLocRow < dstImgSize[batchCount].height; dstLocRow++)
            {
                Rpp8s *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
                dstPtrTempR = dstPtrRowR;
                dstPtrTempG = dstPtrRowG;
                dstPtrTempB = dstPtrRowB;

                Rpp8s *srcRowPtrsForInterp[2];     // kernelSize(2)
                compute_resize_bilinear_src_loc_and_weights(dstLocRow, hRatio, srcLocationRow, &weightParams[0], hOffset);  // Compute the src row location correspoding to the dst row location
                compute_src_row_ptrs_for_bilinear_interpolation(srcRowPtrsForInterp, srcPtrChannel, srcLocationRow, heightLimit, srcDescPtr); // Compute the src row pointers for interpolation
                pWeightParams[0] = _mm256_set1_ps(weightParams[0]);
                pWeightParams[1]  = _mm256_set1_ps(weightParams[1]);
                pDstLoc = avx_pDstLocInit;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m256 pSrc[12], pDst[3];
                    __m256 negativeLocMask;
                    compute_resize_bilinear_src_loc_and_weights_avx(pDstLoc, pWRatio, srcLocationColumnArray, &pWeightParams[2], negativeLocMask, pWOffset, true);   // Compute the src col location correspoding to the dst col location
                    if(srcLocationColumnArray[7] >= widthLimit) break;
                    compute_bilinear_coefficients_avx(pWeightParams, pBilinearCoeffs);                              // Compute Bilinear coefficients
                    rpp_simd_load(rpp_bilinear_load_i8pkd3_to_f32pln3_avx, srcRowPtrsForInterp, srcLocationColumnArray, pSrc, negativeLocMask);  // Load input pixels required for bilinear interpolation
                    compute_bilinear_interpolation_3c_avx(pSrc, pBilinearCoeffs, pDst);                             // Compute Bilinear interpolation
                    rpp_simd_store(rpp_store12_f32pln3_to_i8pln3_avx, dstPtrTempR, dstPtrTempG, dstPtrTempB, pDst); // Store dst pixels
                    dstPtrTempR += vectorIncrementPerChannel;
                    dstPtrTempG += vectorIncrementPerChannel;
                    dstPtrTempB += vectorIncrementPerChannel;
                }
                for (; vectorLoopCount < dstImgSize[batchCount].width; vectorLoopCount++)
                {
                    compute_resize_bilinear_src_loc_and_weights(vectorLoopCount, wRatio, srcLocationColumn, &weightParams[2], wOffset, srcDescPtr->strides.wStride); // Compute the src col location correspoding to the dst col location
                    compute_bilinear_coefficients(weightParams, bilinearCoeffs);    // Compute Bilinear coefficients
                    compute_bilinear_interpolation_3c_pkd(srcRowPtrsForInterp, srcLocationColumn, widthLimit, bilinearCoeffs, dstPtrTempR, dstPtrTempG, dstPtrTempB);   // Compute Bilinear interpolation
                    dstPtrTempR++;
                    dstPtrTempG++;
                    dstPtrTempB++;
                }
                dstPtrRowR += dstDescPtr->strides.hStride;
                dstPtrRowG += dstDescPtr->strides.hStride;
                dstPtrRowB += dstDescPtr->strides.hStride;
            }
        }

        // Resize with fused output-layout toggle (NCHW -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp8s *dstPtrRow;
            dstPtrRow = dstPtrChannel;

            for(int dstLocRow = 0; dstLocRow < dstImgSize[batchCount].height; dstLocRow++)
            {
                Rpp8s *dstPtrTemp;
                dstPtrTemp = dstPtrRow;

                Rpp8s *srcRowPtrsForInterp[6];     // kernelSize(2) * numOfPlanes(3)
                compute_resize_bilinear_src_loc_and_weights(dstLocRow, hRatio, srcLocationRow, &weightParams[0], hOffset);  // Compute the src row location correspoding to the dst row location
                compute_src_row_ptrs_for_bilinear_interpolation_pln(srcRowPtrsForInterp, srcPtrChannel, srcLocationRow, heightLimit, srcDescPtr); // Compute the src row pointers for interpolation
                pWeightParams[0] = _mm256_set1_ps(weightParams[0]);
                pWeightParams[1]  = _mm256_set1_ps(weightParams[1]);
                pDstLoc = avx_pDstLocInit;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m256 pSrc[12], pDst[3];
                    __m256 negativeLocMask;
                    compute_resize_bilinear_src_loc_and_weights_avx(pDstLoc, pWRatio, srcLocationColumnArray, &pWeightParams[2], negativeLocMask, pWOffset); // Compute the src col location correspoding to the dst col location
                    if(srcLocationColumnArray[7] >= widthLimit) break;
                    compute_bilinear_coefficients_avx(pWeightParams, pBilinearCoeffs);      // Compute Bilinear coefficients
                    rpp_simd_load(rpp_bilinear_load_i8pln1_to_f32pln1_avx, &srcRowPtrsForInterp[0], srcLocationColumnArray, pSrc, negativeLocMask);  // Load input pixels required for bilinear interpolation
                    rpp_simd_load(rpp_bilinear_load_i8pln1_to_f32pln1_avx, &srcRowPtrsForInterp[2], srcLocationColumnArray, &pSrc[4], negativeLocMask);
                    rpp_simd_load(rpp_bilinear_load_i8pln1_to_f32pln1_avx, &srcRowPtrsForInterp[4], srcLocationColumnArray, &pSrc[8], negativeLocMask);
                    compute_bilinear_interpolation_3c_avx(pSrc, pBilinearCoeffs, pDst);     // Compute Bilinear interpolation
                    rpp_simd_store(rpp_store12_f32pln3_to_i8pkd3_avx, dstPtrTemp, pDst);    // Store dst pixels
                    dstPtrTemp += vectorIncrementPkd;
                }
                for (; vectorLoopCount < dstImgSize[batchCount].width; vectorLoopCount++)
                {
                    compute_resize_bilinear_src_loc_and_weights(vectorLoopCount, wRatio, srcLocationColumn, &weightParams[2], wOffset);  // Compute the src col location correspoding to the dst col location
                    compute_bilinear_coefficients(weightParams, bilinearCoeffs);    // Compute Bilinear coefficients
                    compute_bilinear_interpolation_3c_pkd(srcRowPtrsForInterp, srcLocationColumn, widthLimit, bilinearCoeffs, &dstPtrTemp[0], &dstPtrTemp[1], &dstPtrTemp[2]);  // Compute Bilinear interpolation
                    dstPtrTemp += dstDescPtr->c;
                }
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }

        // Resize without fused output-layout toggle (NHWC -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp8s *dstPtrRow;
            dstPtrRow = dstPtrChannel;
            for(int dstLocRow = 0; dstLocRow < dstImgSize[batchCount].height; dstLocRow++)
            {
                Rpp8s *dstPtrTemp;
                dstPtrTemp = dstPtrRow;

                Rpp8s *srcRowPtrsForInterp[2];     // kernelSize(2)
                compute_resize_bilinear_src_loc_and_weights(dstLocRow, hRatio, srcLocationRow, &weightParams[0], hOffset);  // Compute the src row location correspoding to the dst row location
                compute_src_row_ptrs_for_bilinear_interpolation(srcRowPtrsForInterp, srcPtrChannel, srcLocationRow, heightLimit, srcDescPtr); // Compute the src row pointers for interpolation
                pWeightParams[0] = _mm256_set1_ps(weightParams[0]);
                pWeightParams[1]  = _mm256_set1_ps(weightParams[1]);
                pDstLoc = avx_pDstLocInit;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m256 pSrc[12], pDst[3];
                    __m256 negativeLocMask;
                    compute_resize_bilinear_src_loc_and_weights_avx(pDstLoc, pWRatio, srcLocationColumnArray, &pWeightParams[2], negativeLocMask, pWOffset, true);   // Compute the src col location correspoding to the dst col location
                    if(srcLocationColumnArray[7] >= widthLimit) break;
                    compute_bilinear_coefficients_avx(pWeightParams, pBilinearCoeffs);      // Compute Bilinear coefficients
                    rpp_simd_load(rpp_bilinear_load_i8pkd3_to_f32pln3_avx, srcRowPtrsForInterp, srcLocationColumnArray, pSrc, negativeLocMask);  // Load input pixels required for bilinear interpolation
                    compute_bilinear_interpolation_3c_avx(pSrc, pBilinearCoeffs, pDst);     // Compute Bilinear interpolation
                    rpp_simd_store(rpp_store12_f32pln3_to_i8pkd3_avx, dstPtrTemp, pDst);    // Store dst pixels
                    dstPtrTemp += vectorIncrementPkd;
                }
                for (; vectorLoopCount < dstImgSize[batchCount].width; vectorLoopCount++)
                {
                    compute_resize_bilinear_src_loc_and_weights(vectorLoopCount, wRatio, srcLocationColumn, &weightParams[2], wOffset, srcDescPtr->strides.wStride); // Compute the src col location correspoding to the dst col location
                    compute_bilinear_coefficients(weightParams, bilinearCoeffs);    // Compute Bilinear coefficients
                    compute_bilinear_interpolation_3c_pkd(srcRowPtrsForInterp, srcLocationColumn, widthLimit, bilinearCoeffs, &dstPtrTemp[0], &dstPtrTemp[1], &dstPtrTemp[2]);  // Compute Bilinear interpolation
                    dstPtrTemp += dstDescPtr->c;
                }

                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }

        // Resize without fused output-layout toggle (NCHW -> NCHW for 1 channel and 3 channel)
        else if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp8s *dstPtrRow;
            dstPtrRow = dstPtrChannel;
            for(int dstLocRow = 0; dstLocRow < dstImgSize[batchCount].height; dstLocRow++)
            {
                Rpp8s *dstPtrTemp;
                dstPtrTemp = dstPtrRow;
                Rpp8s *srcRowPtrsForInterp[6];     // kernelSize(2) * numOfPlanes(3)
                compute_resize_bilinear_src_loc_and_weights(dstLocRow, hRatio, srcLocationRow, &weightParams[0], hOffset);  // Compute the src row location correspoding to the dst row location
                compute_src_row_ptrs_for_bilinear_interpolation_pln(srcRowPtrsForInterp, srcPtrChannel, srcLocationRow, heightLimit, srcDescPtr); // Compute the src row pointers for interpolation
                pWeightParams[0] = _mm256_set1_ps(weightParams[0]);
                pWeightParams[1]  = _mm256_set1_ps(weightParams[1]);
                pDstLoc = avx_pDstLocInit;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    Rpp8s *dstPtrTempChn;
                    dstPtrTempChn = dstPtrTemp;

                    __m256 pSrc[4], pDst;
                    __m256 negativeLocMask;
                    compute_resize_bilinear_src_loc_and_weights_avx(pDstLoc, pWRatio, srcLocationColumnArray, &pWeightParams[2], negativeLocMask, pWOffset); // Compute the src col location correspoding to the dst col location
                    if(srcLocationColumnArray[7] >= widthLimit) break;
                    compute_bilinear_coefficients_avx(pWeightParams, pBilinearCoeffs);          // Compute Bilinear coefficients

                    for (int c = 0; c < dstDescPtr->c; c++)
                    {
                        rpp_simd_load(rpp_bilinear_load_i8pln1_to_f32pln1_avx, &srcRowPtrsForInterp[c * kernelSize], srcLocationColumnArray, pSrc, negativeLocMask); // Load input pixels required for bilinear interpolation
                        compute_bilinear_interpolation_1c_avx(pSrc, pBilinearCoeffs, pDst);     // Compute Bilinear interpolation
                        rpp_simd_store(rpp_store4_f32pln1_to_i8pln1_avx, dstPtrTempChn, pDst);  // Store dst pixels
                        dstPtrTempChn += dstDescPtr->strides.cStride;
                    }
                    dstPtrTemp += vectorIncrementPerChannel;
                }
                for (; vectorLoopCount < dstImgSize[batchCount].width; vectorLoopCount++)
                {
                    Rpp8s *dstPtrTempChn;
                    dstPtrTempChn = dstPtrTemp;
                    compute_resize_bilinear_src_loc_and_weights(vectorLoopCount, wRatio, srcLocationColumn, &weightParams[2], wOffset);  // Compute the src col location correspoding to the dst col location
                    compute_bilinear_coefficients(weightParams, bilinearCoeffs);    // Compute Bilinear coefficients
                    for (int c = 0; c < dstDescPtr->c; c++)
                    {
                        compute_bilinear_interpolation_1c(&srcRowPtrsForInterp[c * kernelSize], widthLimit, srcLocationColumn, bilinearCoeffs, dstPtrTempChn);  // Compute Bilinear interpolation
                        dstPtrTempChn += dstDescPtr->strides.cStride;
                    }
                    dstPtrTemp++;
                }
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }
    }

    return RPP_SUCCESS;
}

template <typename T>
RppStatus resize_separable_host_tensor(T *srcPtr,
                                       RpptDescPtr srcDescPtr,
                                       T *dstPtr,
                                       RpptDescPtr dstDescPtr,
                                       RpptImagePatchPtr dstImgSize,
                                       RpptROIPtr roiTensorPtrSrc,
                                       RpptRoiType roiType,
                                       RppLayoutParams srcLayoutParams,
                                       RpptInterpolationType interpolationType)
{
    RpptROI roiDefault = {0, 0, (Rpp32s)srcDescPtr->w, (Rpp32s)srcDescPtr->h};

omp_set_dynamic(0);
#pragma omp parallel for num_threads(dstDescPtr->n)
    for(int batchCount = 0; batchCount < dstDescPtr->n; batchCount++)
    {
        RpptROI roi;
        RpptROIPtr roiPtrInput = &roiTensorPtrSrc[batchCount];
        compute_roi_validation_host(roiPtrInput, &roi, &roiDefault, roiType);

        compute_dst_size_cap_host(&dstImgSize[batchCount], dstDescPtr);
        Rpp32f wRatio = ((Rpp32f)(roi.xywhROI.roiWidth)) / ((Rpp32f)(dstImgSize[batchCount].width));
        Rpp32f hRatio = ((Rpp32f)(roi.xywhROI.roiHeight)) / ((Rpp32f)(dstImgSize[batchCount].height));
        Rpp32u heightLimit = roi.xywhROI.roiHeight - 1;
        Rpp32u widthLimit = roi.xywhROI.roiWidth - 1;
        Rpp32s hKernelSize, wKernelSize;
        hKernelSize = compute_kernel_size(interpolationType, roi.xywhROI.roiHeight, dstImgSize[batchCount].height, hRatio);
        wKernelSize = compute_kernel_size(interpolationType, roi.xywhROI.roiWidth, dstImgSize[batchCount].width, wRatio);
        Rpp32f hKernelRadius = hKernelSize * 0.5f;
        Rpp32f wKernelRadius = wKernelSize * 0.5f;
        Rpp32f hOffset = (hRatio - 1) * 0.5f - hKernelRadius;
        Rpp32f wOffset = (wRatio - 1) * 0.5f - wKernelRadius;

        Rpp32s rowIndex[dstImgSize[batchCount].height], colIndex[dstImgSize[batchCount].width];
        Rpp32f rowCoeffs[dstImgSize[batchCount].height * hKernelSize];
        Rpp32f colCoeffs[((dstImgSize[batchCount].width + 3) & ~3) * wKernelSize]; // Buffer size is made a multiple of 4 inorder to allocate sufficient memory for Horizontal coefficients

        // Pre-compute row index and coefficients
        for(int indexCount = 0, coeffCount = 0; indexCount < dstImgSize[batchCount].height; indexCount++, coeffCount += hKernelSize)
        {
            Rpp32f weightParam;
            compute_resize_src_loc(indexCount, hRatio, rowIndex[indexCount], weightParam, hOffset);
            compute_row_coefficients(interpolationType, hKernelSize, hKernelRadius, weightParam, &rowCoeffs[coeffCount]);
        }
        // Pre-compute col index and coefficients
        for(int indexCount = 0, coeffCount = 0; indexCount < dstImgSize[batchCount].width; indexCount++)
        {
            Rpp32f weightParam;
            compute_resize_src_loc(indexCount, wRatio, colIndex[indexCount], weightParam, wOffset, srcDescPtr->strides.wStride);
            coeffCount = (indexCount % 4 == 0) ? (indexCount * wKernelSize) : coeffCount + 1; // TODO - Add comment
            compute_col_coefficients(interpolationType, wKernelSize, wKernelRadius, weightParam, &colCoeffs[coeffCount], srcDescPtr->strides.wStride);
        }

        T *srcPtrImage, *dstPtrImage;
        srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;
        srcPtrImage = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi.xywhROI.xy.x * srcLayoutParams.bufferMultiplier);

        RpptImagePatch srcImgSize;
        srcImgSize.width = roi.xywhROI.roiWidth;
        srcImgSize.height = roi.xywhROI.roiHeight;

        // The intermediate result from Vertical Resampling will have the src width and dst height
        RpptImagePatch tempImgSize;
        tempImgSize.width = roi.xywhROI.roiWidth;
        tempImgSize.height = dstImgSize[batchCount].height;

        // Allocate temproary buffer to store intermediate result of separable resampling
        Rpp32f tempPtrImage[srcDescPtr->w * dstDescPtr->h * srcDescPtr->c];

        // Create description pointer for the temporary buffer
        RpptDesc tempDesc;
        tempDesc = *srcDescPtr;
        RpptDescPtr tempDescPtr = &tempDesc;
        tempDescPtr->h = dstDescPtr->h;

        // The channel stride changes with the change in the height for PLN images
        if(srcDescPtr->layout == RpptLayout::NCHW)
            tempDescPtr->strides.cStride = srcDescPtr->w * dstDescPtr->h;

        compute_separable_vertical_resample(srcPtrImage, tempPtrImage, srcDescPtr, tempDescPtr, srcImgSize, tempImgSize, rowIndex, rowCoeffs, hKernelSize);
        compute_separable_horizontal_resample(tempPtrImage, dstPtrImage, tempDescPtr, dstDescPtr, tempImgSize, dstImgSize[batchCount], colIndex, colCoeffs, wKernelSize);
    }

    return RPP_SUCCESS;
}
