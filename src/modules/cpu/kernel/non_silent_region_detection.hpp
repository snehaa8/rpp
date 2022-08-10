#include "rppdefs.h"
#include "rpp_cpu_common.hpp"
#include "rpp_cpu_simd.hpp"

inline Rpp32f compute_square_host(Rpp32f &value)
{
    Rpp32f res = value;

    return (res * res);
}

inline Rpp32f compute_max_in_vector_host(std::vector<float> &values, int length)
{
    Rpp32f max = values[0];
    for(int i = 1; i < length; i++)
        max = std::max(max, values[i]);

    return max;
}

RppStatus non_silent_region_detection_host_tensor(Rpp32f *srcPtr,
                                                  RpptDescPtr srcDescPtr,
                                                  Rpp32s *srcSizeTensor,
                                                  Rpp32s *detectedIndexTensor,
                                                  Rpp32s *detectionLengthTensor,
                                                  Rpp32f *cutOffDBTensor,
                                                  Rpp32s *windowLengthTensor,
                                                  Rpp32f *referencePowerTensor,
                                                  Rpp32s *resetIntervalTensor,
                                                  bool *referenceMaxTensor)
{
    omp_set_dynamic(0);
#pragma omp parallel for num_threads(srcDescPtr->n)
    for(int batchCount = 0; batchCount < srcDescPtr->n; batchCount++)
    {
        Rpp32f *srcPtrTemp = srcPtr + batchCount * srcDescPtr->strides.nStride;
        Rpp32s srcSize = srcSizeTensor[batchCount];
        Rpp32s windowLength = windowLengthTensor[batchCount];
        Rpp32f referencePower = referencePowerTensor[batchCount];
        Rpp32f cutOffDB = cutOffDBTensor[batchCount];
        bool referenceMax = referenceMaxTensor[batchCount];

        // set reset interval based on the user input
        Rpp32s resetInterval = resetIntervalTensor[batchCount];
        resetInterval = (resetInterval == -1) ? srcSize : resetInterval;

        // Calculate buffer size for mms array and allocate mms buffer
        Rpp32s mmsBufferSize = srcSize - windowLength + 1;
        std::vector<float> mmsBuffer;
        mmsBuffer.reserve(mmsBufferSize);

        // Calculate moving mean square of input array and store in mms buffer
        Rpp32f sumOfSquares = 0.0f;
        Rpp32f meanFactor = 1.0f / windowLength;
        Rpp32s windowLengthMinus1 = windowLength - 1;
        __m256 pMeanFactor = _mm256_set1_ps(meanFactor);

        int windowBegin = 0;
        while(windowBegin <= srcSize - windowLength)
        {
            for(int i = windowBegin; i < windowLength; i++)
                sumOfSquares += compute_square_host(srcPtrTemp[i]);
            mmsBuffer[windowBegin] = sumOfSquares * meanFactor;

            auto intervalEndIdx = std::min(windowBegin++ + resetInterval, srcSize) - windowLengthMinus1;
            std::cerr << "\nintervalEndIdx for file " << batchCount + 1 << " = " << intervalEndIdx;

            Rpp32u vectorIncrement = 8;
            Rpp32u alignedLength = (intervalEndIdx / 8) * 8;

            __m256 pSumOfSquares = _mm256_set1_ps(sumOfSquares);
            for(; windowBegin < alignedLength; windowBegin += vectorIncrement)
            {
                __m256 p[2];
                p[0] = _mm256_loadu_ps(&srcPtrTemp[windowBegin + windowLengthMinus1]);
                p[0] = _mm256_mul_ps(p[0], p[0]);
                p[1] = _mm256_loadu_ps(&srcPtrTemp[windowBegin - 1]);
                p[1] = _mm256_mul_ps(p[1], p[1]);
                p[0] = _mm256_sub_ps(p[0], p[1]);
                commpute_scan_8_avx(p[0]);
                p[0] = _mm256_add_ps(p[0], pSumOfSquares);
                p[1] = _mm256_mul_ps(p[0], pMeanFactor);
                _mm256_storeu_ps(&mmsBuffer[windowBegin], p[1]);
                p[0] = _mm256_permute2f128_ps(p[0], p[0], 0x11);
                pSumOfSquares = _mm256_permute_ps(p[0], 0xff);
            }
            sumOfSquares = _mm256_cvtss_f32(pSumOfSquares);
            for(; windowBegin < intervalEndIdx; windowBegin++)
            {
                sumOfSquares += compute_square_host(srcPtrTemp[windowBegin + windowLengthMinus1]) - compute_square_host(srcPtrTemp[windowBegin - 1]);
                mmsBuffer[windowBegin] = sumOfSquares * meanFactor;
            }
        }

        // Convert cutOff from DB to magnitude
        Rpp32f base = (referenceMax) ? compute_max_in_vector_host(mmsBuffer, mmsBufferSize) : referencePower;
        Rpp32f cutOffMag = base * std::pow(10.0f, cutOffDB * 0.1f);

        // Calculate begining index, length of non silent region from the mms buffer
        int endIdx = mmsBufferSize;
        int beginIdx = endIdx;
        for(int i = 0; i < endIdx; i++)
        {
            if(mmsBuffer[i] >= cutOffMag)
            {
                beginIdx = i;
                break;
            }
        }

        if(beginIdx == endIdx)
        {
            detectedIndexTensor[batchCount] = 0;
            detectionLengthTensor[batchCount] = 0;
        }
        else
        {
            for(int i = endIdx - 1; i >= beginIdx; i--)
            {
                if(mmsBuffer[i] >= cutOffMag)
                {
                    endIdx = i;
                    break;
                }
            }
            detectedIndexTensor[batchCount] = beginIdx;
            detectionLengthTensor[batchCount] = endIdx - beginIdx + 1;
        }

        // Extend non silent region
        if(detectionLengthTensor[batchCount] != 0)
            detectionLengthTensor[batchCount] += windowLength - 1;
    }

    return RPP_SUCCESS;
}
