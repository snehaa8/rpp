#include <hip/hip_runtime.h>
#include "rpp_hip_common.hpp"

__global__ void image_sum_grid_result_tensor(float *srcPtr,
                                             uint xBufferLength,
                                             float *dstPtr)
{
    int id_x = hipThreadIdx_x * 8;
    int id_z = hipBlockIdx_z;

    __shared__ float partialSumLDS[1024];                               // 256 * 4 floats of src in a 256 x 1 thread block
    float4 *partialSumLDS_f4 = (float4 *)&partialSumLDS[0];             // float4 pointer to beginning of buffer in LDS
    partialSumLDS_f4[hipThreadIdx_x] = (float4) 0.0f;                   // vectorized float4 initialization of LDS to 0 using all 256 x 1 threads

    if (id_x >= xBufferLength)
    {
        return;
    }

    int xAlignedLength = xBufferLength & ~7;                            // alignedLength for vectorized global loads
    int xDiff = xBufferLength - xAlignedLength;                         // difference between bufferLength and alignedLength
    uint srcIdx = (id_z * xBufferLength) + id_x;

    d_float8 src_f8;
    rpp_hip_load8_and_unpack_to_float8(srcPtr + srcIdx, &src_f8);       // load 8 pixels to local mmemory
    if (id_x + 8 > xBufferLength)
        for(int i = xDiff; i < 8; i++)
            src_f8.f1[i] = 0.0f;                                        // reset invalid loads to 0.0f
    partialSumLDS_f4[hipThreadIdx_x] += (src_f8.f4[0] + src_f8.f4[1]);  // perform small work of vectorized addition of two f4s and store in LDS
    __syncthreads();                                                    // syncthreads after LDS load

    // Vectorized float4 reduction of 1024 floats on 256 threads per block in x dimension
    for (int threadMax = 128; threadMax >= 1; threadMax /= 2)
    {
        if (hipThreadIdx_x < threadMax)
            rpp_hip_math_add4(partialSumLDS_f4[hipThreadIdx_x], partialSumLDS_f4[hipThreadIdx_x + threadMax], partialSumLDS_f4[hipThreadIdx_x]);
        __syncthreads();
    }

    // Final reduction of float4 vector to single float
    if (hipThreadIdx_x == 0)
        dstPtr[hipBlockIdx_z] = partialSumLDS_f4[0].x + partialSumLDS_f4[0].y + partialSumLDS_f4[0].z + partialSumLDS_f4[0].w;
}

template <typename T, typename U>
__global__ void image_sum_pln1_tensor(T *srcPtr,
                                      uint2 srcStridesNH,
                                      U *imageSumArr,
                                      RpptROIPtr roiTensorPtrSrc)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    __shared__ float partialSumLDS[16][64];                                     // 16 rows of src, 64 cols of src in a 16 x 16 thread block
    float *partialSumLDSRowPtr = &partialSumLDS[hipThreadIdx_y][0];             // float pointer to beginning of each row in LDS
    float4 *partialSumLDSRowPtr_f4 = (float4 *)partialSumLDSRowPtr;             // float4 pointer to beginning of each row in LDS
    partialSumLDSRowPtr_f4[hipThreadIdx_x] = (float4) 0.0f;                     // vectorized float4 initialization of LDS to 0 using all 16x16 threads

    if ((id_y >= roiTensorPtrSrc[id_z].xywhROI.roiHeight) || (id_x >= roiTensorPtrSrc[id_z].xywhROI.roiWidth))
    {
        return;
    }

    int xAlignedLength = roiTensorPtrSrc[id_z].xywhROI.roiWidth & ~7;           // alignedLength for vectorized global loads
    int xDiff = roiTensorPtrSrc[id_z].xywhROI.roiWidth - xAlignedLength;        // difference between roiWidth and alignedLength
    uint srcIdx = (id_z * srcStridesNH.x) + ((id_y + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNH.y) + (id_x + roiTensorPtrSrc[id_z].xywhROI.xy.x);

    d_float8 src_f8;
    rpp_hip_load8_and_unpack_to_float8(srcPtr + srcIdx, &src_f8);               // load 8 pixels to local mmemory
    if (id_x + 8 > roiTensorPtrSrc[id_z].xywhROI.roiWidth)
        for(int i = xDiff; i < 8; i++)
            src_f8.f1[i] = 0.0f;                                                // reset invalid loads to 0.0f
    partialSumLDSRowPtr_f4[hipThreadIdx_x] += (src_f8.f4[0] + src_f8.f4[1]);    // perform small work of vectorized addition of two f4s and store in LDS
    __syncthreads();                                                            // syncthreads after LDS load

    // Vectorized float4 reduction of 64 floats on 16 threads per block in x dimension (for every y dimension)
    for (int threadMax = 8; threadMax >= 1; threadMax /= 2)
    {
        if (hipThreadIdx_x < threadMax)
            rpp_hip_math_add4(partialSumLDSRowPtr_f4[hipThreadIdx_x], partialSumLDSRowPtr_f4[hipThreadIdx_x + threadMax], partialSumLDSRowPtr_f4[hipThreadIdx_x]);
        __syncthreads();
    }

    if (hipThreadIdx_x == 0)
    {
        // Vectorized float4 reduction of 64 floats on 16 threads per block in y dimension
        for (int threadMax = 8, increment = 128; threadMax >= 1; threadMax /= 2, increment /= 2)
        {
            if (hipThreadIdx_y < threadMax)
                rpp_hip_math_add4(partialSumLDSRowPtr_f4[hipThreadIdx_x], partialSumLDSRowPtr_f4[hipThreadIdx_x + increment], partialSumLDSRowPtr_f4[hipThreadIdx_x]);
            __syncthreads();
        }

        // Final reduction of float4 vector to single float
        if (hipThreadIdx_y == 0)
            imageSumArr[(hipBlockIdx_z * hipGridDim_y + hipBlockIdx_y) * hipGridDim_x + hipBlockIdx_x] = partialSumLDSRowPtr_f4[0].x +
                                                                                                         partialSumLDSRowPtr_f4[0].y +
                                                                                                         partialSumLDSRowPtr_f4[0].z +
                                                                                                         partialSumLDSRowPtr_f4[0].w;
    }
}

template <typename T, typename U>
__global__ void image_sum_pln1_tensor_without_ilp(T *srcPtr,
                                                  uint2 srcStridesNH,
                                                  U *imageSumArr,
                                                  RpptROIPtr roiTensorPtrSrc)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    __shared__ float partialSumLDS[16][16];                                     // 16 rows of src, 16 cols of src in a 16 x 16 thread block
    float *partialSumLDSRowPtr = &partialSumLDS[hipThreadIdx_y][0];             // float pointer to beginning of each row in LDS
    partialSumLDSRowPtr[hipThreadIdx_x] = 0.0f;                                 // initialization of LDS to 0 using all 16x16 threads

    if ((id_y >= roiTensorPtrSrc[id_z].xywhROI.roiHeight) || (id_x >= roiTensorPtrSrc[id_z].xywhROI.roiWidth))
    {
        return;
    }

    int xAlignedLength = roiTensorPtrSrc[id_z].xywhROI.roiWidth & ~7;           // alignedLength for vectorized global loads
    int xDiff = roiTensorPtrSrc[id_z].xywhROI.roiWidth - xAlignedLength;        // difference between roiWidth and alignedLength
    uint srcIdx = (id_z * srcStridesNH.x) + ((id_y + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNH.y) + (id_x + roiTensorPtrSrc[id_z].xywhROI.xy.x);

    d_float8 src_f8;
    rpp_hip_load8_and_unpack_to_float8(srcPtr + srcIdx, &src_f8);               // load 8 pixels to local mmemory
    if (id_x + 8 > roiTensorPtrSrc[id_z].xywhROI.roiWidth)
        for(int i = xDiff; i < 8; i++)
            src_f8.f1[i] = 0.0f;                                                // reset invalid loads to 0.0f
    partialSumLDSRowPtr[hipThreadIdx_x] += (src_f8.f1[0] + src_f8.f1[1] + src_f8.f1[2] + src_f8.f1[3] + src_f8.f1[4] + src_f8.f1[5] + src_f8.f1[6] + src_f8.f1[7]);    // perform small work of vectorized addition of two f4s and store in LDS
    __syncthreads();                                                            // syncthreads after LDS load

    // Reduction of 16 floats on 16 threads per block in x dimension (for every y dimension)
    for (int threadMax = 8; threadMax >= 1; threadMax /= 2)
    {
        if (hipThreadIdx_x < threadMax)
            partialSumLDSRowPtr[hipThreadIdx_x] += partialSumLDSRowPtr[hipThreadIdx_x + threadMax];
        __syncthreads();
    }

    if (hipThreadIdx_x == 0)
    {
        // Reduction of 16 floats on 16 threads per block in y dimension
        for (int threadMax = 8, increment = 128; threadMax >= 1; threadMax /= 2, increment /= 2)
        {
            if (hipThreadIdx_x < threadMax)
                partialSumLDSRowPtr[hipThreadIdx_x] += partialSumLDSRowPtr[hipThreadIdx_x + increment];
            __syncthreads();
        }

        // Final assignment of single float to dst
        if (hipThreadIdx_y == 0)
            imageSumArr[(hipBlockIdx_z * hipGridDim_y + hipBlockIdx_y) * hipGridDim_x + hipBlockIdx_x] = partialSumLDSRowPtr[0];
    }
}

template <typename T, typename U>
RppStatus hip_exec_image_sum_tensor(T *srcPtr,
                                    RpptDescPtr srcDescPtr,
                                    U *imageSumArr,
                                    RpptROIPtr roiTensorPtrSrc,
                                    RpptRoiType roiType,
                                    rpp::Handle& handle)
{
    if (roiType == RpptRoiType::LTRB)
        hip_exec_roi_converison_ltrb_to_xywh(roiTensorPtrSrc, handle);

    int localThreads_x = LOCAL_THREADS_X;
    int localThreads_y = LOCAL_THREADS_Y;
    int localThreads_z = LOCAL_THREADS_Z;
    int globalThreads_x = (srcDescPtr->strides.hStride + 7) >> 3;
    int globalThreads_y = srcDescPtr->h;
    int globalThreads_z = handle.GetBatchSize();
    int gridDim_x = (int) ceil((float)globalThreads_x/localThreads_x);
    int gridDim_y = (int) ceil((float)globalThreads_y/localThreads_y);
    int gridDim_z = (int) ceil((float)globalThreads_z/localThreads_z);

    Rpp32u imagePartialSumArrLength = gridDim_x * gridDim_y * gridDim_z;
    float *imagePartialSumArr;
    imagePartialSumArr = handle.GetInitHandle()->mem.mgpu.maskArr.floatmem;
    hipMemset(imagePartialSumArr, 0, imagePartialSumArrLength * sizeof(float));

    if ((srcDescPtr->c == 1) && (srcDescPtr->layout == RpptLayout::NCHW))
    {
        hipLaunchKernelGGL(image_sum_pln1_tensor,
                           dim3(gridDim_x, gridDim_y, gridDim_z),
                           dim3(localThreads_x, localThreads_y, localThreads_z),
                           0,
                           handle.GetStream(),
                           srcPtr,
                           make_uint2(srcDescPtr->strides.nStride, srcDescPtr->strides.hStride),
                           imagePartialSumArr,
                           roiTensorPtrSrc);
        hipLaunchKernelGGL(image_sum_grid_result_tensor,
                           dim3(1, 1, gridDim_z),
                           dim3(256, 1, 1),
                           0,
                           handle.GetStream(),
                           imagePartialSumArr,
                           gridDim_x * gridDim_y,
                           imageSumArr);
    }

    return RPP_SUCCESS;
}
