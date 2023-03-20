#include <hip/hip_runtime.h>
#include "rpp_hip_common.hpp"

__global__ void image_sum_grid_result_tensor(float *srcPtr,
                                             uint xBufferLength,
                                             float *dstPtr)
{
    int id_x = hipThreadIdx_x * 8;
    int id_z = hipBlockIdx_z;

    __shared__ float partialSumLDS[256];                            // 1024 floats of src reduced to 256 in a 256 x 1 thread block
    partialSumLDS[hipThreadIdx_x] = 0.0f;                           // initialization of LDS to 0 using all 256 x 1 threads

    if (id_x >= xBufferLength)
    {
        return;
    }

    int xAlignedLength = xBufferLength & ~7;                        // alignedLength for vectorized global loads
    int xDiff = xBufferLength - xAlignedLength;                     // difference between bufferLength and alignedLength
    uint srcIdx = (id_z * xBufferLength) + id_x;

    d_float8 src_f8;
    rpp_hip_load8_and_unpack_to_float8(srcPtr + srcIdx, &src_f8);   // load 8 pixels to local mmemory
    if (id_x + 8 > xBufferLength)
        for(int i = xDiff; i < 8; i++)
            src_f8.f1[i] = 0.0f;                                    // local memory reset of invalid values (from the vectorized global load) to 0.0f
    src_f8.f4[0] += src_f8.f4[1];                                   // perform small work of vectorized float4 addition
    partialSumLDS[hipThreadIdx_x] += (src_f8.f1[0] +
                                      src_f8.f1[1] +
                                      src_f8.f1[2] +
                                      src_f8.f1[3]);                // perform small work of reducing float4s to float using 256 x 1 threads and store in LDS
    __syncthreads();                                                // syncthreads after LDS load

    // Reduction of 256 floats on 256 threads per block in x dimension
    for (int threadMax = 128; threadMax >= 1; threadMax /= 2)
    {
        if (hipThreadIdx_x < threadMax)
            partialSumLDS[hipThreadIdx_x] += partialSumLDS[hipThreadIdx_x + threadMax];
        __syncthreads();
    }

    // Final store to dst
    if (hipThreadIdx_x == 0)
        dstPtr[hipBlockIdx_z] = partialSumLDS[0];
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

    __shared__ float partialSumLDS[16][16];                                 // 16 rows of src, 128 reduced cols of src in a 16 x 16 thread block
    float *partialSumLDSRowPtr = &partialSumLDS[hipThreadIdx_y][0];         // float pointer to beginning of each row in LDS
    partialSumLDSRowPtr[hipThreadIdx_x] = 0.0f;                             // initialization of LDS to 0 using all 16 x 16 threads

    if ((id_y >= roiTensorPtrSrc[id_z].xywhROI.roiHeight) || (id_x >= roiTensorPtrSrc[id_z].xywhROI.roiWidth))
    {
        return;
    }

    int xAlignedLength = roiTensorPtrSrc[id_z].xywhROI.roiWidth & ~7;       // alignedLength for vectorized global loads
    int xDiff = roiTensorPtrSrc[id_z].xywhROI.roiWidth - xAlignedLength;    // difference between roiWidth and alignedLength
    uint srcIdx = (id_z * srcStridesNH.x) + ((id_y + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNH.y) + (id_x + roiTensorPtrSrc[id_z].xywhROI.xy.x);

    d_float8 src_f8;
    rpp_hip_load8_and_unpack_to_float8(srcPtr + srcIdx, &src_f8);           // load 8 pixels to local mmemory
    if (id_x + 8 > roiTensorPtrSrc[id_z].xywhROI.roiWidth)
        for(int i = xDiff; i < 8; i++)
            src_f8.f1[i] = 0.0f;                                            // local memory reset of invalid values (from the vectorized global load) to 0.0f
    src_f8.f4[0] += src_f8.f4[1];                                           // perform small work of vectorized float4 addition
    partialSumLDSRowPtr[hipThreadIdx_x] = (src_f8.f1[0] +
                                           src_f8.f1[1] +
                                           src_f8.f1[2] +
                                           src_f8.f1[3]);                   // perform small work of reducing float4s to float using 16 x 16 threads and store in LDS
    __syncthreads();                                                        // syncthreads after LDS load

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
            if (hipThreadIdx_y < threadMax)
                partialSumLDSRowPtr[0] += partialSumLDSRowPtr[increment];
            __syncthreads();
        }

        // Final store to dst
        if (hipThreadIdx_y == 0)
            imageSumArr[(hipBlockIdx_z * hipGridDim_y + hipBlockIdx_y) * hipGridDim_x + hipBlockIdx_x] = partialSumLDSRowPtr[0];
    }
}

__global__ void image_sum_grid_3channel_result_tensor(float *srcPtr,
                                                      uint xBufferLength,
                                                      float *dstPtr)
{
    int id_x = hipThreadIdx_x * 8;
    int id_z = hipBlockIdx_z;

    __shared__ float partialSumRLDS[256];                            // 1024 floats of src reduced to 256 in a 256 x 1 thread block
    __shared__ float partialSumGLDS[256];
    __shared__ float partialSumBLDS[256];
    partialSumRLDS[hipThreadIdx_x] = 0.0f;                           // initialization of LDS to 0 using all 256 x 1 threads
    partialSumGLDS[hipThreadIdx_x] = 0.0f;
    partialSumBLDS[hipThreadIdx_x] = 0.0f;

    if (id_x >= xBufferLength)
    {
        return;
    }

    int xAlignedLength = xBufferLength & ~23;                        // alignedLength for vectorized global loads
    int xDiff = xBufferLength - xAlignedLength;                     // difference between bufferLength and alignedLength
    uint srcIdx = (id_z * xBufferLength) + id_x;

    d_float24 src_f24;
    rpp_hip_load24_pkd3_and_unpack_to_float24_pln3(srcPtr + srcIdx, &src_f24);           // load 8 pixels to local mmemory
    if (id_x + 24 > roiTensorPtrSrc[id_z].xywhROI.roiWidth)
        for(int i = xDiff; i < 24; i++)
            src_f24.f1[i] = 0.0f;                                            // local memory reset of invalid values (from the vectorized global load) to 0.0f
    src_f24.f8[0].f4[0] += src_f24.f8[0].f4[1];                                           // perform small work of vectorized float4 addition
    src_f24.f8[1].f4[0] += src_f24.f8[1].f4[1];
    src_f24.f8[2].f4[0] += src_f24.f8[2].f4[1];
    partialSumRLDSRowPtr[hipThreadIdx_x] = (src_f24.f8[0].f1[0] +
                                            src_f24.f8[0].f1[1] +
                                            src_f24.f8[0].f1[2] +
                                            src_f24.f8[0].f1[3]);                   // perform small work of reducing R float4s to float using 16 x 16 threads and store in LDS
    partialSumGLDSRowPtr[hipThreadIdx_x] = (src_f24.f8[1].f1[0] +
                                            src_f24.f8[1].f1[1] +
                                            src_f24.f8[1].f1[2] +
                                            src_f24.f8[1].f1[3]);                   // perform small work of reducing G float4s to float using 16 x 16 threads and store in LDS
    partialSumBLDSRowPtr[hipThreadIdx_x] = (src_f24.f8[1].f1[0] +
                                            src_f24.f8[1].f1[1] +
                                            src_f24.f8[1].f1[2] +
                                            src_f24.f8[1].f1[3]);                   // perform small work of reducing B float4s to float using 16 x 16 threads and store in LDS

    __syncthreads();                                                // syncthreads after LDS load

    // Reduction of 256 floats on 256 threads per block in x dimension
    for (int threadMax = 128; threadMax >= 1; threadMax /= 2)
    {
        if (hipThreadIdx_x < threadMax)
        {
            partialRSumLDS[hipThreadIdx_x] += partialRSumLDS[hipThreadIdx_x + threadMax];
            partialGSumLDS[hipThreadIdx_x] += partialGSumLDS[hipThreadIdx_x + threadMax];
            partialBSumLDS[hipThreadIdx_x] += partialBSumLDS[hipThreadIdx_x + threadMax];
        }
        __syncthreads();
    }

    // Final store to dst
    if (hipThreadIdx_x == 0)
    {
        float sum = partialSumRLDS[0] + partialSumGLDS[0] + partialSumBLDS[0];
        dstPtr[hipBlockIdx_z * 4] = partialSumRLDS[0];
        dstPtr[(hipBlockIdx_z * 4) + 1] = partialSumGLDS[0];
        dstPtr[(hipBlockIdx_z * 4) + 2] = partialSumBLDS[0];
        dstPtr[(hipBlockIdx_z * 4) + 3] = sum;
    }
}

template <typename T, typename U>
__global__ void image_sum_pln3_tensor(T *srcPtr,
                                      uint2 srcStridesNH,
                                      U *imageSumArr,
                                      RpptROIPtr roiTensorPtrSrc)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    __shared__ float partialSumRLDS[16][16];                                 // 16 rows of src, 128 reduced cols of src in a 16 x 16 thread block
    __shared__ float partialSumGLDS[16][16];
    __shared__ float partialSumBLDS[16][16];
    float *partialSumRLDSRowPtr = &partialSumRLDS[hipThreadIdx_y][0];         // float pointer to beginning of each row in LDS
    float *partialSumGLDSRowPtr = &partialSumGLDS[hipThreadIdx_y][0];
    float *partialSumBLDSRowPtr = &partialSumBLDS[hipThreadIdx_y][0];
    partialSumRLDSRowPtr[hipThreadIdx_x] = 0.0f;                             // initialization of LDS to 0 using all 16 x 16 threads
    partialSumGLDSRowPtr[hipThreadIdx_x] = 0.0f;
    partialSumBLDSRowPtr[hipThreadIdx_x] = 0.0f;

    if ((id_y >= roiTensorPtrSrc[id_z].xywhROI.roiHeight) || (id_x >= roiTensorPtrSrc[id_z].xywhROI.roiWidth))
    {
        return;
    }

    int xAlignedLength = roiTensorPtrSrc[id_z].xywhROI.roiWidth & ~23;       // alignedLength for vectorized global loads
    int xDiff = roiTensorPtrSrc[id_z].xywhROI.roiWidth - xAlignedLength;    // difference between roiWidth and alignedLength
    uint srcIdx = (id_z * srcStridesNH.x) + ((id_y + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNH.y) + (id_x + roiTensorPtrSrc[id_z].xywhROI.xy.x);

    d_float24 srcR_f24;
    rpp_hip_load24_pln3_and_unpack_to_float24_pln3(srcPtr + srcIdx, srcStridesNCH.y, &src_f24);           // load 8 pixels to local mmemory
    if (id_x + 24 > roiTensorPtrSrc[id_z].xywhROI.roiWidth)
        for(int i = xDiff; i < 24; i++)
            src_f24.f1[i] = 0.0f;                                            // local memory reset of invalid values (from the vectorized global load) to 0.0f
    src_f24.f8[0].f4[0] += src_f24.f8[0].f4[1];                                           // perform small work of vectorized float4 addition
    src_f24.f8[1].f4[0] += src_f24.f8[1].f4[1];
    src_f24.f8[2].f4[0] += src_f24.f8[2].f4[1];
    partialSumRLDSRowPtr[hipThreadIdx_x] = (src_f24.f8[0].f1[0] +
                                            src_f24.f8[0].f1[1] +
                                            src_f24.f8[0].f1[2] +
                                            src_f24.f8[0].f1[3]);                   // perform small work of reducing R float4s to float using 16 x 16 threads and store in LDS
    partialSumGLDSRowPtr[hipThreadIdx_x] = (src_f24.f8[1].f1[0] +
                                            src_f24.f8[1].f1[1] +
                                            src_f24.f8[1].f1[2] +
                                            src_f24.f8[1].f1[3]);                   // perform small work of reducing G float4s to float using 16 x 16 threads and store in LDS
    partialSumBLDSRowPtr[hipThreadIdx_x] = (src_f24.f8[1].f1[0] +
                                            src_f24.f8[1].f1[1] +
                                            src_f24.f8[1].f1[2] +
                                            src_f24.f8[1].f1[3]);                   // perform small work of reducing B float4s to float using 16 x 16 threads and store in LDS

    __syncthreads();                                                        // syncthreads after LDS load

    // Reduction of 16 floats on 16 threads per block in x dimension (for every y dimension)
    for (int threadMax = 8; threadMax >= 1; threadMax /= 2)
    {
        if (hipThreadIdx_x < threadMax)
        {
            partialSumRLDSRowPtr[hipThreadIdx_x] += partialSumRLDSRowPtr[hipThreadIdx_x + threadMax];
            partialSumGLDSRowPtr[hipThreadIdx_x] += partialSumGLDSRowPtr[hipThreadIdx_x + threadMax];
            partialSumBLDSRowPtr[hipThreadIdx_x] += partialSumBLDSRowPtr[hipThreadIdx_x + threadMax];
        }
        __syncthreads();
    }

    if (hipThreadIdx_x == 0)
    {
        // Reduction of 16 floats on 16 threads per block in y dimension
        for (int threadMax = 8, increment = 128; threadMax >= 1; threadMax /= 2, increment /= 2)
        {
            if (hipThreadIdx_y < threadMax)
            {
                partialSumRLDSRowPtr[0] += partialSumRLDSRowPtr[increment];
                partialSumGLDSRowPtr[0] += partialSumGLDSRowPtr[increment];
                partialSumBLDSRowPtr[0] += partialSumBLDSRowPtr[increment];
            }
            __syncthreads();
        }

        // Final store to dst
        if (hipThreadIdx_y == 0)
        {
            imageSumArr[((hipBlockIdx_z * hipGridDim_y + hipBlockIdx_y) * hipGridDim_x + hipBlockIdx_x) * 3] = partialSumRLDSRowPtr[0];
            imageSumArr[((hipBlockIdx_z * hipGridDim_y + hipBlockIdx_y) * hipGridDim_x + hipBlockIdx_x) * 3 + 1] = partialSumGLDSRowPtr[0];
            imageSumArr[((hipBlockIdx_z * hipGridDim_y + hipBlockIdx_y) * hipGridDim_x + hipBlockIdx_x) * 3 + 2] = partialSumBLDSRowPtr[0];
        }
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

    if ((srcDescPtr->c == 1) && (srcDescPtr->layout == RpptLayout::NCHW))
    {
        Rpp32u imagePartialSumArrLength = gridDim_x * gridDim_y * gridDim_z;
        float *imagePartialSumArr;
        imagePartialSumArr = handle.GetInitHandle()->mem.mgpu.maskArr.floatmem;
        hipMemset(imagePartialSumArr, 0, imagePartialSumArrLength * sizeof(float));

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
    else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW))
    {
        Rpp32u imagePartialSumArrLength = gridDim_x * gridDim_y * gridDim_z * 4;
        float *imagePartialSumArr;
        imagePartialSumArr = handle.GetInitHandle()->mem.mgpu.maskArr.floatmem;
        hipMemset(imagePartialSumArr, 0, imagePartialSumArrLength * sizeof(float));

        hipLaunchKernelGGL(image_sum_pln3_tensor,
                           dim3(gridDim_x, gridDim_y, gridDim_z),
                           dim3(localThreads_x, localThreads_y, localThreads_z),
                           0,
                           handle.GetStream(),
                           srcPtr,
                           make_uint2(srcDescPtr->strides.nStride, srcDescPtr->strides.hStride),
                           imagePartialSumArr,
                           roiTensorPtrSrc);
        hipLaunchKernelGGL(image_sum_grid_3channel_result_tensor,
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