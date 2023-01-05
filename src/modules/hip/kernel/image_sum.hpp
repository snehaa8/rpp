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

    // Unrolled loop to run a vectorized reduction of 1024 floats on 256 threads per block in x dimension
    if (hipThreadIdx_x < 128) partialSumLDS_f4[hipThreadIdx_x] += partialSumLDS_f4[hipThreadIdx_x + 128];
    __syncthreads();
    if (hipThreadIdx_x < 64) partialSumLDS_f4[hipThreadIdx_x] += partialSumLDS_f4[hipThreadIdx_x + 64];
    __syncthreads();
    if (hipThreadIdx_x < 32) partialSumLDS_f4[hipThreadIdx_x] += partialSumLDS_f4[hipThreadIdx_x + 32];
    __syncthreads();
    if (hipThreadIdx_x < 16) partialSumLDS_f4[hipThreadIdx_x] += partialSumLDS_f4[hipThreadIdx_x + 16];
    __syncthreads();
    if (hipThreadIdx_x < 8) partialSumLDS_f4[hipThreadIdx_x] += partialSumLDS_f4[hipThreadIdx_x + 8];
    __syncthreads();
    if (hipThreadIdx_x < 4) partialSumLDS_f4[hipThreadIdx_x] += partialSumLDS_f4[hipThreadIdx_x + 4];
    __syncthreads();
    if (hipThreadIdx_x < 2) partialSumLDS_f4[hipThreadIdx_x] += partialSumLDS_f4[hipThreadIdx_x + 2];
    __syncthreads();
    if (hipThreadIdx_x < 1) partialSumLDS_f4[hipThreadIdx_x] += partialSumLDS_f4[hipThreadIdx_x + 1];
    __syncthreads();

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

    // Unrolled loop to run a vectorized reduction of 64 floats on 16 threads per block in x dimension (for every y dimension)
    if (hipThreadIdx_x < 8) partialSumLDSRowPtr_f4[hipThreadIdx_x] += partialSumLDSRowPtr_f4[hipThreadIdx_x + 8];
    __syncthreads();
    if (hipThreadIdx_x < 4) partialSumLDSRowPtr_f4[hipThreadIdx_x] += partialSumLDSRowPtr_f4[hipThreadIdx_x + 4];
    __syncthreads();
    if (hipThreadIdx_x < 2) partialSumLDSRowPtr_f4[hipThreadIdx_x] += partialSumLDSRowPtr_f4[hipThreadIdx_x + 2];
    __syncthreads();
    if (hipThreadIdx_x < 1) partialSumLDSRowPtr_f4[hipThreadIdx_x] += partialSumLDSRowPtr_f4[hipThreadIdx_x + 1];
    __syncthreads();

    if (hipThreadIdx_x == 0)
    {
        // Unrolled loop to run a vectorized reduction of 64 floats on 16 threads per block in y dimension
        if (hipThreadIdx_y < 8) partialSumLDSRowPtr_f4[0] += partialSumLDSRowPtr_f4[128];
        __syncthreads();
        if (hipThreadIdx_y < 4) partialSumLDSRowPtr_f4[0] += partialSumLDSRowPtr_f4[64];
        __syncthreads();
        if (hipThreadIdx_y < 2) partialSumLDSRowPtr_f4[0] += partialSumLDSRowPtr_f4[32];
        __syncthreads();
        if (hipThreadIdx_y < 1) partialSumLDSRowPtr_f4[0] += partialSumLDSRowPtr_f4[16];
        __syncthreads();

        // Final reduction of float4 vector to single float
        if (hipThreadIdx_y == 0)
            imageSumArr[(hipBlockIdx_z * hipGridDim_y + hipBlockIdx_y) * hipGridDim_x + hipBlockIdx_x] = partialSumLDSRowPtr_f4[0].x +
                                                                                                         partialSumLDSRowPtr_f4[0].y +
                                                                                                         partialSumLDSRowPtr_f4[0].z +
                                                                                                         partialSumLDSRowPtr_f4[0].w;
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

    printf("\n\nKernel Launch:");
    printf("\nlx, ly, lz = %d, %d, %d", localThreads_x, localThreads_y, localThreads_z);
    printf("\ngx, gy, gz = %d, %d, %d", (int)ceil((float)globalThreads_x/localThreads_x), (int)ceil((float)globalThreads_y/localThreads_y), (int)ceil((float)globalThreads_z/localThreads_z));
    printf("\n\n");

    float *imagePartialSumArr;
    imagePartialSumArr = handle.GetInitHandle()->mem.mgpu.maskArr.floatmem;

    Rpp32u outputSize = gridDim_x * gridDim_y * gridDim_z;
    Rpp32f output[outputSize];
    hipMemset(imagePartialSumArr, 0, outputSize * sizeof(float));
    hipMemset(imageSumArr, 0, gridDim_z * sizeof(float));

    printf("\nstrides = %d, %d, %d, %d", srcDescPtr->strides.nStride, srcDescPtr->strides.cStride, srcDescPtr->strides.hStride, srcDescPtr->strides.wStride);
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
        hipDeviceSynchronize();
        hipMemcpy(output, imagePartialSumArr, outputSize * sizeof(float), hipMemcpyDeviceToHost);
        printf("\n\nimagePartialSumArr: ");
        for (int i = 0; i < outputSize; i++)
        {
            printf(" %0.3f ", output[i]);
        }

        hipLaunchKernelGGL(image_sum_grid_result_tensor,
                           dim3(1, 1, gridDim_z),
                           dim3(256, 1, 1),
                           0,
                           handle.GetStream(),
                           imagePartialSumArr,
                           gridDim_x * gridDim_y,
                           imageSumArr);
        hipDeviceSynchronize();
        hipMemcpy(output, imageSumArr, gridDim_z * sizeof(float), hipMemcpyDeviceToHost);
        printf("\n\nimageSumArr: ");
        for (int i = 0; i < gridDim_z; i++)
        {
            printf(" %0.3f ", output[i]);
        }

    }

    return RPP_SUCCESS;
}
