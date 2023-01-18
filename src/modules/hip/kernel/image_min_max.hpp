#include <hip/hip_runtime.h>
#include "rpp_hip_common.hpp"

__global__ void image_min_max_grid_result_tensor(float *srcPtr,
                                                 uint xBufferLength,
                                                 float *dstPtr)
{
    int id_x = hipThreadIdx_x * 8;
    int id_z = hipBlockIdx_z;

    __shared__ float partialMinMaxLDS[1024];                        // 512 * 2 floats of src in a 512 x 1 thread block
    float2 *partialMinMaxLDS_f2 = (float2 *)partialMinMaxLDS;       // float2 pointer to beginning of buffer in LDS

    uint srcIdx = (id_z * xBufferLength);
    float2 *srcPtr_f2 = (float2 *)srcPtr;
    float2 srcRef_f2 = srcPtr_f2[srcIdx];
    partialMinMaxLDS_f2[hipThreadIdx_x] = srcRef_f2;

    if (id_x >= xBufferLength)
    {
        return;
    }

    int xAlignedLength = xBufferLength & ~3;                        // alignedLength for vectorized global loads
    int xDiff = xBufferLength - xAlignedLength;                     // difference between bufferLength and alignedLength
    srcIdx += id_x;

    d_float8 src_f8;
    rpp_hip_load8_and_unpack_to_float8(srcPtr + srcIdx, &src_f8);   // load 8 pixels to local mmemory
    if (id_x + 8 > xBufferLength * 2)
        for(int i = xDiff; i < 4; i++)
            src_f8.f2[i] = srcRef_f2;                               // reset invalid loads to srcRef_f2
    partialMinMaxLDS_f2[hipThreadIdx_x].x = fminf(src_f8.f1[0], fminf(src_f8.f1[2], fminf(src_f8.f1[4], src_f8.f1[6])));    // perform small work of min/max a vector f8 and store in LDS
    partialMinMaxLDS_f2[hipThreadIdx_x].y = fmaxf(src_f8.f1[1], fmaxf(src_f8.f1[3], fmaxf(src_f8.f1[5], src_f8.f1[7])));    // perform small work of min/max a vector f8 and store in LDS
    __syncthreads();                                                // syncthreads after LDS load

    // Vectorized float2 reduction of 1024 floats on 512 threads per block in x dimension
    for (int threadMax = 256; threadMax >= 1; threadMax /= 2)
    {
        if (hipThreadIdx_x < threadMax)
        {
            partialMinMaxLDS_f2[hipThreadIdx_x].x = fminf(partialMinMaxLDS_f2[hipThreadIdx_x].x, partialMinMaxLDS_f2[hipThreadIdx_x + threadMax].x);
            partialMinMaxLDS_f2[hipThreadIdx_x].y = fmaxf(partialMinMaxLDS_f2[hipThreadIdx_x].y, partialMinMaxLDS_f2[hipThreadIdx_x + threadMax].y);
        }
        __syncthreads();
    }

    // Final store of float2 vector min_max to dst
    if (hipThreadIdx_x == 0)
    {
        float2 *dstPtr_f2 = (float2 *)dstPtr;
        dstPtr_f2[hipBlockIdx_z] = partialMinMaxLDS_f2[0];
    }
}

template <typename T, typename U>
__global__ void image_min_max_pln1_tensor(T *srcPtr,
                                          uint2 srcStridesNH,
                                          U *imageMinMaxArr,
                                          RpptROIPtr roiTensorPtrSrc)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    __shared__ float partialMinMaxLDS[16][128];                                                         // 16 rows of src, 64 cols of src in a 16 x 16 thread block (each producing outputs for min and max)
    float *partialMinMaxLDSRowPtr = &partialMinMaxLDS[hipThreadIdx_y][0];                               // float pointer to beginning of each row in LDS
    d_float8 *partialMinMaxLDSRowPtr_f8 = (d_float8 *)partialMinMaxLDSRowPtr;                           // d_float8 pointer to beginning of each row in LDS

    uint srcIdx = (id_z * srcStridesNH.x);
    float srcRef = srcPtr[srcIdx];
    partialMinMaxLDSRowPtr_f8[hipThreadIdx_x].f4[0] = (float4) srcRef;                                  // vectorized float4 initialization of LDS to 0 using all 16x16 threads
    partialMinMaxLDSRowPtr_f8[hipThreadIdx_x].f4[1] = partialMinMaxLDSRowPtr_f8[hipThreadIdx_x].f4[0];  // vectorized float4 initialization of LDS to 0 using all 16x16 threads

    if ((id_y >= roiTensorPtrSrc[id_z].xywhROI.roiHeight) || (id_x >= roiTensorPtrSrc[id_z].xywhROI.roiWidth))
    {
        return;
    }

    int xAlignedLength = roiTensorPtrSrc[id_z].xywhROI.roiWidth & ~7;                                   // alignedLength for vectorized global loads
    int xDiff = roiTensorPtrSrc[id_z].xywhROI.roiWidth - xAlignedLength;                                // difference between roiWidth and alignedLength
    srcIdx += (((id_y + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNH.y) + (id_x + roiTensorPtrSrc[id_z].xywhROI.xy.x));

    d_float8 src_f8;
    rpp_hip_load8_and_unpack_to_float8(srcPtr + srcIdx, &src_f8);                                       // load 8 pixels to local mmemory
    if (id_x + 8 > roiTensorPtrSrc[id_z].xywhROI.roiWidth)
        for(int i = xDiff; i < 8; i++)
            src_f8.f1[i] = srcRef;                                                                      // reset invalid loads to srcRef
    rpp_hip_math_minmax4(src_f8.f4[0], src_f8.f4[1], partialMinMaxLDSRowPtr_f8[hipThreadIdx_x]);        // perform small work of finding minmax for two vector f4s and store in LDS
    __syncthreads();                                                                                    // syncthreads after LDS load

    // Vectorized float4 reduction of 64 floats on 16 threads per block in x dimension (for every y dimension)
    for (int threadMax = 8; threadMax >= 1; threadMax /= 2)
    {
        if (hipThreadIdx_x < threadMax)
            rpp_hip_math_minmax4(partialMinMaxLDSRowPtr_f8[hipThreadIdx_x], partialMinMaxLDSRowPtr_f8[hipThreadIdx_x + threadMax], partialMinMaxLDSRowPtr_f8[hipThreadIdx_x]);
        __syncthreads();
    }

    if (hipThreadIdx_x == 0)
    {
        // Vectorized float4 reduction of 128 floats on 16 threads per block in y dimension
        for (int threadMax = 8, increment = 128; threadMax >= 1; threadMax /= 2, increment /= 2)
        {
            if (hipThreadIdx_x < threadMax)
                rpp_hip_math_minmax4(partialMinMaxLDSRowPtr_f8[hipThreadIdx_x], partialMinMaxLDSRowPtr_f8[hipThreadIdx_x + increment], partialMinMaxLDSRowPtr_f8[hipThreadIdx_x]);
            __syncthreads();
        }

        // Final reduction of d_float8 vector to float2
        if (hipThreadIdx_y == 0)
        {
            float2 *imageMinMaxArr_f2 = (float2 *)imageMinMaxArr;
            float2 minmax_f2 = make_float2(
                fminf(fminf(fminf(partialMinMaxLDSRowPtr_f8[0].f1[0], partialMinMaxLDSRowPtr_f8[0].f1[1]), partialMinMaxLDSRowPtr_f8[0].f1[2]), partialMinMaxLDSRowPtr_f8[0].f1[3]),
                fmaxf(fmaxf(fmaxf(partialMinMaxLDSRowPtr_f8[0].f1[4], partialMinMaxLDSRowPtr_f8[0].f1[5]), partialMinMaxLDSRowPtr_f8[0].f1[6]), partialMinMaxLDSRowPtr_f8[0].f1[7])
            );
            imageMinMaxArr_f2[(hipBlockIdx_z * hipGridDim_y + hipBlockIdx_y) * hipGridDim_x + hipBlockIdx_x] = minmax_f2;
        }
    }
}

template <typename T, typename U>
RppStatus hip_exec_image_min_max_tensor(T *srcPtr,
                                        RpptDescPtr srcDescPtr,
                                        U *imageMinMaxArr,
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

    Rpp32u imagePartialMinMaxArrLength = gridDim_x * gridDim_y * gridDim_z * 2;
    float *imagePartialMinMaxArr;
    imagePartialMinMaxArr = handle.GetInitHandle()->mem.mgpu.maskArr.floatmem;
    hipMemset(imagePartialMinMaxArr, 0, imagePartialMinMaxArrLength * sizeof(float));

    if ((srcDescPtr->c == 1) && (srcDescPtr->layout == RpptLayout::NCHW))
    {
        hipLaunchKernelGGL(image_min_max_pln1_tensor,
                           dim3(gridDim_x, gridDim_y, gridDim_z),
                           dim3(localThreads_x, localThreads_y, localThreads_z),
                           0,
                           handle.GetStream(),
                           srcPtr,
                           make_uint2(srcDescPtr->strides.nStride, srcDescPtr->strides.hStride),
                           imagePartialMinMaxArr,
                           roiTensorPtrSrc);
        hipLaunchKernelGGL(image_min_max_grid_result_tensor,
                           dim3(1, 1, gridDim_z),
                           dim3(512, 1, 1),
                           0,
                           handle.GetStream(),
                           imagePartialMinMaxArr,
                           gridDim_x * gridDim_y,
                           imageMinMaxArr);
    }

    return RPP_SUCCESS;
}
