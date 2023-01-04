#include <hip/hip_runtime.h>
#include "rpp_hip_common.hpp"

template <typename T, typename U>
__global__ void image_sum_pln1_tensor(T *srcPtr,
                                      uint2 srcStridesNH,
                                      U *imageSumArr,
                                      RpptROIPtr roiTensorPtrSrc)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    __shared__ float partialSumLDS[16][64];                                 // 16 rows of src, 64 cols of src in a 16 x 16 thread block
    float *partialSumLDSRowPtr = &partialSumLDS[hipThreadIdx_y][0];         // float pointer to beginning of each row in LDS
    float4 *partialSumLDSRowPtr_f4 = (float4 *)partialSumLDSRowPtr;         // float4 pointer to beginning of each row in LDS
    partialSumLDSRowPtr_f4[hipThreadIdx_x] = (float4) 0.0f;                 // vectorized initialization of LDS to 0

    if ((id_y >= roiTensorPtrSrc[id_z].xywhROI.roiHeight) || (id_x >= roiTensorPtrSrc[id_z].xywhROI.roiWidth))
    {
        return;
    }

    // __shared__ float partialSumLDS[16][64];                               // 16 rows of src, 64 cols of src in a 16 x 16 thread block
    // float *partialSumLDSRowPtr = &partialSumLDS[hipThreadIdx_y][0];
    // float4 *partialSumLDSRowPtr_f4 = (float4 *)partialSumLDSRowPtr;
    // partialSumLDSRowPtr_f4[hipThreadIdx_x] = (float4) 0.0f;
    // uint srcIdx = (id_z * srcStridesNH.x) + ((id_y + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNH.y) + (roiTensorPtrSrc[id_z].xywhROI.xy.x);

    int xAlignedLength = roiTensorPtrSrc[id_z].xywhROI.roiWidth & ~7;       // 216 or 48
    int xDiff = roiTensorPtrSrc[id_z].xywhROI.roiWidth - xAlignedLength;    // 7 or 2
    // int xThreadLimit = xAlignedLength / 8;                                  // 27 or 6
    // if ((xAlignedLength < roiTensorPtrSrc[id_z].xywhROI.roiWidth) && (hipBlockIdx_x == 0) && (hipThreadIdx_x == 0))
    // {
    //     T *srcPtrUnaligned = srcPtr + srcIdx + xAlignedLength;

    //     for (int i = 0; i < xDiff; i++)
    //         *partialSumLDSRowPtr += (float)srcPtrUnaligned[i];

    //     // printf("\n\nInsideInitial - %f, %f, %f, %f", (float)partialSumLDSRowPtr_f4[0].x, (float)partialSumLDSRowPtr_f4[0].y, (float)partialSumLDSRowPtr_f4[0].z, (float)partialSumLDSRowPtr_f4[0].w);
    // }

    // __syncthreads();

    // if(id_x < xThreadLimit)
    // if(hipBlockIdx_x < xThreadLimit)
    {
        // srcIdx += id_x; // temp remove


        // __shared__ float partialSumLDS[16][64];                                 // 16 rows of src, 64 cols of src in a 16 x 16 thread block
        // float *partialSumLDSRowPtr = &partialSumLDS[hipThreadIdx_y][0];
        // float4 *partialSumLDSRowPtr_f4 = (float4 *)partialSumLDSRowPtr;
        // partialSumLDSRowPtr_f4[hipThreadIdx_x] = (float4) 0.0f;
        uint srcIdx = (id_z * srcStridesNH.x) + ((id_y + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNH.y) + (id_x + roiTensorPtrSrc[id_z].xywhROI.xy.x);

        d_float8 src_f8;
        rpp_hip_load8_and_unpack_to_float8(srcPtr + srcIdx, &src_f8);           // Loading 8 pixels to local mmemory
        if (id_x + 8 > roiTensorPtrSrc[id_z].xywhROI.roiWidth)
            for(int i = xDiff; i < 8; i++)
                src_f8.f1[i] = 0.0f;
        partialSumLDSRowPtr_f4[hipThreadIdx_x] += (src_f8.f4[0] + src_f8.f4[1]);   // Performing small work of vectorized addition of two f4s and storing in lds
        __syncthreads();

        // if((hipBlockIdx_x == 0) && (hipBlockIdx_y == 0) && (hipThreadIdx_y == 0) && (hipThreadIdx_z == 0))
        // {
        //     printf("\n\nid_x = %d, Src = %f, %f, %f, %f", id_x, (float)partialSumLDSRowPtr_f4[hipThreadIdx_x].x, (float)partialSumLDSRowPtr_f4[hipThreadIdx_x].y, (float)partialSumLDSRowPtr_f4[hipThreadIdx_x].z, (float)partialSumLDSRowPtr_f4[hipThreadIdx_x].w);
        // }
        // __syncthreads();

        // if((hipBlockIdx_x == 0) && (hipBlockIdx_y == 0) && (hipThreadIdx_y == 0) && (hipThreadIdx_x < 20) && (hipThreadIdx_z == 0))
        // {
        //     printf("\n\nsizeof(float4) = %d", (int)sizeof(float4));
        //     printf("\n\nInside Initial - hipThreadIdx_x - %zu - %f, %f, %f, %f", hipThreadIdx_x, (float)partialSumLDSRowPtr_f4[hipThreadIdx_x].x, (float)partialSumLDSRowPtr_f4[hipThreadIdx_x].y, (float)partialSumLDSRowPtr_f4[hipThreadIdx_x].z, (float)partialSumLDSRowPtr_f4[hipThreadIdx_x].w);
        //     printf("\n\nInside Initial 2 - hipThreadIdx_x - %zu - %f, %f, %f, %f", hipThreadIdx_x, (float)partialSumLDSRowPtr_f4[hipThreadIdx_x + 8].x, (float)partialSumLDSRowPtr_f4[hipThreadIdx_x + 8].y, (float)partialSumLDSRowPtr_f4[hipThreadIdx_x + 8].z, (float)partialSumLDSRowPtr_f4[hipThreadIdx_x + 8].w);
        // }




        if (hipThreadIdx_x < 8) partialSumLDSRowPtr_f4[hipThreadIdx_x] = partialSumLDSRowPtr_f4[hipThreadIdx_x] + partialSumLDSRowPtr_f4[hipThreadIdx_x + 8];
        __syncthreads();

        // if((hipBlockIdx_x == 0) && (hipBlockIdx_y == 0) && (hipThreadIdx_y == 0) && (hipThreadIdx_x == 2) && (hipThreadIdx_z == 0))
        // {
        //     printf("\n\n");
        //     printf("\n\nInside - %f, %f, %f, %f", (float)partialSumLDSRowPtr_f4[hipThreadIdx_x].x, (float)partialSumLDSRowPtr_f4[hipThreadIdx_x].y, (float)partialSumLDSRowPtr_f4[hipThreadIdx_x].z, (float)partialSumLDSRowPtr_f4[hipThreadIdx_x].w);
        //     // printf("\n\nInside2 - %f, %f, %f, %f", (float)partialSumLDSRowPtr_f4[4].x, (float)partialSumLDSRowPtr_f4[4].y, (float)partialSumLDSRowPtr_f4[4].z, (float)partialSumLDSRowPtr_f4[4].w);
        // }





        if (hipThreadIdx_x < 4) partialSumLDSRowPtr_f4[hipThreadIdx_x] += partialSumLDSRowPtr_f4[hipThreadIdx_x + 4];
        __syncthreads();

        // if((hipBlockIdx_x == 0) && (hipBlockIdx_y == 0) && (hipThreadIdx_y == 0) && (hipThreadIdx_x == 1) && (hipThreadIdx_z == 0))
        // {
        //     printf("\n\n");
        //     printf("\n\nInside - %f, %f, %f, %f", (float)partialSumLDSRowPtr_f4[hipThreadIdx_x].x, (float)partialSumLDSRowPtr_f4[hipThreadIdx_x].y, (float)partialSumLDSRowPtr_f4[hipThreadIdx_x].z, (float)partialSumLDSRowPtr_f4[hipThreadIdx_x].w);
        // }




        if (hipThreadIdx_x < 2) partialSumLDSRowPtr_f4[hipThreadIdx_x] += partialSumLDSRowPtr_f4[hipThreadIdx_x + 2];
        __syncthreads();

        // if((hipBlockIdx_x == 0) && (hipBlockIdx_y == 0) && (hipThreadIdx_y == 0) && (hipThreadIdx_x == 0) && (hipThreadIdx_z == 0))
        // {
        //     printf("\n\n");
        //     printf("\n\nInside - %f, %f, %f, %f", (float)partialSumLDSRowPtr_f4[hipThreadIdx_x].x, (float)partialSumLDSRowPtr_f4[hipThreadIdx_x].y, (float)partialSumLDSRowPtr_f4[hipThreadIdx_x].z, (float)partialSumLDSRowPtr_f4[hipThreadIdx_x].w);
        // }




        if (hipThreadIdx_x < 1) partialSumLDSRowPtr_f4[hipThreadIdx_x] += partialSumLDSRowPtr_f4[hipThreadIdx_x + 1];
        __syncthreads();

        if((hipBlockIdx_x == 0) && (hipBlockIdx_y == 0) && (hipThreadIdx_x == 0) && (hipThreadIdx_z == 0))
        {
            printf("\n\n");
            printf("\n\nInside - %f, %f, %f, %f", (float)partialSumLDSRowPtr_f4[hipThreadIdx_x].x, (float)partialSumLDSRowPtr_f4[hipThreadIdx_x].y, (float)partialSumLDSRowPtr_f4[hipThreadIdx_x].z, (float)partialSumLDSRowPtr_f4[hipThreadIdx_x].w);
        }


        if (hipThreadIdx_x == 0)
        {
            if (hipThreadIdx_y < 8) partialSumLDSRowPtr_f4[0] += partialSumLDSRowPtr_f4[128];
            __syncthreads();
            if (hipThreadIdx_y < 4) partialSumLDSRowPtr_f4[0] += partialSumLDSRowPtr_f4[64];
            __syncthreads();
            if (hipThreadIdx_y < 2) partialSumLDSRowPtr_f4[0] += partialSumLDSRowPtr_f4[32];
            __syncthreads();
            if (hipThreadIdx_y < 1) partialSumLDSRowPtr_f4[0] += partialSumLDSRowPtr_f4[16];
            __syncthreads();

            if (hipThreadIdx_y == 0)
                // printf("\n\nInside - hipBlockIdx_x, hipBlockIdx_y, hipGridDim_x, hipGridDim_y - %zu, %zu, %zu, %zu", hipBlockIdx_x, hipBlockIdx_y, hipGridDim_x, hipGridDim_y);
                imageSumArr[hipBlockIdx_y * hipGridDim_x + hipBlockIdx_x] = partialSumLDSRowPtr_f4[0].x + partialSumLDSRowPtr_f4[0].y + partialSumLDSRowPtr_f4[0].z + partialSumLDSRowPtr_f4[0].w;
        }
    }

    // if ((hipThreadIdx_x < 1) && (hipThreadIdx_y < 8))
    // {
    //     partialSumLDSRowPtr_f4[0] += partialSumLDSRowPtr_f4[128];
    // }
    // __syncthreads();

    // if ((hipThreadIdx_x < 1) && (hipThreadIdx_y < 4))
    // {
    //     partialSumLDSRowPtr_f4[0] += partialSumLDSRowPtr_f4[64];
    // }
    // __syncthreads();

    // if ((hipThreadIdx_x < 1) && (hipThreadIdx_y < 2))
    // {
    //     partialSumLDSRowPtr_f4[0] += partialSumLDSRowPtr_f4[32];
    // }
    // __syncthreads();

    // if ((hipThreadIdx_x < 1) && (hipThreadIdx_y < 1))
    // {
    //     partialSumLDSRowPtr_f4[0] += partialSumLDSRowPtr_f4[16];
    // }
    // __syncthreads();

    // if ((hipThreadIdx_x == 0) && (hipThreadIdx_y == 0))
    // {
    //     imageSumArr[hipBlockIdx_y * hipBlockDim_x + hipBlockIdx_x] = partialSumLDSRowPtr_f4[0].x + partialSumLDSRowPtr_f4[0].y + partialSumLDSRowPtr_f4[0].z + partialSumLDSRowPtr_f4[0].w;
    // }















    // // uint dstIdx = (id_z * dstStridesNCH.x) + (id_y * dstStridesNCH.z) + id_x;

    // float4 alpha_f4 = (float4)(alpha[id_z]);
    // float4 beta_f4 = (float4)(beta[id_z]);

    // d_float8 src_f8, dst_f8;

    // rpp_hip_load8_and_unpack_to_float8(srcPtr + srcIdx, &src_f8);
    // image_sum_hip_compute(srcPtr, &src_f8, &dst_f8, &alpha_f4, &beta_f4);
    // rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &dst_f8);

    // if (channelsDst == 3)
    // {
    //     srcIdx += srcStridesNCH.y;
    //     dstIdx += dstStridesNCH.y;

    //     rpp_hip_load8_and_unpack_to_float8(srcPtr + srcIdx, &src_f8);
    //     image_sum_hip_compute(srcPtr, &src_f8, &dst_f8, &alpha_f4, &beta_f4);
    //     rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &dst_f8);

    //     srcIdx += srcStridesNCH.y;
    //     dstIdx += dstStridesNCH.y;

    //     rpp_hip_load8_and_unpack_to_float8(srcPtr + srcIdx, &src_f8);
    //     image_sum_hip_compute(srcPtr, &src_f8, &dst_f8, &alpha_f4, &beta_f4);
    //     rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &dst_f8);
    // }
}

// template <typename T>
// __global__ void image_sum_pkd_tensor(T *srcPtr,
//                                       uint2 srcStridesNH,
//                                       T *dstPtr,
//                                       uint2 dstStridesNH,
//                                       float *alpha,
//                                       float *beta,
//                                       RpptROIPtr roiTensorPtrSrc)
// {
//     int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
//     int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
//     int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

//     if ((id_y >= roiTensorPtrSrc[id_z].xywhROI.roiHeight) || (id_x >= roiTensorPtrSrc[id_z].xywhROI.roiWidth * 3))
//     {
//         return;
//     }

//     uint srcIdx = (id_z * srcStridesNH.x) + ((id_y + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNH.y) + (id_x + roiTensorPtrSrc[id_z].xywhROI.xy.x * 3);
//     uint dstIdx = (id_z * dstStridesNH.x) + (id_y * dstStridesNH.y) + id_x;

//     float4 alpha_f4 = (float4)alpha[id_z];
//     float4 beta_f4 = (float4)beta[id_z];

//     d_float8 src_f8, dst_f8;

//     rpp_hip_load8_and_unpack_to_float8(srcPtr + srcIdx, &src_f8);
//     image_sum_hip_compute(srcPtr, &src_f8, &dst_f8, &alpha_f4, &beta_f4);
//     rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &dst_f8);
// }

// template <typename T>
// __global__ void image_sum_pln_tensor(T *srcPtr,
//                                       uint3 srcStridesNCH,
//                                       T *dstPtr,
//                                       uint3 dstStridesNCH,
//                                       int channelsDst,
//                                       float *alpha,
//                                       float *beta,
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

//     float4 alpha_f4 = (float4)(alpha[id_z]);
//     float4 beta_f4 = (float4)(beta[id_z]);

//     d_float8 src_f8, dst_f8;

//     rpp_hip_load8_and_unpack_to_float8(srcPtr + srcIdx, &src_f8);
//     image_sum_hip_compute(srcPtr, &src_f8, &dst_f8, &alpha_f4, &beta_f4);
//     rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &dst_f8);

//     if (channelsDst == 3)
//     {
//         srcIdx += srcStridesNCH.y;
//         dstIdx += dstStridesNCH.y;

//         rpp_hip_load8_and_unpack_to_float8(srcPtr + srcIdx, &src_f8);
//         image_sum_hip_compute(srcPtr, &src_f8, &dst_f8, &alpha_f4, &beta_f4);
//         rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &dst_f8);

//         srcIdx += srcStridesNCH.y;
//         dstIdx += dstStridesNCH.y;

//         rpp_hip_load8_and_unpack_to_float8(srcPtr + srcIdx, &src_f8);
//         image_sum_hip_compute(srcPtr, &src_f8, &dst_f8, &alpha_f4, &beta_f4);
//         rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &dst_f8);
//     }
// }

// template <typename T>
// __global__ void image_sum_pkd3_pln3_tensor(T *srcPtr,
//                                             uint2 srcStridesNH,
//                                             T *dstPtr,
//                                             uint3 dstStridesNCH,
//                                             float *alpha,
//                                             float *beta,
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

//     float4 alpha_f4 = (float4)alpha[id_z];
//     float4 beta_f4 = (float4)beta[id_z];

//     d_float24 src_f24, dst_f24;

//     rpp_hip_load24_pkd3_and_unpack_to_float24_pln3(srcPtr + srcIdx, &src_f24);
//     image_sum_hip_compute(srcPtr, &src_f24.f8[0], &dst_f24.f8[0], &alpha_f4, &beta_f4);
//     image_sum_hip_compute(srcPtr, &src_f24.f8[1], &dst_f24.f8[1], &alpha_f4, &beta_f4);
//     image_sum_hip_compute(srcPtr, &src_f24.f8[2], &dst_f24.f8[2], &alpha_f4, &beta_f4);
//     rpp_hip_pack_float24_pln3_and_store24_pln3(dstPtr + dstIdx, dstStridesNCH.y, &dst_f24);
// }

// template <typename T>
// __global__ void image_sum_pln3_pkd3_tensor(T *srcPtr,
//                                             uint3 srcStridesNCH,
//                                             T *dstPtr,
//                                             uint2 dstStridesNH,
//                                             float *alpha,
//                                             float *beta,
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

//     float4 alpha_f4 = (float4)(alpha[id_z]);
//     float4 beta_f4 = (float4)(beta[id_z]);

//     d_float24 src_f24, dst_f24;

//     rpp_hip_load24_pln3_and_unpack_to_float24_pkd3(srcPtr + srcIdx, srcStridesNCH.y, &src_f24);
//     image_sum_hip_compute(srcPtr, &src_f24.f8[0], &dst_f24.f8[0], &alpha_f4, &beta_f4);
//     image_sum_hip_compute(srcPtr, &src_f24.f8[1], &dst_f24.f8[1], &alpha_f4, &beta_f4);
//     image_sum_hip_compute(srcPtr, &src_f24.f8[2], &dst_f24.f8[2], &alpha_f4, &beta_f4);
//     rpp_hip_pack_float24_pkd3_and_store24_pkd3(dstPtr + dstIdx, &dst_f24);
// }

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



    printf("\n\nKernel Launch:");
    printf("\nlx, ly, lz = %d, %d, %d", localThreads_x, localThreads_y, localThreads_z);
    printf("\ngx, gy, gz = %d, %d, %d", (int)ceil((float)globalThreads_x/localThreads_x), (int)ceil((float)globalThreads_y/localThreads_y), (int)ceil((float)globalThreads_z/localThreads_z));
    printf("\n\n");

    float *imagePartialSumArr;
    imagePartialSumArr = handle.GetInitHandle()->mem.mgpu.maskArr.floatmem;

    Rpp32u outputSize = ceil((float)globalThreads_x/localThreads_x) * ceil((float)globalThreads_y/localThreads_y);
    Rpp32f output[outputSize];
    hipMemset(imagePartialSumArr, 0, outputSize * sizeof(float));

    printf("\nstrides = %d, %d, %d, %d", srcDescPtr->strides.nStride, srcDescPtr->strides.cStride, srcDescPtr->strides.hStride, srcDescPtr->strides.wStride);
    if ((srcDescPtr->c == 1) && (srcDescPtr->layout == RpptLayout::NCHW))
    {
        hipLaunchKernelGGL(image_sum_pln1_tensor,
                           dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
                           dim3(localThreads_x, localThreads_y, localThreads_z),
                           0,
                           handle.GetStream(),
                           srcPtr,
                           make_uint2(srcDescPtr->strides.nStride, srcDescPtr->strides.hStride),
                           imagePartialSumArr,
                           roiTensorPtrSrc);
        hipDeviceSynchronize();
        hipMemcpy(output, imagePartialSumArr, outputSize * sizeof(float), hipMemcpyDeviceToHost);
        printf("\n\n");
        for (int i = 0; i < outputSize * 2; i++)
        {
            printf(" %0.3f ", output[i]);
        }
    }
    else if (srcDescPtr->c == 3)
    {
        // if ((srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
        // {
        //     hipLaunchKernelGGL(image_sum_pkd3_pln3_tensor,
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
        //                        roiTensorPtrSrc);
        // }
        // else if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
        // {
        //     globalThreads_x = (srcDescPtr->strides.hStride + 7) >> 3;
        //     hipLaunchKernelGGL(image_sum_pln3_pkd3_tensor,
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
        //                        roiTensorPtrSrc);
        // }
        // else if ((srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NHWC))
        // {
        //     hipLaunchKernelGGL(image_sum_pkd_tensor,
        //                        dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
        //                        dim3(localThreads_x, localThreads_y, localThreads_z),
        //                        0,
        //                        handle.GetStream(),
        //                        srcPtr,
        //                        make_uint2(srcDescPtr->strides.nStride, srcDescPtr->strides.hStride),
        //                        dstPtr,
        //                        make_uint2(dstDescPtr->strides.nStride, dstDescPtr->strides.hStride),
        //                        handle.GetInitHandle()->mem.mgpu.floatArr[0].floatmem,
        //                        handle.GetInitHandle()->mem.mgpu.floatArr[1].floatmem,
        //                        roiTensorPtrSrc);
        // }
        // else if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
        // {

        // }
    }

    return RPP_SUCCESS;
}
