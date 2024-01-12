#include <hip/hip_runtime.h>
#include "rpp_hip_common.hpp"

// Vectorized dst->src mapping
template <typename T>
__global__ void transpose_generic_hip_tensor(T *srcPtr,
                                             uint *srcStrides,
                                             T *dstPtr,
                                             uint *dstStrides,
                                             uint *dstDims,
                                             uint dstNumDims,
                                             uint *permTensor)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;

    if(id_x >= dstStrides[0])
        return;

    int maxLength = dstStrides[0];
    int xDiff = maxLength - (maxLength & ~7);
    uint dstIdx = (id_y * *dstStrides++);
    uint srcIdx = (id_y * *srcStrides++);

    d_uint8 dstCoords[RPPT_MAX_DIMS], srcIdxs;
    uint4 idx0123 = make_uint4(id_x, id_x + 1, id_x + 2, id_x + 3);
    uint4 idx4567 = make_uint4(id_x + 4, id_x + 5, id_x + 6, id_x + 7);
    srcIdxs.ui4[0] = make_uint4(srcIdx, srcIdx, srcIdx, srcIdx);
    srcIdxs.ui4[1] = make_uint4(srcIdx, srcIdx, srcIdx, srcIdx);

    for (int i = 0; i < dstNumDims; i++)
    {
        dstCoords[i].ui4[0] = idx0123 / dstStrides[i] % dstDims[i];
        dstCoords[i].ui4[1] = idx4567 / dstStrides[i] % dstDims[i];
    }
    for (int i = 0; i < dstNumDims; i++)
    {
        for (int j = 0; j < 8; j++)
            srcIdxs.ui1[j] += (dstCoords[permTensor[i]].ui1[j] * srcStrides[permTensor[permTensor[i]]]);
        dstIdx += (dstCoords[i].ui1[0] * dstStrides[i]);
    }
    if((id_x + 8) > maxLength)
        for(int i = xDiff; i < 8; i++)
            srcIdxs.ui1[i] += maxLength;

    d_float8 dst_f8;
    dst_f8.f1[0] = (float)srcPtr[srcIdxs.ui1[0]]; // load 8 src pixels
    dst_f8.f1[1] = (float)srcPtr[srcIdxs.ui1[1]];
    dst_f8.f1[2] = (float)srcPtr[srcIdxs.ui1[2]];
    dst_f8.f1[3] = (float)srcPtr[srcIdxs.ui1[3]];
    dst_f8.f1[4] = (float)srcPtr[srcIdxs.ui1[4]];
    dst_f8.f1[5] = (float)srcPtr[srcIdxs.ui1[5]];
    dst_f8.f1[6] = (float)srcPtr[srcIdxs.ui1[6]];
    dst_f8.f1[7] = (float)srcPtr[srcIdxs.ui1[7]];

    rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &dst_f8);
}

template <typename T>
RppStatus hip_exec_transpose_generic_tensor(T *srcPtr,
                                            RpptGenericDescPtr srcGenericDescPtr,
                                            T *dstPtr,
                                            RpptGenericDescPtr dstGenericDescPtr,
                                            Rpp32u *permTensor,
                                            Rpp32u *roiTensor,
                                            rpp::Handle& handle)
{

    bool copyInput = true;
        for(int i = 0; i < dstGenericDescPtr->numDims - 1; i++)
            copyInput *= (permTensor[i] == i);

    if (copyInput)
        hipMemcpy(dstPtr, srcPtr, handle.GetBatchSize() * dstGenericDescPtr->strides[0] * sizeof(T), hipMemcpyDeviceToDevice);
    else
    {
        int globalThreads_x = (dstGenericDescPtr->strides[0] + 7) >> 3;
        int globalThreads_y = handle.GetBatchSize();
        int globalThreads_z = 1;

        hipLaunchKernelGGL(transpose_generic_hip_tensor,
                        dim3(ceil((float)globalThreads_x/1024), ceil((float)globalThreads_y/LOCAL_THREADS_Y_1DIM), ceil((float)globalThreads_z/LOCAL_THREADS_Z_1DIM)),
                        dim3(1024, LOCAL_THREADS_Y_1DIM, LOCAL_THREADS_Z_1DIM),
                        0,
                        handle.GetStream(),
                        srcPtr,
                        srcGenericDescPtr->strides,
                        dstPtr,
                        dstGenericDescPtr->strides,
                        dstGenericDescPtr->dims + 1,
                        dstGenericDescPtr->numDims - 1,
                        permTensor);
    }

    return RPP_SUCCESS;
}
