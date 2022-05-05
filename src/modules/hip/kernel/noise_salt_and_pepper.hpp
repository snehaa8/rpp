#include <hip/hip_runtime.h>
#include "hip/rpp_hip_common.hpp"
#include "func_specific/rng_seed_stream.hpp"

__device__ void salt_and_pepper_noise_1_hip_compute(float *src, float *dst, float noiseProbability, float saltProbability, float salt, float pepper, float randomNumberFloat)
{
    if (randomNumberFloat > noiseProbability)
        *dst = *src;
    else
        *dst = ((randomNumberFloat <= saltProbability) ? salt : pepper);
}

__device__ void salt_and_pepper_noise_8_hip_compute(d_float8 *src_f8, d_float8 *dst_f8, float noiseProbability, float saltProbability, float salt, float pepper, d_float8 *randomNumbers_f8)
{
    salt_and_pepper_noise_1_hip_compute(&src_f8->f1[0], &dst_f8->f1[0], noiseProbability, saltProbability, salt, pepper, randomNumbers_f8->f1[0]);
    salt_and_pepper_noise_1_hip_compute(&src_f8->f1[1], &dst_f8->f1[1], noiseProbability, saltProbability, salt, pepper, randomNumbers_f8->f1[1]);
    salt_and_pepper_noise_1_hip_compute(&src_f8->f1[2], &dst_f8->f1[2], noiseProbability, saltProbability, salt, pepper, randomNumbers_f8->f1[2]);
    salt_and_pepper_noise_1_hip_compute(&src_f8->f1[3], &dst_f8->f1[3], noiseProbability, saltProbability, salt, pepper, randomNumbers_f8->f1[3]);
    salt_and_pepper_noise_1_hip_compute(&src_f8->f1[4], &dst_f8->f1[4], noiseProbability, saltProbability, salt, pepper, randomNumbers_f8->f1[4]);
    salt_and_pepper_noise_1_hip_compute(&src_f8->f1[5], &dst_f8->f1[5], noiseProbability, saltProbability, salt, pepper, randomNumbers_f8->f1[5]);
    salt_and_pepper_noise_1_hip_compute(&src_f8->f1[6], &dst_f8->f1[6], noiseProbability, saltProbability, salt, pepper, randomNumbers_f8->f1[6]);
    salt_and_pepper_noise_1_hip_compute(&src_f8->f1[7], &dst_f8->f1[7], noiseProbability, saltProbability, salt, pepper, randomNumbers_f8->f1[7]);
}

__device__ void salt_and_pepper_noise_adjusted_input_hip_compute(uchar *srcPtr, float *saltValue, float *pepperValue) { *saltValue *= 255.0f; *pepperValue *= 255.0f; }
__device__ void salt_and_pepper_noise_adjusted_input_hip_compute(float *srcPtr, float *saltValue, float *pepperValue) {}
__device__ void salt_and_pepper_noise_adjusted_input_hip_compute(schar *srcPtr, float *saltValue, float *pepperValue) { *saltValue = (*saltValue * 255.0f) - 128.0f; *pepperValue = (*pepperValue * 255.0f) - 128.0f; }
__device__ void salt_and_pepper_noise_adjusted_input_hip_compute(half *srcPtr, float *saltValue, float *pepperValue) {}

template <typename T>
__global__ void salt_and_pepper_noise_pkd_tensor(T *srcPtr,
                                                 uint2 srcStridesNH,
                                                 T *dstPtr,
                                                 uint2 dstStridesNH,
                                                 float *noiseProbabilityTensor,
                                                 float *saltProbabilityTensor,
                                                 float *saltValueTensor,
                                                 float *pepperValueTensor,
                                                 RpptXorwowState *xorwowInitialStatePtr,
                                                 uint *xorwowSeedStream,
                                                 RpptROIPtr roiTensorPtrSrc)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if ((id_y >= roiTensorPtrSrc[id_z].xywhROI.roiHeight) || (id_x >= roiTensorPtrSrc[id_z].xywhROI.roiWidth))
    {
        return;
    }

    uint srcIdx = (id_z * srcStridesNH.x) + ((id_y + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNH.y) + ((id_x + roiTensorPtrSrc[id_z].xywhROI.xy.x) * 3);
    uint dstIdx = (id_z * dstStridesNH.x) + (id_y * dstStridesNH.y) + id_x * 3;
    uint seedStreamIdx = (id_y * dstStridesNH.y) + (hipBlockIdx_x * hipBlockDim_x) + hipThreadIdx_x;

    float noiseProbability = noiseProbabilityTensor[id_z];
    float saltProbability = saltProbabilityTensor[id_z] * noiseProbability;
    float saltValue = saltValueTensor[id_z];
    float pepperValue = pepperValueTensor[id_z];

    RpptXorwowState xorwowState;
    uint xorwowSeed = (seedStreamIdx >= SEED_STREAM_MAX_SIZE) ? xorwowSeedStream[seedStreamIdx - SEED_STREAM_MAX_SIZE] : xorwowSeedStream[seedStreamIdx];
    xorwowState.x[0] = xorwowInitialStatePtr->x[0] + xorwowSeed;
    xorwowState.x[1] = xorwowInitialStatePtr->x[1] + xorwowSeed;
    xorwowState.x[2] = xorwowInitialStatePtr->x[2] + xorwowSeed;
    xorwowState.x[3] = xorwowInitialStatePtr->x[3] + xorwowSeed;
    xorwowState.x[4] = xorwowInitialStatePtr->x[4] + xorwowSeed;
    xorwowState.counter = xorwowInitialStatePtr->counter + xorwowSeed;

    d_float8 randomNumbers_f8;
    d_float24 src_f24, dst_f24;

    rpp_hip_rng_8_xorwow_f32(&xorwowState, &randomNumbers_f8);
    salt_and_pepper_noise_adjusted_input_hip_compute(srcPtr, &saltValue, &pepperValue);
    rpp_hip_load24_pkd3_and_unpack_to_float24_pln3(srcPtr + srcIdx, &src_f24);
    salt_and_pepper_noise_8_hip_compute(&src_f24.f8[0], &dst_f24.f8[0], noiseProbability, saltProbability, saltValue, pepperValue, &randomNumbers_f8);
    salt_and_pepper_noise_8_hip_compute(&src_f24.f8[1], &dst_f24.f8[1], noiseProbability, saltProbability, saltValue, pepperValue, &randomNumbers_f8);
    salt_and_pepper_noise_8_hip_compute(&src_f24.f8[2], &dst_f24.f8[2], noiseProbability, saltProbability, saltValue, pepperValue, &randomNumbers_f8);
    rpp_hip_pack_float24_pln3_and_store24_pkd3(dstPtr + dstIdx, &dst_f24);
}

template <typename T>
__global__ void salt_and_pepper_noise_pln_tensor(T *srcPtr,
                                                 uint3 srcStridesNCH,
                                                 T *dstPtr,
                                                 uint3 dstStridesNCH,
                                                 int channelsDst,
                                                 float *noiseProbabilityTensor,
                                                 float *saltProbabilityTensor,
                                                 float *saltValueTensor,
                                                 float *pepperValueTensor,
                                                 RpptXorwowState *xorwowInitialStatePtr,
                                                 uint *xorwowSeedStream,
                                                 RpptROIPtr roiTensorPtrSrc)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if ((id_y >= roiTensorPtrSrc[id_z].xywhROI.roiHeight) || (id_x >= roiTensorPtrSrc[id_z].xywhROI.roiWidth))
    {
        return;
    }

    uint srcIdx = (id_z * srcStridesNCH.x) + ((id_y + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNCH.z) + (id_x + roiTensorPtrSrc[id_z].xywhROI.xy.x);
    uint dstIdx = (id_z * dstStridesNCH.x) + (id_y * dstStridesNCH.z) + id_x;
    uint seedStreamIdx = (id_y * dstStridesNCH.z) + (hipBlockIdx_x * hipBlockDim_x) + hipThreadIdx_x;

    float noiseProbability = noiseProbabilityTensor[id_z];
    float saltProbability = saltProbabilityTensor[id_z] * noiseProbability;
    float saltValue = saltValueTensor[id_z];
    float pepperValue = pepperValueTensor[id_z];

    RpptXorwowState xorwowState;
    uint xorwowSeed = (seedStreamIdx >= SEED_STREAM_MAX_SIZE) ? xorwowSeedStream[seedStreamIdx - SEED_STREAM_MAX_SIZE] : xorwowSeedStream[seedStreamIdx];
    xorwowState.x[0] = xorwowInitialStatePtr->x[0] + xorwowSeed;
    xorwowState.x[1] = xorwowInitialStatePtr->x[1] + xorwowSeed;
    xorwowState.x[2] = xorwowInitialStatePtr->x[2] + xorwowSeed;
    xorwowState.x[3] = xorwowInitialStatePtr->x[3] + xorwowSeed;
    xorwowState.x[4] = xorwowInitialStatePtr->x[4] + xorwowSeed;
    xorwowState.counter = xorwowInitialStatePtr->counter + xorwowSeed;

    d_float8 src_f8, dst_f8, randomNumbers_f8;
    rpp_hip_rng_8_xorwow_f32(&xorwowState, &randomNumbers_f8);
    salt_and_pepper_noise_adjusted_input_hip_compute(srcPtr, &saltValue, &pepperValue);

    rpp_hip_load8_and_unpack_to_float8(srcPtr + srcIdx, &src_f8);
    salt_and_pepper_noise_8_hip_compute(&src_f8, &dst_f8, noiseProbability, saltProbability, saltValue, pepperValue, &randomNumbers_f8);
    rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &dst_f8);

    if (channelsDst == 3)
    {
        srcIdx += srcStridesNCH.y;
        dstIdx += dstStridesNCH.y;

        rpp_hip_load8_and_unpack_to_float8(srcPtr + srcIdx, &src_f8);
        salt_and_pepper_noise_8_hip_compute(&src_f8, &dst_f8, noiseProbability, saltProbability, saltValue, pepperValue, &randomNumbers_f8);
        rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &dst_f8);

        srcIdx += srcStridesNCH.y;
        dstIdx += dstStridesNCH.y;

        rpp_hip_load8_and_unpack_to_float8(srcPtr + srcIdx, &src_f8);
        salt_and_pepper_noise_8_hip_compute(&src_f8, &dst_f8, noiseProbability, saltProbability, saltValue, pepperValue, &randomNumbers_f8);
        rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &dst_f8);
    }
}

template <typename T>
__global__ void salt_and_pepper_noise_pkd3_pln3_tensor(T *srcPtr,
                                                       uint2 srcStridesNH,
                                                       T *dstPtr,
                                                       uint3 dstStridesNCH,
                                                       float *noiseProbabilityTensor,
                                                       float *saltProbabilityTensor,
                                                       float *saltValueTensor,
                                                       float *pepperValueTensor,
                                                       RpptXorwowState *xorwowInitialStatePtr,
                                                       uint *xorwowSeedStream,
                                                       RpptROIPtr roiTensorPtrSrc)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if ((id_y >= roiTensorPtrSrc[id_z].xywhROI.roiHeight) || (id_x >= roiTensorPtrSrc[id_z].xywhROI.roiWidth))
    {
        return;
    }

    uint srcIdx = (id_z * srcStridesNH.x) + ((id_y + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNH.y) + ((id_x + roiTensorPtrSrc[id_z].xywhROI.xy.x) * 3);
    uint dstIdx = (id_z * dstStridesNCH.x) + (id_y * dstStridesNCH.z) + id_x;
    uint seedStreamIdx = (id_y * dstStridesNCH.z) + (hipBlockIdx_x * hipBlockDim_x) + hipThreadIdx_x;

    float noiseProbability = noiseProbabilityTensor[id_z];
    float saltProbability = saltProbabilityTensor[id_z] * noiseProbability;
    float saltValue = saltValueTensor[id_z];
    float pepperValue = pepperValueTensor[id_z];

    RpptXorwowState xorwowState;
    uint xorwowSeed = (seedStreamIdx >= SEED_STREAM_MAX_SIZE) ? xorwowSeedStream[seedStreamIdx - SEED_STREAM_MAX_SIZE] : xorwowSeedStream[seedStreamIdx];
    xorwowState.x[0] = xorwowInitialStatePtr->x[0] + xorwowSeed;
    xorwowState.x[1] = xorwowInitialStatePtr->x[1] + xorwowSeed;
    xorwowState.x[2] = xorwowInitialStatePtr->x[2] + xorwowSeed;
    xorwowState.x[3] = xorwowInitialStatePtr->x[3] + xorwowSeed;
    xorwowState.x[4] = xorwowInitialStatePtr->x[4] + xorwowSeed;
    xorwowState.counter = xorwowInitialStatePtr->counter + xorwowSeed;

    d_float8 randomNumbers_f8;
    d_float24 src_f24, dst_f24;

    rpp_hip_rng_8_xorwow_f32(&xorwowState, &randomNumbers_f8);
    salt_and_pepper_noise_adjusted_input_hip_compute(srcPtr, &saltValue, &pepperValue);
    rpp_hip_load24_pkd3_and_unpack_to_float24_pln3(srcPtr + srcIdx, &src_f24);
    salt_and_pepper_noise_8_hip_compute(&src_f24.f8[0], &dst_f24.f8[0], noiseProbability, saltProbability, saltValue, pepperValue, &randomNumbers_f8);
    salt_and_pepper_noise_8_hip_compute(&src_f24.f8[1], &dst_f24.f8[1], noiseProbability, saltProbability, saltValue, pepperValue, &randomNumbers_f8);
    salt_and_pepper_noise_8_hip_compute(&src_f24.f8[2], &dst_f24.f8[2], noiseProbability, saltProbability, saltValue, pepperValue, &randomNumbers_f8);
    rpp_hip_pack_float24_pln3_and_store24_pln3(dstPtr + dstIdx, dstStridesNCH.y, &dst_f24);
}

template <typename T>
__global__ void salt_and_pepper_noise_pln3_pkd3_tensor(T *srcPtr,
                                                       uint3 srcStridesNCH,
                                                       T *dstPtr,
                                                       uint2 dstStridesNH,
                                                       float *noiseProbabilityTensor,
                                                       float *saltProbabilityTensor,
                                                       float *saltValueTensor,
                                                       float *pepperValueTensor,
                                                       RpptXorwowState *xorwowInitialStatePtr,
                                                       uint *xorwowSeedStream,
                                                       RpptROIPtr roiTensorPtrSrc)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if ((id_y >= roiTensorPtrSrc[id_z].xywhROI.roiHeight) || (id_x >= roiTensorPtrSrc[id_z].xywhROI.roiWidth))
    {
        return;
    }

    uint srcIdx = (id_z * srcStridesNCH.x) + ((id_y + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNCH.z) + (id_x + roiTensorPtrSrc[id_z].xywhROI.xy.x);
    uint dstIdx = (id_z * dstStridesNH.x) + (id_y * dstStridesNH.y) + id_x * 3;
    uint seedStreamIdx = (id_y * dstStridesNH.y) + (hipBlockIdx_x * hipBlockDim_x) + hipThreadIdx_x;

    float noiseProbability = noiseProbabilityTensor[id_z];
    float saltProbability = saltProbabilityTensor[id_z] * noiseProbability;
    float saltValue = saltValueTensor[id_z];
    float pepperValue = pepperValueTensor[id_z];

    RpptXorwowState xorwowState;
    uint xorwowSeed = (seedStreamIdx >= SEED_STREAM_MAX_SIZE) ? xorwowSeedStream[seedStreamIdx - SEED_STREAM_MAX_SIZE] : xorwowSeedStream[seedStreamIdx];
    xorwowState.x[0] = xorwowInitialStatePtr->x[0] + xorwowSeed;
    xorwowState.x[1] = xorwowInitialStatePtr->x[1] + xorwowSeed;
    xorwowState.x[2] = xorwowInitialStatePtr->x[2] + xorwowSeed;
    xorwowState.x[3] = xorwowInitialStatePtr->x[3] + xorwowSeed;
    xorwowState.x[4] = xorwowInitialStatePtr->x[4] + xorwowSeed;
    xorwowState.counter = xorwowInitialStatePtr->counter + xorwowSeed;

    d_float8 randomNumbers_f8;
    d_float24 src_f24, dst_f24;

    rpp_hip_rng_8_xorwow_f32(&xorwowState, &randomNumbers_f8);
    salt_and_pepper_noise_adjusted_input_hip_compute(srcPtr, &saltValue, &pepperValue);
    rpp_hip_load24_pln3_and_unpack_to_float24_pln3(srcPtr + srcIdx, srcStridesNCH.y, &src_f24);
    salt_and_pepper_noise_8_hip_compute(&src_f24.f8[0], &dst_f24.f8[0], noiseProbability, saltProbability, saltValue, pepperValue, &randomNumbers_f8);
    salt_and_pepper_noise_8_hip_compute(&src_f24.f8[1], &dst_f24.f8[1], noiseProbability, saltProbability, saltValue, pepperValue, &randomNumbers_f8);
    salt_and_pepper_noise_8_hip_compute(&src_f24.f8[2], &dst_f24.f8[2], noiseProbability, saltProbability, saltValue, pepperValue, &randomNumbers_f8);
    rpp_hip_pack_float24_pln3_and_store24_pkd3(dstPtr + dstIdx, &dst_f24);
}

template <typename T>
RppStatus hip_exec_salt_and_pepper_noise_tensor(T *srcPtr,
                                                RpptDescPtr srcDescPtr,
                                                T *dstPtr,
                                                RpptDescPtr dstDescPtr,
                                                RpptXorwowState *xorwowInitialStatePtr,
                                                RpptROIPtr roiTensorPtrSrc,
                                                RpptRoiType roiType,
                                                rpp::Handle& handle)
{
    if (roiType == RpptRoiType::LTRB)
        hip_exec_roi_converison_ltrb_to_xywh(roiTensorPtrSrc, handle);

    int localThreads_x = LOCAL_THREADS_X;
    int localThreads_y = LOCAL_THREADS_Y;
    int localThreads_z = LOCAL_THREADS_Z;
    int globalThreads_x = (dstDescPtr->strides.hStride + 7) >> 3;
    int globalThreads_y = dstDescPtr->h;
    int globalThreads_z = handle.GetBatchSize();

    Rpp32u *xorwowSeedStream;
    xorwowSeedStream = (Rpp32u *)&xorwowInitialStatePtr[1];
    hipMemcpy(xorwowSeedStream, rngSeedStream1036800, SEED_STREAM_MAX_SIZE * sizeof(Rpp32u), hipMemcpyHostToDevice);

    if ((srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NHWC))
    {
        globalThreads_x = (dstDescPtr->strides.hStride / 3 + 7) >> 3;
        hipLaunchKernelGGL(salt_and_pepper_noise_pkd_tensor,
                           dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
                           dim3(localThreads_x, localThreads_y, localThreads_z),
                           0,
                           handle.GetStream(),
                           srcPtr,
                           make_uint2(srcDescPtr->strides.nStride, srcDescPtr->strides.hStride),
                           dstPtr,
                           make_uint2(dstDescPtr->strides.nStride, dstDescPtr->strides.hStride),
                           handle.GetInitHandle()->mem.mgpu.floatArr[0].floatmem,
                           handle.GetInitHandle()->mem.mgpu.floatArr[1].floatmem,
                           handle.GetInitHandle()->mem.mgpu.floatArr[2].floatmem,
                           handle.GetInitHandle()->mem.mgpu.floatArr[3].floatmem,
                           xorwowInitialStatePtr,
                           xorwowSeedStream,
                           roiTensorPtrSrc);
    }
    else if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
    {
        hipLaunchKernelGGL(salt_and_pepper_noise_pln_tensor,
                           dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
                           dim3(localThreads_x, localThreads_y, localThreads_z),
                           0,
                           handle.GetStream(),
                           srcPtr,
                           make_uint3(srcDescPtr->strides.nStride, srcDescPtr->strides.cStride, srcDescPtr->strides.hStride),
                           dstPtr,
                           make_uint3(dstDescPtr->strides.nStride, dstDescPtr->strides.cStride, dstDescPtr->strides.hStride),
                           dstDescPtr->c,
                           handle.GetInitHandle()->mem.mgpu.floatArr[0].floatmem,
                           handle.GetInitHandle()->mem.mgpu.floatArr[1].floatmem,
                           handle.GetInitHandle()->mem.mgpu.floatArr[2].floatmem,
                           handle.GetInitHandle()->mem.mgpu.floatArr[3].floatmem,
                           xorwowInitialStatePtr,
                           xorwowSeedStream,
                           roiTensorPtrSrc);
    }
    else if ((srcDescPtr->c == 3) && (dstDescPtr->c == 3))
    {
        if ((srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            hipLaunchKernelGGL(salt_and_pepper_noise_pkd3_pln3_tensor,
                               dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
                               dim3(localThreads_x, localThreads_y, localThreads_z),
                               0,
                               handle.GetStream(),
                               srcPtr,
                               make_uint2(srcDescPtr->strides.nStride, srcDescPtr->strides.hStride),
                               dstPtr,
                               make_uint3(dstDescPtr->strides.nStride, dstDescPtr->strides.cStride, dstDescPtr->strides.hStride),
                               handle.GetInitHandle()->mem.mgpu.floatArr[0].floatmem,
                               handle.GetInitHandle()->mem.mgpu.floatArr[1].floatmem,
                               handle.GetInitHandle()->mem.mgpu.floatArr[2].floatmem,
                               handle.GetInitHandle()->mem.mgpu.floatArr[3].floatmem,
                               xorwowInitialStatePtr,
                               xorwowSeedStream,
                               roiTensorPtrSrc);
        }
        else if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            globalThreads_x = (srcDescPtr->strides.hStride + 7) >> 3;
            hipLaunchKernelGGL(salt_and_pepper_noise_pln3_pkd3_tensor,
                               dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
                               dim3(localThreads_x, localThreads_y, localThreads_z),
                               0,
                               handle.GetStream(),
                               srcPtr,
                               make_uint3(srcDescPtr->strides.nStride, srcDescPtr->strides.cStride, srcDescPtr->strides.hStride),
                               dstPtr,
                               make_uint2(dstDescPtr->strides.nStride, dstDescPtr->strides.hStride),
                               handle.GetInitHandle()->mem.mgpu.floatArr[0].floatmem,
                               handle.GetInitHandle()->mem.mgpu.floatArr[1].floatmem,
                               handle.GetInitHandle()->mem.mgpu.floatArr[2].floatmem,
                               handle.GetInitHandle()->mem.mgpu.floatArr[3].floatmem,
                               xorwowInitialStatePtr,
                               xorwowSeedStream,
                               roiTensorPtrSrc);
        }
    }

    return RPP_SUCCESS;
}
