/*
Copyright (c) 2019 - 2022 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include "rppdefs.h"
#include "rppi_validate.hpp"
#include "rppt_tensor_statistical_operations.h"
// #include "cpu/host_tensor_statistical_operations.hpp"

#ifdef HIP_COMPILE
    #include <hip/hip_fp16.h>
    #include "hip/hip_tensor_statistical_operations.hpp"
#endif // HIP_COMPILE

/******************** cartesian_to_polar ********************/

RppStatus rppt_cartesian_to_polar_host(RppPtr_t srcPtr,
                                       RpptGenericDescPtr srcGenericDescPtr,
                                       RppPtr_t dstPtr,
                                       RpptGenericDescPtr dstGenericDescPtr,
                                       RpptAngleType angleType,
                                       RpptROIPtr roiTensorPtrSrc,
                                       RpptRoiType roiType,
                                       rppHandle_t rppHandle)
{
    if (srcGenericDescPtr->dataType != RpptDataType::F32)   // src tensor data type is F32 for cartesian coordinates
        return RPP_ERROR_INVALID_SRC_DATA_TYPE;
    if (dstGenericDescPtr->dataType != RpptDataType::F32)   // dst tensor data type is F32 for polar coordinates
        return RPP_ERROR_INVALID_DST_DATA_TYPE;

    RpptDesc srcDesc, dstDesc;
    RpptDescPtr srcDescPtr, dstDescPtr;
    rpp_tensor_generic_to_image_desc(srcGenericDescPtr, srcDescPtr);
    rpp_tensor_generic_to_image_desc(dstGenericDescPtr, dstDescPtr);

    if (srcDescPtr->c != 2)
        return RPP_ERROR_INVALID_SRC_CHANNELS;  // src tensor channels is 2 for (x, y) coordinates
    if (dstDescPtr->c != 2)
        return RPP_ERROR_INVALID_DST_CHANNELS;  // dst tensor channels is 2 for (magnitude, angle) coordinates

    // cartesian_to_polar_f32_f32_host_tensor((Rpp32f*) (static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
    //                                        srcDescPtr,
    //                                        (Rpp32f*) (static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
    //                                        dstDescPtr,
    //                                        angleType,
    //                                        roiTensorPtrSrc,
    //                                        roiType);

    return RPP_SUCCESS;
}

/********************************************************************************************************************/
/*********************************************** RPP_GPU_SUPPORT = ON ***********************************************/
/********************************************************************************************************************/

#ifdef GPU_SUPPORT

/******************** cartesian_to_polar ********************/

RppStatus rppt_cartesian_to_polar_gpu(RppPtr_t srcPtr,
                                      RpptGenericDescPtr srcGenericDescPtr,
                                      RppPtr_t dstPtr,
                                      RpptGenericDescPtr dstGenericDescPtr,
                                      RpptAngleType angleType,
                                      RpptROIPtr roiTensorPtrSrc,
                                      RpptRoiType roiType,
                                      rppHandle_t rppHandle)
{
#ifdef HIP_COMPILE
    if (srcGenericDescPtr->dataType != RpptDataType::F32)   // src tensor data type is F32 for cartesian coordinates
        return RPP_ERROR_INVALID_SRC_DATA_TYPE;
    if (dstGenericDescPtr->dataType != RpptDataType::F32)   // dst tensor data type is F32 for polar coordinates
        return RPP_ERROR_INVALID_DST_DATA_TYPE;

    RpptDesc srcDesc, dstDesc;
    RpptDescPtr srcDescPtr, dstDescPtr;
    srcDescPtr = &srcDesc;
    dstDescPtr = &dstDesc;

    rpp_tensor_generic_to_image_desc(srcGenericDescPtr, srcDescPtr);
    rpp_tensor_generic_to_image_desc(dstGenericDescPtr, dstDescPtr);

    if (srcDescPtr->c != 2)
        return RPP_ERROR_INVALID_SRC_CHANNELS;  // src tensor channels is 2 for (x, y) coordinates
    if (dstDescPtr->c != 2)
        return RPP_ERROR_INVALID_DST_CHANNELS;  // dst tensor channels is 2 for (magnitude, angle) coordinates

    hip_exec_cartesian_to_polar_tensor((Rpp32f*) (static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                       srcDescPtr,
                                       (Rpp32f*) (static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                       dstDescPtr,
                                       angleType,
                                       roiTensorPtrSrc,
                                       roiType,
                                       rpp::deref(rppHandle));

    return RPP_SUCCESS;
#elif defined(OCL_COMPILE)
    return RPP_ERROR_NOT_IMPLEMENTED;
#endif // backend
}

#endif // GPU_SUPPORT
