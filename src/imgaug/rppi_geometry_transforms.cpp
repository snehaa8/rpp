#include <rppi_geometry_transforms.h>
#include <rppdefs.h>
#include "rppi_validate.hpp"

#ifdef HIP_COMPILE
#include <hip/rpp_hip_common.hpp>

#elif defined(OCL_COMPILE)
#include <cl/rpp_cl_common.hpp>
#include "cl/cl_declarations.hpp"
#endif //backend
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <chrono>
using namespace std::chrono; 

#include "cpu/host_geometry_transforms.hpp" 
 
// ----------------------------------------
// Host flip functions calls 
// ----------------------------------------


RppStatus
rppi_flip_u8_pln1_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,RppiAxis flipAxis)
{

 	 validate_image_size(srcSize);
 	 validate_int_range( 0, 2, flipAxis);
	 flip_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr), 
			srcSize,
			static_cast<Rpp8u*>(dstPtr), 
			flipAxis,
			RPPI_CHN_PLANAR, 1);
	return RPP_SUCCESS;
}

RppStatus
rppi_flip_u8_pln3_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,RppiAxis flipAxis)
{

 	 validate_image_size(srcSize);
 	 validate_int_range( 0, 2, flipAxis);
	 flip_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr), 
			srcSize,
			static_cast<Rpp8u*>(dstPtr), 
			flipAxis,
			RPPI_CHN_PLANAR, 3);
	return RPP_SUCCESS;
}

RppStatus
rppi_flip_u8_pkd3_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,RppiAxis flipAxis)
{

 	 validate_image_size(srcSize);
 	 validate_int_range( 0, 2, flipAxis);
	 flip_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr), 
			srcSize,
			static_cast<Rpp8u*>(dstPtr), 
			flipAxis,
			RPPI_CHN_PACKED, 3);
	return RPP_SUCCESS;
}
 
// ----------------------------------------
// Host resize functions calls 
// ----------------------------------------


RppStatus
rppi_resize_u8_pln1_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,RppiSize dstSize)
{

 	 validate_image_size(srcSize);
 	 validate_image_size(dstSize);
	 resize_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr), 
			srcSize,
			static_cast<Rpp8u*>(dstPtr), 
			dstSize,
			RPPI_CHN_PLANAR, 1);
	return RPP_SUCCESS;
}

RppStatus
rppi_resize_u8_pln3_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,RppiSize dstSize)
{

 	 validate_image_size(srcSize);
 	 validate_image_size(dstSize);
	 resize_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr), 
			srcSize,
			static_cast<Rpp8u*>(dstPtr), 
			dstSize,
			RPPI_CHN_PLANAR, 3);
	return RPP_SUCCESS;
}

RppStatus
rppi_resize_u8_pkd3_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,RppiSize dstSize)
{

 	 validate_image_size(srcSize);
 	 validate_image_size(dstSize);
	 resize_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr), 
			srcSize,
			static_cast<Rpp8u*>(dstPtr), 
			dstSize,
			RPPI_CHN_PACKED, 3);
	return RPP_SUCCESS;
}
 
// ----------------------------------------
// Host resize_crop functions calls 
// ----------------------------------------


RppStatus
rppi_resize_crop_u8_pln1_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,RppiSize dstSize,Rpp32u x1,Rpp32u y1,Rpp32u x2,Rpp32u y2)
{

 	 validate_image_size(srcSize);
 	 validate_image_size(dstSize);
 	 validate_int_range( 0, srcSize.height, x1);
 	 validate_int_range( 0, srcSize.width, y1);
 	 validate_int_range( 0, srcSize.height, x2);
 	 validate_int_range( 0, srcSize.width, y2);
	 resize_crop_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr), 
			srcSize,
			static_cast<Rpp8u*>(dstPtr), 
			dstSize,
			x1,
			y1,
			x2,
			y2,
			RPPI_CHN_PLANAR, 1);
	return RPP_SUCCESS;
}

RppStatus
rppi_resize_crop_u8_pln3_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,RppiSize dstSize,Rpp32u x1,Rpp32u y1,Rpp32u x2,Rpp32u y2)
{

 	 validate_image_size(srcSize);
 	 validate_image_size(dstSize);
 	 validate_int_range( 0, srcSize.height, x1);
 	 validate_int_range( 0, srcSize.width, y1);
 	 validate_int_range( 0, srcSize.height, x2);
 	 validate_int_range( 0, srcSize.width, y2);
	 resize_crop_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr), 
			srcSize,
			static_cast<Rpp8u*>(dstPtr), 
			dstSize,
			x1,
			y1,
			x2,
			y2,
			RPPI_CHN_PLANAR, 3);
	return RPP_SUCCESS;
}

RppStatus
rppi_resize_crop_u8_pkd3_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,RppiSize dstSize,Rpp32u x1,Rpp32u y1,Rpp32u x2,Rpp32u y2)
{

 	 validate_image_size(srcSize);
 	 validate_image_size(dstSize);
 	 validate_int_range( 0, srcSize.height, x1);
 	 validate_int_range( 0, srcSize.width, y1);
 	 validate_int_range( 0, srcSize.height, x2);
 	 validate_int_range( 0, srcSize.width, y2);
	 resize_crop_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr), 
			srcSize,
			static_cast<Rpp8u*>(dstPtr), 
			dstSize,
			x1,
			y1,
			x2,
			y2,
			RPPI_CHN_PACKED, 3);
	return RPP_SUCCESS;
}
 
// ----------------------------------------
// Host rotate functions calls 
// ----------------------------------------


RppStatus
rppi_rotate_u8_pln1_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,RppiSize dstSize,Rpp32f angleDeg)
{

 	 validate_image_size(srcSize);
	 rotate_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr), 
			srcSize,
			static_cast<Rpp8u*>(dstPtr), 
			dstSize,
			angleDeg,
			RPPI_CHN_PLANAR, 1);
	return RPP_SUCCESS;
}

RppStatus
rppi_rotate_u8_pln3_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,RppiSize dstSize,Rpp32f angleDeg)
{

 	 validate_image_size(srcSize);
	 rotate_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr), 
			srcSize,
			static_cast<Rpp8u*>(dstPtr), 
			dstSize,
			angleDeg,
			RPPI_CHN_PLANAR, 3);
	return RPP_SUCCESS;
}

RppStatus
rppi_rotate_u8_pkd3_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,RppiSize dstSize,Rpp32f angleDeg)
{

 	 validate_image_size(srcSize);
	 rotate_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr), 
			srcSize,
			static_cast<Rpp8u*>(dstPtr), 
			dstSize,
			angleDeg,
			RPPI_CHN_PACKED, 3);
	return RPP_SUCCESS;
}
 
// ----------------------------------------
// GPU flip functions  calls 
// ----------------------------------------


RppStatus
rppi_flip_u8_pln1_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,RppiAxis flipAxis, RppHandle_t rppHandle) 
{

 	 validate_image_size(srcSize);
 	 validate_int_range( 0, 2, flipAxis);

#ifdef OCL_COMPILE
 	 {
 	 flip_cl(static_cast<cl_mem>(srcPtr), 
			srcSize,
			static_cast<cl_mem>(dstPtr), 
			flipAxis,
			RPPI_CHN_PLANAR, 1,
			static_cast<cl_command_queue>(rppHandle));
 	 } 
#elif defined (HIP_COMPILE) 
 	 { 
 	 } 
#endif //BACKEND 
		return RPP_SUCCESS;
}

RppStatus
rppi_flip_u8_pln3_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,RppiAxis flipAxis, RppHandle_t rppHandle) 
{

 	 validate_image_size(srcSize);
 	 validate_int_range( 0, 2, flipAxis);

#ifdef OCL_COMPILE
 	 {
 	 flip_cl(static_cast<cl_mem>(srcPtr), 
			srcSize,
			static_cast<cl_mem>(dstPtr), 
			flipAxis,
			RPPI_CHN_PLANAR, 3,
			static_cast<cl_command_queue>(rppHandle));
 	 } 
#elif defined (HIP_COMPILE) 
 	 { 
 	 } 
#endif //BACKEND 
		return RPP_SUCCESS;
}

RppStatus
rppi_flip_u8_pkd3_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,RppiAxis flipAxis, RppHandle_t rppHandle) 
{

 	 validate_image_size(srcSize);
 	 validate_int_range( 0, 2, flipAxis);

#ifdef OCL_COMPILE
 	 {
 	 flip_cl(static_cast<cl_mem>(srcPtr), 
			srcSize,
			static_cast<cl_mem>(dstPtr), 
			flipAxis,
			RPPI_CHN_PACKED, 3,
			static_cast<cl_command_queue>(rppHandle));
 	 } 
#elif defined (HIP_COMPILE) 
 	 { 
 	 } 
#endif //BACKEND 
		return RPP_SUCCESS;
}
 
// ----------------------------------------
// GPU resize functions  calls 
// ----------------------------------------


RppStatus
rppi_resize_u8_pln1_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,RppiSize dstSize, RppHandle_t rppHandle) 
{

 	 validate_image_size(srcSize);
 	 validate_image_size(dstSize);

#ifdef OCL_COMPILE
 	 {
 	 resize_cl(static_cast<cl_mem>(srcPtr), 
			srcSize,
			static_cast<cl_mem>(dstPtr), 
			dstSize,
			RPPI_CHN_PLANAR, 1,
			static_cast<cl_command_queue>(rppHandle));
 	 } 
#elif defined (HIP_COMPILE) 
 	 { 
 	 } 
#endif //BACKEND 
		return RPP_SUCCESS;
}

RppStatus
rppi_resize_u8_pln3_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,RppiSize dstSize, RppHandle_t rppHandle) 
{

 	 validate_image_size(srcSize);
 	 validate_image_size(dstSize);

#ifdef OCL_COMPILE
 	 {
 	 resize_cl(static_cast<cl_mem>(srcPtr), 
			srcSize,
			static_cast<cl_mem>(dstPtr), 
			dstSize,
			RPPI_CHN_PLANAR, 3,
			static_cast<cl_command_queue>(rppHandle));
 	 } 
#elif defined (HIP_COMPILE) 
 	 { 
 	 } 
#endif //BACKEND 
		return RPP_SUCCESS;
}

RppStatus
rppi_resize_u8_pkd3_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,RppiSize dstSize, RppHandle_t rppHandle) 
{

 	 validate_image_size(srcSize);
 	 validate_image_size(dstSize);

#ifdef OCL_COMPILE
 	 {
 	 resize_cl(static_cast<cl_mem>(srcPtr), 
			srcSize,
			static_cast<cl_mem>(dstPtr), 
			dstSize,
			RPPI_CHN_PACKED, 3,
			static_cast<cl_command_queue>(rppHandle));
 	 } 
#elif defined (HIP_COMPILE) 
 	 { 
 	 } 
#endif //BACKEND 
		return RPP_SUCCESS;
}
 
// ----------------------------------------
// GPU resize_crop functions  calls 
// ----------------------------------------


RppStatus
rppi_resize_crop_u8_pln1_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,RppiSize dstSize,Rpp32u x1,Rpp32u y1,Rpp32u x2,Rpp32u y2, RppHandle_t rppHandle) 
{

 	 validate_image_size(srcSize);
 	 validate_image_size(dstSize);
 	 validate_int_range( 0, srcSize.height, x1);
 	 validate_int_range( 0, srcSize.width, y1);
 	 validate_int_range( 0, srcSize.height, x2);
 	 validate_int_range( 0, srcSize.width, y2);

#ifdef OCL_COMPILE
 	 {
 	 resize_crop_cl(static_cast<cl_mem>(srcPtr), 
			srcSize,
			static_cast<cl_mem>(dstPtr), 
			dstSize,
			x1,
			y1,
			x2,
			y2,
			RPPI_CHN_PLANAR, 1,
			static_cast<cl_command_queue>(rppHandle));
 	 } 
#elif defined (HIP_COMPILE) 
 	 { 
 	 } 
#endif //BACKEND 
		return RPP_SUCCESS;
}

RppStatus
rppi_resize_crop_u8_pln3_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,RppiSize dstSize,Rpp32u x1,Rpp32u y1,Rpp32u x2,Rpp32u y2, RppHandle_t rppHandle) 
{

 	 validate_image_size(srcSize);
 	 validate_image_size(dstSize);
 	 validate_int_range( 0, srcSize.height, x1);
 	 validate_int_range( 0, srcSize.width, y1);
 	 validate_int_range( 0, srcSize.height, x2);
 	 validate_int_range( 0, srcSize.width, y2);

#ifdef OCL_COMPILE
 	 {
 	 resize_crop_cl(static_cast<cl_mem>(srcPtr), 
			srcSize,
			static_cast<cl_mem>(dstPtr), 
			dstSize,
			x1,
			y1,
			x2,
			y2,
			RPPI_CHN_PLANAR, 3,
			static_cast<cl_command_queue>(rppHandle));
 	 } 
#elif defined (HIP_COMPILE) 
 	 { 
 	 } 
#endif //BACKEND 
		return RPP_SUCCESS;
}

RppStatus
rppi_resize_crop_u8_pkd3_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,RppiSize dstSize,Rpp32u x1,Rpp32u y1,Rpp32u x2,Rpp32u y2, RppHandle_t rppHandle) 
{

 	 validate_image_size(srcSize);
 	 validate_image_size(dstSize);
 	 validate_int_range( 0, srcSize.height, x1);
 	 validate_int_range( 0, srcSize.width, y1);
 	 validate_int_range( 0, srcSize.height, x2);
 	 validate_int_range( 0, srcSize.width, y2);

#ifdef OCL_COMPILE
 	 {
 	 resize_crop_cl(static_cast<cl_mem>(srcPtr), 
			srcSize,
			static_cast<cl_mem>(dstPtr), 
			dstSize,
			x1,
			y1,
			x2,
			y2,
			RPPI_CHN_PACKED, 3,
			static_cast<cl_command_queue>(rppHandle));
 	 } 
#elif defined (HIP_COMPILE) 
 	 { 
 	 } 
#endif //BACKEND 
		return RPP_SUCCESS;
}
 
// ----------------------------------------
// GPU rotate functions  calls 
// ----------------------------------------


RppStatus
rppi_rotate_u8_pln1_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,RppiSize dstSize,Rpp32f angleDeg, RppHandle_t rppHandle) 
{

 	 validate_image_size(srcSize);

#ifdef OCL_COMPILE
 	 {
 	 rotate_cl(static_cast<cl_mem>(srcPtr), 
			srcSize,
			static_cast<cl_mem>(dstPtr), 
			dstSize,
			angleDeg,
			RPPI_CHN_PLANAR, 1,
			static_cast<cl_command_queue>(rppHandle));
 	 } 
#elif defined (HIP_COMPILE) 
 	 { 
 	 } 
#endif //BACKEND 
		return RPP_SUCCESS;
}

RppStatus
rppi_rotate_u8_pln3_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,RppiSize dstSize,Rpp32f angleDeg, RppHandle_t rppHandle) 
{

 	 validate_image_size(srcSize);

#ifdef OCL_COMPILE
 	 {
 	 rotate_cl(static_cast<cl_mem>(srcPtr), 
			srcSize,
			static_cast<cl_mem>(dstPtr), 
			dstSize,
			angleDeg,
			RPPI_CHN_PLANAR, 3,
			static_cast<cl_command_queue>(rppHandle));
 	 } 
#elif defined (HIP_COMPILE) 
 	 { 
 	 } 
#endif //BACKEND 
		return RPP_SUCCESS;
}

RppStatus
rppi_rotate_u8_pkd3_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,RppiSize dstSize,Rpp32f angleDeg, RppHandle_t rppHandle) 
{

 	 validate_image_size(srcSize);

#ifdef OCL_COMPILE
 	 {
 	 rotate_cl(static_cast<cl_mem>(srcPtr), 
			srcSize,
			static_cast<cl_mem>(dstPtr), 
			dstSize,
			angleDeg,
			RPPI_CHN_PACKED, 3,
			static_cast<cl_command_queue>(rppHandle));
 	 } 
#elif defined (HIP_COMPILE) 
 	 { 
 	 } 
#endif //BACKEND 
		return RPP_SUCCESS;
}

// ----------------------------------------
// GPU RandomCropLetterBox functions  calls 
// ----------------------------------------

RppStatus
rppi_Random_Crop_Letter_Box_u8_pln1_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,RppiSize dstSize,Rpp32u x1,Rpp32u y1,Rpp32u x2,Rpp32u y2, RppHandle_t rppHandle) 
{
 	 validate_image_size(srcSize);

#ifdef OCL_COMPILE
 	 {
 	 Random_Crop_Letter_Box_cl(static_cast<cl_mem>(srcPtr), 
			srcSize,
			static_cast<cl_mem>(dstPtr), 
			dstSize,
			x1,y1,x2,y2,
			RPPI_CHN_PLANAR, 1,
			static_cast<cl_command_queue>(rppHandle));
 	 } 
#elif defined (HIP_COMPILE) 
 	 { 
 	 } 
#endif //BACKEND 
		return RPP_SUCCESS;

}

RppStatus
rppi_Random_Crop_Letter_Box_u8_pln3_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,RppiSize dstSize,Rpp32u x1,Rpp32u y1,Rpp32u x2,Rpp32u y2, RppHandle_t rppHandle) 
{
 	 validate_image_size(srcSize);

#ifdef OCL_COMPILE
 	 {
 	 Random_Crop_Letter_Box_cl(static_cast<cl_mem>(srcPtr), 
			srcSize,
			static_cast<cl_mem>(dstPtr), 
			dstSize,
			x1,y1,x2,y2,
			RPPI_CHN_PLANAR, 3,
			static_cast<cl_command_queue>(rppHandle));
 	 } 
#elif defined (HIP_COMPILE) 
 	 { 
 	 } 
#endif //BACKEND 
		return RPP_SUCCESS;

}

RppStatus
rppi_Random_Crop_Letter_Box_u8_pkd3_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,RppiSize dstSize,Rpp32u x1,Rpp32u y1,Rpp32u x2,Rpp32u y2, RppHandle_t rppHandle) 
{
 	 validate_image_size(srcSize);

#ifdef OCL_COMPILE
 	 {
 	 Random_Crop_Letter_Box_cl(static_cast<cl_mem>(srcPtr), 
			srcSize,
			static_cast<cl_mem>(dstPtr), 
			dstSize,
			x1,y1,x2,y2,
			RPPI_CHN_PACKED, 3,
			static_cast<cl_command_queue>(rppHandle));
 	 } 
#elif defined (HIP_COMPILE) 
 	 { 
 	 } 
#endif //BACKEND 
		return RPP_SUCCESS;

}

RppStatus
rppi_warp_affine_u8_pln1_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize dstSize,
                         Rpp32f *affine, RppHandle_t rppHandle)
{
    

    /*call that offset function */
    #ifdef OCL_COMPILE

    warp_affine_cl(static_cast<cl_mem>(srcPtr), srcSize,
            static_cast<cl_mem>(dstPtr), dstSize, affine, RPPI_CHN_PLANAR, 1 /* Channel */,
            static_cast<cl_command_queue>(rppHandle) );

    return RPP_SUCCESS;
    #endif

}

RppStatus
rppi_warp_affine_u8_pln3_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize dstSize,
                         Rpp32f *affine, RppHandle_t rppHandle)
{

    /*call that offset function */
    #ifdef OCL_COMPILE

    warp_affine_cl(static_cast<cl_mem>(srcPtr), srcSize,
            static_cast<cl_mem>(dstPtr), dstSize, affine, RPPI_CHN_PLANAR, 3 /* Channel */,
            static_cast<cl_command_queue>(rppHandle) );

    return RPP_SUCCESS;
    #endif

}

RppStatus
rppi_warp_affine_u8_pkd3_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize dstSize,
                         Rpp32f *affine, RppHandle_t rppHandle)
{
    

    /*call that offset function */
    #ifdef OCL_COMPILE

    warp_affine_cl(static_cast<cl_mem>(srcPtr), srcSize,
            static_cast<cl_mem>(dstPtr), dstSize, affine, RPPI_CHN_PACKED, 3 /* Channel */,
            static_cast<cl_command_queue>(rppHandle) );

    return RPP_SUCCESS;
    #endif

}