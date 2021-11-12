#include <stdio.h>
#include <dirent.h>
#include <string.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include "/opt/rocm/rpp/include/rpp.h"
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <time.h>
#include <omp.h>
#include <half.hpp>
#include <fstream>
#include "helpers/testSuite_helper.hpp"

using namespace cv;
using namespace std;
using half_float::half;

typedef half Rpp16f;

#define RPPPIXELCHECK(pixel) (pixel < (Rpp32f)0) ? ((Rpp32f)0) : ((pixel < (Rpp32f)255) ? pixel : ((Rpp32f)255))
#define RPPMAX2(a,b) ((a > b) ? a : b)
#define RPPMIN2(a,b) ((a < b) ? a : b)

int main(int argc, char **argv)
{
    // Handle inputs

    const int MIN_ARG_COUNT = 7;

    if (argc < MIN_ARG_COUNT)
    {
        printf("\nImproper Usage! Needs all arguments!\n");
        printf("\nUsage: ./Tensor_host_pln1 <src1 folder> <src2 folder (place same as src1 folder for single image functionalities)> <u8 = 0 / f16 = 1 / f32 = 2 / u8->f16 = 3 / u8->f32 = 4 / i8 = 5 / u8->i8 = 6> <outputFormatToggle (pkd->pkd = 0 / pkd->pln = 1)> <case number = 0:81> <verbosity = 0/1>\n");
        return -1;
    }
    if (atoi(argv[4]) != 0)
    {
        printf("\nPLN1 cases don't have outputFormatToggle! Please input outputFormatToggle = 0\n");
        return -1;
    }

    char *src = argv[1];
    char *src_second = argv[2];
    int ip_bitDepth = atoi(argv[3]);
    unsigned int outputFormatToggle = atoi(argv[4]);
    int test_case = atoi(argv[5]);
    unsigned int verbosity = (test_case == 49) ? atoi(argv[7]) : atoi(argv[6]);
    unsigned int additionalParam = (test_case == 49) ? atoi(argv[6]) : 1;
    char additionalParam_char[2];
    std::sprintf(additionalParam_char, "%u", additionalParam);

    if (verbosity == 1)
    {
        printf("\nInputs for this test case are:");
        printf("\nsrc1 = %s", argv[1]);
        printf("\nsrc2 = %s", argv[2]);
        printf("\nu8 / f16 / f32 / u8->f16 / u8->f32 / i8 / u8->i8 (0/1/2/3/4/5/6) = %s", argv[3]);
        printf("\noutputFormatToggle (pkd->pkd = 0 / pkd->pln = 1) = %s", argv[4]);
        printf("\ncase number (0:81) = %s", argv[5]);
    }

    int ip_channel = 1;

    // Set case names

    char funcType[1000] = {"Tensor_HOST_PLN1_toPLN1"};

    char funcName[1000];
    switch (test_case)
    {
    case 0:
        strcpy(funcName, "brightness");
        break;
    case 1:
        strcpy(funcName, "gamma_correction");
        break;
    case 2:
        strcpy(funcName, "blend");
        break;
    case 31:
        strcpy(funcName, "color_cast");
        break;
    case 49:
        strcpy(funcName, "box_filter");
        break;
    case 81:
        strcpy(funcName, "color_jitter");
        break;
    default:
        strcpy(funcName, "test_case");
        break;
    }

    // Initialize tensor descriptors

    RpptDesc srcDesc, dstDesc;
    RpptDescPtr srcDescPtr, dstDescPtr;
    srcDescPtr = &srcDesc;
    dstDescPtr = &dstDesc;

    // Set src/dst layouts in tensor descriptors

    srcDescPtr->layout = RpptLayout::NCHW;
    dstDescPtr->layout = RpptLayout::NCHW;

    // Set src/dst data types in tensor descriptors

    if (ip_bitDepth == 0)
    {
        strcat(funcName, "_u8_");
        srcDescPtr->dataType = RpptDataType::U8;
        dstDescPtr->dataType = RpptDataType::U8;
    }
    else if (ip_bitDepth == 1)
    {
        strcat(funcName, "_f16_");
        srcDescPtr->dataType = RpptDataType::F16;
        dstDescPtr->dataType = RpptDataType::F16;
    }
    else if (ip_bitDepth == 2)
    {
        strcat(funcName, "_f32_");
        srcDescPtr->dataType = RpptDataType::F32;
        dstDescPtr->dataType = RpptDataType::F32;
    }
    else if (ip_bitDepth == 3)
    {
        strcat(funcName, "_u8_f16_");
        srcDescPtr->dataType = RpptDataType::U8;
        dstDescPtr->dataType = RpptDataType::F16;
    }
    else if (ip_bitDepth == 4)
    {
        strcat(funcName, "_u8_f32_");
        srcDescPtr->dataType = RpptDataType::U8;
        dstDescPtr->dataType = RpptDataType::F32;
    }
    else if (ip_bitDepth == 5)
    {
        strcat(funcName, "_i8_");
        srcDescPtr->dataType = RpptDataType::I8;
        dstDescPtr->dataType = RpptDataType::I8;
    }
    else if (ip_bitDepth == 6)
    {
        strcat(funcName, "_u8_i8_");
        srcDescPtr->dataType = RpptDataType::U8;
        dstDescPtr->dataType = RpptDataType::I8;
    }

    // Other initializations

    int missingFuncFlag = 0;
    int i = 0, j = 0;
    int maxHeight = 0, maxWidth = 0;
    int maxDstHeight = 0, maxDstWidth = 0;
    unsigned long long count = 0;
    unsigned long long ioBufferSize = 0;
    unsigned long long oBufferSize = 0;
    static int noOfImages = 0;
    Mat image, image_second;

    // String ops on function name

    char src1[1000];
    strcpy(src1, src);
    strcat(src1, "/");
    char src1_second[1000];
    strcpy(src1_second, src_second);
    strcat(src1_second, "/");

    char func[1000];
    strcpy(func, funcName);
    strcat(func, funcType);
    if (test_case == 49)
    {
        strcat(func, "_kSize");
        strcat(func, additionalParam_char);
    }

    // Get number of images

    struct dirent *de;
    DIR *dr = opendir(src);
    while ((de = readdir(dr)) != NULL)
    {
        if (strcmp(de->d_name, ".") == 0 || strcmp(de->d_name, "..") == 0)
            continue;
        noOfImages += 1;
    }
    closedir(dr);

    // Initialize ROI tensors for src/dst

    RpptROI *roiTensorPtrSrc = (RpptROI *) calloc(noOfImages, sizeof(RpptROI));
    RpptROI *roiTensorPtrDst = (RpptROI *) calloc(noOfImages, sizeof(RpptROI));

    // Set ROI tensors types for src/dst

    RpptRoiType roiTypeSrc, roiTypeDst;
    roiTypeSrc = RpptRoiType::XYWH;
    roiTypeDst = RpptRoiType::XYWH;

    // Set maxHeight, maxWidth and ROIs for src/dst

    const int images = noOfImages;
    char imageNames[images][1000];

    DIR *dr1 = opendir(src);
    while ((de = readdir(dr1)) != NULL)
    {
        if (strcmp(de->d_name, ".") == 0 || strcmp(de->d_name, "..") == 0)
            continue;
        strcpy(imageNames[count], de->d_name);
        char temp[1000];
        strcpy(temp, src1);
        strcat(temp, imageNames[count]);

        image = imread(temp, 0);

        roiTensorPtrSrc[count].xywhROI.xy.x = 0;
        roiTensorPtrSrc[count].xywhROI.xy.y = 0;
        roiTensorPtrSrc[count].xywhROI.roiWidth = image.cols;
        roiTensorPtrSrc[count].xywhROI.roiHeight = image.rows;

        roiTensorPtrDst[count].xywhROI.xy.x = 0;
        roiTensorPtrDst[count].xywhROI.xy.y = 0;
        roiTensorPtrDst[count].xywhROI.roiWidth = image.cols;
        roiTensorPtrDst[count].xywhROI.roiHeight = image.rows;

        maxHeight = RPPMAX2(maxHeight, roiTensorPtrSrc[count].xywhROI.roiHeight);
        maxWidth = RPPMAX2(maxWidth, roiTensorPtrSrc[count].xywhROI.roiWidth);
        maxDstHeight = RPPMAX2(maxDstHeight, roiTensorPtrDst[count].xywhROI.roiHeight);
        maxDstWidth = RPPMAX2(maxDstWidth, roiTensorPtrDst[count].xywhROI.roiWidth);

        count++;
    }
    closedir(dr1);

    // Set numDims, n/c/h/w values for src/dst

    srcDescPtr->numDims = 4;
    dstDescPtr->numDims = 4;

    srcDescPtr->n = noOfImages;
    srcDescPtr->c = ip_channel;
    srcDescPtr->h = maxHeight;
    srcDescPtr->w = maxWidth;

    dstDescPtr->n = noOfImages;
    dstDescPtr->c = ip_channel;
    dstDescPtr->h = maxDstHeight;
    dstDescPtr->w = maxDstWidth;

    // Optionally set w stride as a multiple of 8 for src/dst

    srcDescPtr->w = ((srcDescPtr->w / 8) * 8) + 8;
    dstDescPtr->w = ((dstDescPtr->w / 8) * 8) + 8;

    // Set offsetInBytes for src/dst

    srcDescPtr->offsetInBytes = (test_case == 49) ? ((additionalParam / 2) * (srcDescPtr->w + 1) * 12) : 0;
    dstDescPtr->offsetInBytes = 0;

    // Set n/c/h/w strides for src/dst

    srcDescPtr->strides.nStride = ip_channel * srcDescPtr->w * srcDescPtr->h;
    srcDescPtr->strides.cStride = srcDescPtr->w * srcDescPtr->h;
    srcDescPtr->strides.hStride = srcDescPtr->w;
    srcDescPtr->strides.wStride = 1;

    if (dstDescPtr->layout == RpptLayout::NHWC)
    {
        dstDescPtr->strides.nStride = ip_channel * dstDescPtr->w * dstDescPtr->h;
        dstDescPtr->strides.hStride = ip_channel * dstDescPtr->w;
        dstDescPtr->strides.wStride = ip_channel;
        dstDescPtr->strides.cStride = 1;
    }
    else if (dstDescPtr->layout == RpptLayout::NCHW)
    {
        dstDescPtr->strides.nStride = ip_channel * dstDescPtr->w * dstDescPtr->h;
        dstDescPtr->strides.cStride = dstDescPtr->w * dstDescPtr->h;
        dstDescPtr->strides.hStride = dstDescPtr->w;
        dstDescPtr->strides.wStride = 1;
    }

    // Set buffer sizes in pixels for src/dst

    ioBufferSize = (unsigned long long)srcDescPtr->h * (unsigned long long)srcDescPtr->w * (unsigned long long)ip_channel * (unsigned long long)noOfImages;
    oBufferSize = (unsigned long long)dstDescPtr->h * (unsigned long long)dstDescPtr->w * (unsigned long long)ip_channel * (unsigned long long)noOfImages;

    // Set buffer sizes in bytes for src/dst (including offsets)

    unsigned long long ioBufferSizeInBytes_u8 = ioBufferSize + (2 * srcDescPtr->offsetInBytes);
    unsigned long long oBufferSizeInBytes_u8 = oBufferSize + (2 * dstDescPtr->offsetInBytes);
    unsigned long long ioBufferSizeInBytes_f16 = (ioBufferSize * 2) + (2 * srcDescPtr->offsetInBytes);
    unsigned long long oBufferSizeInBytes_f16 = (oBufferSize * 2) + (2 * dstDescPtr->offsetInBytes);
    unsigned long long ioBufferSizeInBytes_f32 = (ioBufferSize * 4) + (2 * srcDescPtr->offsetInBytes);
    unsigned long long oBufferSizeInBytes_f32 = (oBufferSize * 4) + (2 * dstDescPtr->offsetInBytes);
    unsigned long long ioBufferSizeInBytes_i8 = ioBufferSize + (2 * srcDescPtr->offsetInBytes);
    unsigned long long oBufferSizeInBytes_i8 = oBufferSize + (2 * dstDescPtr->offsetInBytes);

    // Initialize 8u host buffers for src/dst

    Rpp8u *input = (Rpp8u *)calloc(ioBufferSizeInBytes_u8, 1);
    Rpp8u *input_second = (Rpp8u *)calloc(ioBufferSizeInBytes_u8, 1);
    Rpp8u *output = (Rpp8u *)calloc(oBufferSizeInBytes_u8, 1);

    // Set 8u host buffers for src/dst

    DIR *dr2 = opendir(src);
    DIR *dr2_second = opendir(src_second);
    count = 0;
    i = 0;

    Rpp8u *offsetted_input, *offsetted_input_second;
    offsetted_input = input + srcDescPtr->offsetInBytes;
    offsetted_input_second = input_second + srcDescPtr->offsetInBytes;

    Rpp32u elementsInRowMax = srcDescPtr->w * ip_channel;

    while ((de = readdir(dr2)) != NULL)
    {
        Rpp8u *input_temp, *input_second_temp;
        input_temp = offsetted_input + (i * srcDescPtr->strides.nStride);
        input_second_temp = offsetted_input_second + (i * srcDescPtr->strides.nStride);

        if (strcmp(de->d_name, ".") == 0 || strcmp(de->d_name, "..") == 0)
            continue;

        char temp[1000];
        strcpy(temp, src1);
        strcat(temp, de->d_name);

        char temp_second[1000];
        strcpy(temp_second, src1_second);
        strcat(temp_second, de->d_name);

        image = imread(temp, 0);
        image_second = imread(temp_second, 0);

        Rpp8u *ip_image = image.data;
        Rpp8u *ip_image_second = image_second.data;

        Rpp32u elementsInRow = roiTensorPtrSrc[i].xywhROI.roiWidth * ip_channel;

        for (j = 0; j < roiTensorPtrSrc[i].xywhROI.roiHeight; j++)
        {
            memcpy(input_temp, ip_image, elementsInRow * sizeof (Rpp8u));
            memcpy(input_second_temp, ip_image_second, elementsInRow * sizeof (Rpp8u));
            ip_image += elementsInRow;
            ip_image_second += elementsInRow;
            input_temp += elementsInRowMax;
            input_second_temp += elementsInRowMax;
        }
        i++;
        count += srcDescPtr->strides.nStride;
    }
    closedir(dr2);

    // Convert inputs to test various other bit depths

    Rpp16f *inputf16, *inputf16_second, *outputf16;
    Rpp32f *inputf32, *inputf32_second, *outputf32;
    Rpp8s *inputi8, *inputi8_second, *outputi8;

    if (ip_bitDepth == 1)
    {
        inputf16 = (Rpp16f *)calloc(ioBufferSizeInBytes_f16, 1);
        inputf16_second = (Rpp16f *)calloc(ioBufferSizeInBytes_f16, 1);
        outputf16 = (Rpp16f *)calloc(oBufferSizeInBytes_f16, 1);

        Rpp8u *inputTemp, *input_secondTemp;
        Rpp16f *inputf16Temp, *inputf16_secondTemp;

        inputTemp = input + srcDescPtr->offsetInBytes;
        input_secondTemp = input_second + srcDescPtr->offsetInBytes;

        inputf16Temp = (Rpp16f *)((Rpp8u *)inputf16 + srcDescPtr->offsetInBytes);
        inputf16_secondTemp = (Rpp16f *)((Rpp8u *)inputf16_second + srcDescPtr->offsetInBytes);

        for (int i = 0; i < ioBufferSize; i++)
        {
            *inputf16Temp = ((Rpp16f)*inputTemp) / 255.0;
            *inputf16_secondTemp = ((Rpp16f)*input_secondTemp) / 255.0;
            inputTemp++;
            inputf16Temp++;
            input_secondTemp++;
            inputf16_secondTemp++;
        }
    }
    else if (ip_bitDepth == 2)
    {
        inputf32 = (Rpp32f *)calloc(ioBufferSizeInBytes_f32, 1);
        inputf32_second = (Rpp32f *)calloc(ioBufferSizeInBytes_f32, 1);
        outputf32 = (Rpp32f *)calloc(oBufferSizeInBytes_f32, 1);

        Rpp8u *inputTemp, *input_secondTemp;
        Rpp32f *inputf32Temp, *inputf32_secondTemp;

        inputTemp = input + srcDescPtr->offsetInBytes;
        input_secondTemp = input_second + srcDescPtr->offsetInBytes;

        inputf32Temp = (Rpp32f *)((Rpp8u *)inputf32 + srcDescPtr->offsetInBytes);
        inputf32_secondTemp = (Rpp32f *)((Rpp8u *)inputf32_second + srcDescPtr->offsetInBytes);

        for (int i = 0; i < ioBufferSize; i++)
        {
            *inputf32Temp = ((Rpp32f)*inputTemp) / 255.0;
            *inputf32_secondTemp = ((Rpp32f)*input_secondTemp) / 255.0;
            inputTemp++;
            inputf32Temp++;
            input_secondTemp++;
            inputf32_secondTemp++;
        }
    }
    else if (ip_bitDepth == 5)
    {
        inputi8 = (Rpp8s *)calloc(ioBufferSizeInBytes_i8, 1);
        inputi8_second = (Rpp8s *)calloc(ioBufferSizeInBytes_i8, 1);
        outputi8 = (Rpp8s *)calloc(oBufferSizeInBytes_i8, 1);

        Rpp8u *inputTemp, *input_secondTemp;
        Rpp8s *inputi8Temp, *inputi8_secondTemp;

        inputTemp = input + srcDescPtr->offsetInBytes;
        input_secondTemp = input_second + srcDescPtr->offsetInBytes;

        inputi8Temp = inputi8 + srcDescPtr->offsetInBytes;
        inputi8_secondTemp = inputi8_second + srcDescPtr->offsetInBytes;

        for (int i = 0; i < ioBufferSize; i++)
        {
            *inputi8Temp = (Rpp8s) (((Rpp32s) *inputTemp) - 128);
            *inputi8_secondTemp = (Rpp8s) (((Rpp32s) *input_secondTemp) - 128);
            inputTemp++;
            inputi8Temp++;
            input_secondTemp++;
            inputi8_secondTemp++;
        }
    }

    // Run case-wise RPP API and measure time

    rppHandle_t handle;
    rppCreateWithBatchSize(&handle, noOfImages);

    double max_time_used = 0, min_time_used = 500, avg_time_used = 0;

    string test_case_name;

    printf("\nRunning %s 100 times (each time with a batch size of %d images) and computing mean statistics...", func, noOfImages);

    for (int perfRunCount = 0; perfRunCount < 100; perfRunCount++)
    {
        clock_t start, end;
        double start_omp, end_omp;
        double cpu_time_used, omp_time_used;
        switch (test_case)
        {
        case 0:
        {
            test_case_name = "brightness";

            Rpp32f alpha[images];
            Rpp32f beta[images];
            for (i = 0; i < images; i++)
            {
                alpha[i] = 1.75;
                beta[i] = 50;

                // xywhROI override sample
                // roiTensorPtrSrc[i].xywhROI.xy.x = 0;
                // roiTensorPtrSrc[i].xywhROI.xy.y = 0;
                // roiTensorPtrSrc[i].xywhROI.roiWidth = 100;
                // roiTensorPtrSrc[i].xywhROI.roiHeight = 180;

                // ltrbROI override sample
                // roiTensorPtrSrc[i].ltrbROI.lt.x = 50;
                // roiTensorPtrSrc[i].ltrbROI.lt.y = 50;
                // roiTensorPtrSrc[i].ltrbROI.rb.x = 199;
                // roiTensorPtrSrc[i].ltrbROI.rb.y = 149;
            }

            // Change RpptRoiType for ltrbROI override sample
            // roiTypeSrc = RpptRoiType::LTRB;
            // roiTypeDst = RpptRoiType::LTRB;

            start_omp = omp_get_wtime();
            start = clock();
            if (ip_bitDepth == 0)
                rppt_brightness_host(input, srcDescPtr, output, dstDescPtr, alpha, beta, roiTensorPtrSrc, roiTypeSrc, handle);
            else if (ip_bitDepth == 1)
                rppt_brightness_host(inputf16, srcDescPtr, outputf16, dstDescPtr, alpha, beta, roiTensorPtrSrc, roiTypeSrc, handle);
            else if (ip_bitDepth == 2)
                rppt_brightness_host(inputf32, srcDescPtr, outputf32, dstDescPtr, alpha, beta, roiTensorPtrSrc, roiTypeSrc, handle);
            else if (ip_bitDepth == 3)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 4)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 5)
                rppt_brightness_host(inputi8, srcDescPtr, outputi8, dstDescPtr, alpha, beta, roiTensorPtrSrc, roiTypeSrc, handle);
            else if (ip_bitDepth == 6)
                missingFuncFlag = 1;
            else
                missingFuncFlag = 1;
            end = clock();
            end_omp = omp_get_wtime();

            break;
        }
        case 1:
        {
            test_case_name = "gamma_correction";

            Rpp32f gammaVal[images];
            for (i = 0; i < images; i++)
            {
                gammaVal[i] = 1.9;

                // xywhROI override sample
                // roiTensorPtrSrc[i].xywhROI.xy.x = 0;
                // roiTensorPtrSrc[i].xywhROI.xy.y = 0;
                // roiTensorPtrSrc[i].xywhROI.roiWidth = 100;
                // roiTensorPtrSrc[i].xywhROI.roiHeight = 180;

                // ltrbROI override sample
                // roiTensorPtrSrc[i].ltrbROI.lt.x = 50;
                // roiTensorPtrSrc[i].ltrbROI.lt.y = 50;
                // roiTensorPtrSrc[i].ltrbROI.rb.x = 199;
                // roiTensorPtrSrc[i].ltrbROI.rb.y = 149;
            }

            // Change RpptRoiType for ltrbROI override sample
            // roiTypeSrc = RpptRoiType::LTRB;
            // roiTypeDst = RpptRoiType::LTRB;

            start_omp = omp_get_wtime();
            start = clock();
            if (ip_bitDepth == 0)
                rppt_gamma_correction_host(input, srcDescPtr, output, dstDescPtr, gammaVal, roiTensorPtrSrc, roiTypeSrc, handle);
            else if (ip_bitDepth == 1)
                rppt_gamma_correction_host(inputf16, srcDescPtr, outputf16, dstDescPtr, gammaVal, roiTensorPtrSrc, roiTypeSrc, handle);
            else if (ip_bitDepth == 2)
                rppt_gamma_correction_host(inputf32, srcDescPtr, outputf32, dstDescPtr, gammaVal, roiTensorPtrSrc, roiTypeSrc, handle);
            else if (ip_bitDepth == 3)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 4)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 5)
                rppt_gamma_correction_host(inputi8, srcDescPtr, outputi8, dstDescPtr, gammaVal, roiTensorPtrSrc, roiTypeSrc, handle);
            else if (ip_bitDepth == 6)
                missingFuncFlag = 1;
            else
                missingFuncFlag = 1;
            end = clock();
            end_omp = omp_get_wtime();

            break;
        }
        case 2:
        {
            test_case_name = "blend";

            Rpp32f alpha[images];
            for (i = 0; i < images; i++)
            {
                alpha[i] = 0.4;

                // xywhROI override sample
                // roiTensorPtrSrc[i].xywhROI.xy.x = 0;
                // roiTensorPtrSrc[i].xywhROI.xy.y = 0;
                // roiTensorPtrSrc[i].xywhROI.roiWidth = 100;
                // roiTensorPtrSrc[i].xywhROI.roiHeight = 180;

                // ltrbROI override sample
                // roiTensorPtrSrc[i].ltrbROI.lt.x = 50;
                // roiTensorPtrSrc[i].ltrbROI.lt.y = 50;
                // roiTensorPtrSrc[i].ltrbROI.rb.x = 199;
                // roiTensorPtrSrc[i].ltrbROI.rb.y = 149;
            }

            // Change RpptRoiType for ltrbROI override sample
            // roiTypeSrc = RpptRoiType::LTRB;
            // roiTypeDst = RpptRoiType::LTRB;

            start_omp = omp_get_wtime();
            start = clock();
            if (ip_bitDepth == 0)
                rppt_blend_host(input, input_second, srcDescPtr, output, dstDescPtr, alpha, roiTensorPtrSrc, roiTypeSrc, handle);
            else if (ip_bitDepth == 1)
                rppt_blend_host(inputf16, inputf16_second, srcDescPtr, outputf16, dstDescPtr, alpha, roiTensorPtrSrc, roiTypeSrc, handle);
            else if (ip_bitDepth == 2)
                rppt_blend_host(inputf32, inputf32_second, srcDescPtr, outputf32, dstDescPtr, alpha, roiTensorPtrSrc, roiTypeSrc, handle);
            else if (ip_bitDepth == 3)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 4)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 5)
                rppt_blend_host(inputi8, inputi8_second, srcDescPtr, outputi8, dstDescPtr, alpha, roiTensorPtrSrc, roiTypeSrc, handle);
            else if (ip_bitDepth == 6)
                missingFuncFlag = 1;
            else
                missingFuncFlag = 1;
            end = clock();
            end_omp = omp_get_wtime();

            break;
        }
        case 49:
        {
            test_case_name = "box_filter";

            Rpp32u kernelSize = additionalParam;
            for (i = 0; i < images; i++)
            {
                // xywhROI override sample
                // roiTensorPtrSrc[i].xywhROI.xy.x = 0;
                // roiTensorPtrSrc[i].xywhROI.xy.y = 0;
                // roiTensorPtrSrc[i].xywhROI.roiWidth = 100;
                // roiTensorPtrSrc[i].xywhROI.roiHeight = 180;

                // ltrbROI override sample
                // roiTensorPtrSrc[i].ltrbROI.lt.x = 50;
                // roiTensorPtrSrc[i].ltrbROI.lt.y = 50;
                // roiTensorPtrSrc[i].ltrbROI.rb.x = 199;
                // roiTensorPtrSrc[i].ltrbROI.rb.y = 149;
            }

            // Change RpptRoiType for ltrbROI override sample
            // roiTypeSrc = RpptRoiType::LTRB;
            // roiTypeDst = RpptRoiType::LTRB;

            start_omp = omp_get_wtime();
            start = clock();
            if (ip_bitDepth == 0)
                rppt_box_filter_host(input, srcDescPtr, ioBufferSizeInBytes_u8, output, dstDescPtr, kernelSize, roiTensorPtrSrc, roiTypeSrc, handle);
            else if (ip_bitDepth == 1)
                rppt_box_filter_host(inputf16, srcDescPtr, ioBufferSizeInBytes_f16, outputf16, dstDescPtr, kernelSize, roiTensorPtrSrc, roiTypeSrc, handle);
            else if (ip_bitDepth == 2)
                rppt_box_filter_host(inputf32, srcDescPtr, ioBufferSizeInBytes_f32, outputf32, dstDescPtr, kernelSize, roiTensorPtrSrc, roiTypeSrc, handle);
            else if (ip_bitDepth == 3)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 4)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 5)
                rppt_box_filter_host(inputi8, srcDescPtr, ioBufferSizeInBytes_i8, outputi8, dstDescPtr, kernelSize, roiTensorPtrSrc, roiTypeSrc, handle);
            else if (ip_bitDepth == 6)
                missingFuncFlag = 1;
            else
                missingFuncFlag = 1;
            end = clock();
            end_omp = omp_get_wtime();

            break;
        }
        default:
            missingFuncFlag = 1;
            break;
        }

        if (missingFuncFlag == 1)
        {
            printf("\nThe functionality %s doesn't yet exist in RPP\n", func);
            return -1;
        }

        cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
        omp_time_used = end_omp - start_omp;
        if (cpu_time_used > max_time_used)
            max_time_used = cpu_time_used;
        if (cpu_time_used < min_time_used)
            min_time_used = cpu_time_used;
        avg_time_used += cpu_time_used;
    }

    avg_time_used /= 100;

    // Display measured times

    cout << fixed << "\nmax,min,avg = " << max_time_used << "," << min_time_used << "," << avg_time_used << endl;

    rppDestroyHost(handle);

    // Free memory

    free(roiTensorPtrSrc);
    free(roiTensorPtrDst);
    free(input);
    free(input_second);
    free(output);

    if (ip_bitDepth == 1)
    {
        free(inputf16);
        free(inputf16_second);
        free(outputf16);
    }
    else if (ip_bitDepth == 2)
    {
        free(inputf32);
        free(inputf32_second);
        free(outputf32);
    }
    else if (ip_bitDepth == 5)
    {
        free(inputi8);
        free(inputi8_second);
        free(outputi8);
    }

    return 0;
}
