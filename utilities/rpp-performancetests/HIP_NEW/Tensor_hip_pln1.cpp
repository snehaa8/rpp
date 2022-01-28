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
#include <hip/hip_fp16.h>
#include <fstream>
#include "helpers/testSuite_helper.hpp"
#include <random>
#include <boost/math/distributions.hpp>
#include <boost/math/special_functions/beta.hpp>
using namespace boost::math;

using namespace cv;
using namespace std;

#define RPPPIXELCHECK(pixel) (pixel < (Rpp32f)0) ? ((Rpp32f)0) : ((pixel < (Rpp32f)255) ? pixel : ((Rpp32f)255))
#define RPPMAX2(a,b) ((a > b) ? a : b)
#define RPPMIN2(a,b) ((a < b) ? a : b)

inline void randomize (unsigned int arr[], unsigned int n)
{
    // Use a different seed value each time
    srand (time(NULL));
    for (unsigned int i = n - 1; i > 0; i--)
    {
        // Pick a random index from 0 to i
        unsigned int j = rand() % (i + 1);
        std::swap(arr[i], arr[j]);
    }
}

int inline random_val(int min, int max)
{
    if(max<0)
        return -1;
    return rand() % (max - min + 1) + min;
}

int main(int argc, char **argv)
{
    // Handle inputs

    const int MIN_ARG_COUNT = 7;

    if (argc < MIN_ARG_COUNT)
    {
        printf("\nImproper Usage! Needs all arguments!\n");
        printf("\nUsage: ./Tensor_host_pln1 <src1 folder> <src2 folder (place same as src1 folder for single image functionalities)> <u8 = 0 / f16 = 1 / f32 = 2 / u8->f16 = 3 / u8->f32 = 4 / i8 = 5 / u8->i8 = 6> <outputFormatToggle (pkd->pkd = 0 / pkd->pln = 1)> <case number = 0:84> <verbosity = 0/1>\n");
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
    unsigned int verbosity = (test_case == 40 || test_case == 41 || test_case == 49) ? atoi(argv[7]) : atoi(argv[6]);
    unsigned int additionalParam = (test_case == 40 || test_case == 41 || test_case == 49) ? atoi(argv[6]) : 1;
    char additionalParam_char[2];
    std::sprintf(additionalParam_char, "%u", additionalParam);

    if (verbosity == 1)
    {
        printf("\nInputs for this test case are:");
        printf("\nsrc1 = %s", argv[1]);
        printf("\nsrc2 = %s", argv[2]);
        printf("\nu8 / f16 / f32 / u8->f16 / u8->f32 / i8 / u8->i8 (0/1/2/3/4/5/6) = %s", argv[3]);
        printf("\noutputFormatToggle (pkd->pkd = 0 / pkd->pln = 1) = %s", argv[4]);
        printf("\ncase number (0:84) = %s", argv[5]);
    }

    int ip_channel = 1;

    // Set case names

    char funcType[1000] = {"Tensor_HIP_PLN1_toPLN1"};

    char funcName[1000];
    switch (test_case)
    {
    case 0:
        strcpy(funcName, "brightness");
        outputFormatToggle = 0;
        break;
    case 1:
        strcpy(funcName, "gamma_correction");
        outputFormatToggle = 0;
        break;
    case 2:
        strcpy(funcName, "blend");
        outputFormatToggle = 0;
        break;
    case 31:
        strcpy(funcName, "color_cast");
        outputFormatToggle = 0;
        break;
    case 37:
        strcpy(funcName, "crop");
        break;
    case 40:
        strcpy(funcName, "erode");
        outputFormatToggle = 0;
        break;
    case 41:
        strcpy(funcName, "dilate");
        outputFormatToggle = 0;
        break;
    case 49:
        strcpy(funcName, "box_filter");
        outputFormatToggle = 0;
        break;
    case 82:
        strcpy(funcName, "ricap");
        break;
    case 83:
        strcpy(funcName, "gridmask");
        break;
    case 84:
        strcpy(funcName, "spatter");
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
    if (test_case == 40 || test_case == 41 || test_case == 49)
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

    RpptROI *d_roiTensorPtrSrc, *d_roiTensorPtrDst;
    hipMalloc(&d_roiTensorPtrSrc, noOfImages * sizeof(RpptROI));
    hipMalloc(&d_roiTensorPtrDst, noOfImages * sizeof(RpptROI));

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

    // Set numDims, offset, n/c/h/w values, n/c/h/w strides for src/dst

    srcDescPtr->numDims = 4;
    dstDescPtr->numDims = 4;

    srcDescPtr->offsetInBytes = 64;
    dstDescPtr->offsetInBytes = 0;

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

    unsigned long long ioBufferSizeInBytes_u8 = ioBufferSize + srcDescPtr->offsetInBytes;
    unsigned long long oBufferSizeInBytes_u8 = oBufferSize + dstDescPtr->offsetInBytes;
    unsigned long long ioBufferSizeInBytes_f16 = (ioBufferSize * 2) + srcDescPtr->offsetInBytes;
    unsigned long long oBufferSizeInBytes_f16 = (oBufferSize * 2) + dstDescPtr->offsetInBytes;
    unsigned long long ioBufferSizeInBytes_f32 = (ioBufferSize * 4) + srcDescPtr->offsetInBytes;
    unsigned long long oBufferSizeInBytes_f32 = (oBufferSize * 4) + dstDescPtr->offsetInBytes;
    unsigned long long ioBufferSizeInBytes_i8 = ioBufferSize + srcDescPtr->offsetInBytes;
    unsigned long long oBufferSizeInBytes_i8 = oBufferSize + dstDescPtr->offsetInBytes;

    // Initialize host buffers for src/dst

    Rpp8u *input = (Rpp8u *)calloc(ioBufferSize, sizeof(Rpp8u));
    Rpp8u *input_second = (Rpp8u *)calloc(ioBufferSize, sizeof(Rpp8u));
    Rpp8u *output = (Rpp8u *)calloc(oBufferSize, sizeof(Rpp8u));

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

    // Convert inputs to test various other bit depths and copy to hip buffers

    half *inputf16, *inputf16_second, *outputf16;
    Rpp32f *inputf32, *inputf32_second, *outputf32;
    Rpp8s *inputi8, *inputi8_second, *outputi8;
    int *d_input, *d_input_second, *d_inputf16, *d_inputf16_second, *d_inputf32, *d_inputf32_second, *d_inputi8, *d_inputi8_second;
    int *d_output, *d_outputf16, *d_outputf32, *d_outputi8;

    if (ip_bitDepth == 0)
    {
        hipMalloc(&d_input, ioBufferSizeInBytes_u8);
        hipMalloc(&d_input_second, ioBufferSizeInBytes_u8);
        hipMalloc(&d_output, oBufferSizeInBytes_u8);
        hipMemcpy(d_input, input, ioBufferSizeInBytes_u8, hipMemcpyHostToDevice);
        hipMemcpy(d_input_second, input_second, ioBufferSizeInBytes_u8, hipMemcpyHostToDevice);
        hipMemcpy(d_output, output, oBufferSizeInBytes_u8, hipMemcpyHostToDevice);
    }
    else if (ip_bitDepth == 1)
    {
        inputf16 = (half *)calloc(ioBufferSizeInBytes_f16, 1);
        inputf16_second = (half *)calloc(ioBufferSizeInBytes_f16, 1);
        outputf16 = (half *)calloc(oBufferSizeInBytes_f16, 1);

        Rpp8u *inputTemp, *input_secondTemp;
        half *inputf16Temp, *inputf16_secondTemp;

        inputTemp = input + srcDescPtr->offsetInBytes;
        input_secondTemp = input_second + srcDescPtr->offsetInBytes;

        inputf16Temp = (half *)((Rpp8u *)inputf16 + srcDescPtr->offsetInBytes);
        inputf16_secondTemp = (half *)((Rpp8u *)inputf16_second + srcDescPtr->offsetInBytes);

        for (int i = 0; i < ioBufferSize; i++)
        {
            *inputf16Temp = (half)(((float)*inputTemp) / 255.0);
            *inputf16_secondTemp = (half)(((float)*input_secondTemp) / 255.0);
            inputTemp++;
            inputf16Temp++;
            input_secondTemp++;
            inputf16_secondTemp++;
        }

        hipMalloc(&d_inputf16, ioBufferSizeInBytes_f16);
        hipMalloc(&d_inputf16_second, ioBufferSizeInBytes_f16);
        hipMalloc(&d_outputf16, oBufferSizeInBytes_f16);
        hipMemcpy(d_inputf16, inputf16, ioBufferSizeInBytes_f16, hipMemcpyHostToDevice);
        hipMemcpy(d_inputf16_second, inputf16_second, ioBufferSizeInBytes_f16, hipMemcpyHostToDevice);
        hipMemcpy(d_outputf16, outputf16, oBufferSizeInBytes_f16, hipMemcpyHostToDevice);
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

        hipMalloc(&d_inputf32, ioBufferSizeInBytes_f32);
        hipMalloc(&d_inputf32_second, ioBufferSizeInBytes_f32);
        hipMalloc(&d_outputf32, oBufferSizeInBytes_f32);
        hipMemcpy(d_inputf32, inputf32, ioBufferSizeInBytes_f32, hipMemcpyHostToDevice);
        hipMemcpy(d_inputf32_second, inputf32_second, ioBufferSizeInBytes_f32, hipMemcpyHostToDevice);
        hipMemcpy(d_outputf32, outputf32, oBufferSizeInBytes_f32, hipMemcpyHostToDevice);
    }
    else if (ip_bitDepth == 3)
    {
        outputf16 = (half *)calloc(oBufferSizeInBytes_f16, 1);
        hipMalloc(&d_input, ioBufferSizeInBytes_u8);
        hipMalloc(&d_input_second, ioBufferSizeInBytes_u8);
        hipMalloc(&d_outputf16, oBufferSizeInBytes_f16);
        hipMemcpy(d_input, input, ioBufferSizeInBytes_u8, hipMemcpyHostToDevice);
        hipMemcpy(d_input_second, input_second, ioBufferSizeInBytes_u8, hipMemcpyHostToDevice);
        hipMemcpy(d_outputf16, outputf16, oBufferSizeInBytes_f16, hipMemcpyHostToDevice);
    }
    else if (ip_bitDepth == 4)
    {
        outputf32 = (Rpp32f *)calloc(oBufferSizeInBytes_f32, 1);
        hipMalloc(&d_input, ioBufferSizeInBytes_u8);
        hipMalloc(&d_input_second, ioBufferSizeInBytes_u8);
        hipMalloc(&d_outputf32, oBufferSizeInBytes_f32);
        hipMemcpy(d_input, input, ioBufferSizeInBytes_u8, hipMemcpyHostToDevice);
        hipMemcpy(d_input_second, input_second, ioBufferSizeInBytes_u8, hipMemcpyHostToDevice);
        hipMemcpy(d_outputf32, outputf32, oBufferSizeInBytes_f32, hipMemcpyHostToDevice);
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

        hipMalloc(&d_inputi8, ioBufferSizeInBytes_i8);
        hipMalloc(&d_inputi8_second, ioBufferSizeInBytes_i8);
        hipMalloc(&d_outputi8, oBufferSizeInBytes_i8);
        hipMemcpy(d_inputi8, inputi8, ioBufferSizeInBytes_i8, hipMemcpyHostToDevice);
        hipMemcpy(d_inputi8_second, inputi8_second, ioBufferSizeInBytes_i8, hipMemcpyHostToDevice);
        hipMemcpy(d_outputi8, outputi8, oBufferSizeInBytes_i8, hipMemcpyHostToDevice);
    }
    else if (ip_bitDepth == 6)
    {
        outputi8 = (Rpp8s *)calloc(oBufferSizeInBytes_i8, 1);
        hipMalloc(&d_input, ioBufferSizeInBytes_u8);
        hipMalloc(&d_input_second, ioBufferSizeInBytes_u8);
        hipMalloc(&d_outputi8, oBufferSizeInBytes_i8);
        hipMemcpy(d_input, input, ioBufferSizeInBytes_u8, hipMemcpyHostToDevice);
        hipMemcpy(d_input_second, input_second, ioBufferSizeInBytes_u8, hipMemcpyHostToDevice);
        hipMemcpy(d_outputi8, outputi8, oBufferSizeInBytes_i8, hipMemcpyHostToDevice);
    }

    // Run case-wise RPP API and measure time

    rppHandle_t handle;
    hipStream_t stream;
    hipStreamCreate(&stream);
    rppCreateWithStreamAndBatchSize(&handle, stream, noOfImages);

    clock_t start, end;
    double max_time_used = 0, min_time_used = 500, avg_time_used = 0;

    string test_case_name;

    printf("\nRunning %s 100 times (each time with a batch size of %d images) and computing mean statistics...", func, noOfImages);

    for (int perfRunCount = 0; perfRunCount < 100; perfRunCount++)
    {
        double gpu_time_used;
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
            }

            // Uncomment to run test case with an xywhROI override
            /*for (i = 0; i < images; i++)
            {
                roiTensorPtrSrc[i].xywhROI.xy.x = 0;
                roiTensorPtrSrc[i].xywhROI.xy.y = 0;
                roiTensorPtrSrc[i].xywhROI.roiWidth = 100;
                roiTensorPtrSrc[i].xywhROI.roiHeight = 180;
            }*/

            // Uncomment to run test case with an ltrbROI override
            /*for (i = 0; i < images; i++)
                roiTensorPtrSrc[i].ltrbROI.lt.x = 50;
                roiTensorPtrSrc[i].ltrbROI.lt.y = 30;
                roiTensorPtrSrc[i].ltrbROI.rb.x = 210;
                roiTensorPtrSrc[i].ltrbROI.rb.y = 210;
            }
            roiTypeSrc = RpptRoiType::LTRB;
            roiTypeDst = RpptRoiType::LTRB;*/

            hipMemcpy(d_roiTensorPtrSrc, roiTensorPtrSrc, images * sizeof(RpptROI), hipMemcpyHostToDevice);

            start = clock();

            if (ip_bitDepth == 0)
                rppt_brightness_gpu(d_input, srcDescPtr, d_output, dstDescPtr, alpha, beta, d_roiTensorPtrSrc, roiTypeSrc, handle);
            else if (ip_bitDepth == 1)
                rppt_brightness_gpu(d_inputf16, srcDescPtr, d_outputf16, dstDescPtr, alpha, beta, d_roiTensorPtrSrc, roiTypeSrc, handle);
            else if (ip_bitDepth == 2)
                rppt_brightness_gpu(d_inputf32, srcDescPtr, d_outputf32, dstDescPtr, alpha, beta, d_roiTensorPtrSrc, roiTypeSrc, handle);
            else if (ip_bitDepth == 3)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 4)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 5)
                rppt_brightness_gpu(d_inputi8, srcDescPtr, d_outputi8, dstDescPtr, alpha, beta, d_roiTensorPtrSrc, roiTypeSrc, handle);
            else if (ip_bitDepth == 6)
                missingFuncFlag = 1;
            else
                missingFuncFlag = 1;

            break;
        }
        case 1:
        {
            test_case_name = "gamma_correction";

            Rpp32f gammaVal[images];
            for (i = 0; i < images; i++)
            {
                gammaVal[i] = 1.9;
            }

            // Uncomment to run test case with an xywhROI override
            /*for (i = 0; i < images; i++)
            {
                roiTensorPtrSrc[i].xywhROI.xy.x = 0;
                roiTensorPtrSrc[i].xywhROI.xy.y = 0;
                roiTensorPtrSrc[i].xywhROI.roiWidth = 100;
                roiTensorPtrSrc[i].xywhROI.roiHeight = 180;
            }*/

            // Uncomment to run test case with an ltrbROI override
            /*for (i = 0; i < images; i++)
                roiTensorPtrSrc[i].ltrbROI.lt.x = 50;
                roiTensorPtrSrc[i].ltrbROI.lt.y = 30;
                roiTensorPtrSrc[i].ltrbROI.rb.x = 210;
                roiTensorPtrSrc[i].ltrbROI.rb.y = 210;
            }
            roiTypeSrc = RpptRoiType::LTRB;
            roiTypeDst = RpptRoiType::LTRB;*/

            hipMemcpy(d_roiTensorPtrSrc, roiTensorPtrSrc, images * sizeof(RpptROI), hipMemcpyHostToDevice);

            start = clock();

            if (ip_bitDepth == 0)
                rppt_gamma_correction_gpu(d_input, srcDescPtr, d_output, dstDescPtr, gammaVal, d_roiTensorPtrSrc, roiTypeSrc, handle);
            else if (ip_bitDepth == 1)
                rppt_gamma_correction_gpu(d_inputf16, srcDescPtr, d_outputf16, dstDescPtr, gammaVal, d_roiTensorPtrSrc, roiTypeSrc, handle);
            else if (ip_bitDepth == 2)
                rppt_gamma_correction_gpu(d_inputf32, srcDescPtr, d_outputf32, dstDescPtr, gammaVal, d_roiTensorPtrSrc, roiTypeSrc, handle);
            else if (ip_bitDepth == 3)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 4)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 5)
                rppt_gamma_correction_gpu(d_inputi8, srcDescPtr, d_outputi8, dstDescPtr, gammaVal, d_roiTensorPtrSrc, roiTypeSrc, handle);
            else if (ip_bitDepth == 6)
                missingFuncFlag = 1;
            else
                missingFuncFlag = 1;

            break;
        }
        case 2:
        {
            test_case_name = "blend";

            Rpp32f alpha[images];
            for (i = 0; i < images; i++)
            {
                alpha[i] = 0.4;
            }

            // Uncomment to run test case with an xywhROI override
            /*for (i = 0; i < images; i++)
            {
                roiTensorPtrSrc[i].xywhROI.xy.x = 0;
                roiTensorPtrSrc[i].xywhROI.xy.y = 0;
                roiTensorPtrSrc[i].xywhROI.roiWidth = 100;
                roiTensorPtrSrc[i].xywhROI.roiHeight = 180;
            }*/

            // Uncomment to run test case with an ltrbROI override
            /*for (i = 0; i < images; i++)
                roiTensorPtrSrc[i].ltrbROI.lt.x = 50;
                roiTensorPtrSrc[i].ltrbROI.lt.y = 30;
                roiTensorPtrSrc[i].ltrbROI.rb.x = 210;
                roiTensorPtrSrc[i].ltrbROI.rb.y = 210;
            }
            roiTypeSrc = RpptRoiType::LTRB;
            roiTypeDst = RpptRoiType::LTRB;*/

            hipMemcpy(d_roiTensorPtrSrc, roiTensorPtrSrc, images * sizeof(RpptROI), hipMemcpyHostToDevice);

            start = clock();

            if (ip_bitDepth == 0)
                rppt_blend_gpu(d_input, d_input_second, srcDescPtr, d_output, dstDescPtr, alpha, d_roiTensorPtrSrc, roiTypeSrc, handle);
            else if (ip_bitDepth == 1)
                rppt_blend_gpu(d_inputf16, d_inputf16_second, srcDescPtr, d_outputf16, dstDescPtr, alpha, d_roiTensorPtrSrc, roiTypeSrc, handle);
            else if (ip_bitDepth == 2)
                rppt_blend_gpu(d_inputf32, d_inputf32_second, srcDescPtr, d_outputf32, dstDescPtr, alpha, d_roiTensorPtrSrc, roiTypeSrc, handle);
            else if (ip_bitDepth == 3)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 4)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 5)
                rppt_blend_gpu(d_inputi8, d_inputi8_second, srcDescPtr, d_outputi8, dstDescPtr, alpha, d_roiTensorPtrSrc, roiTypeSrc, handle);
            else if (ip_bitDepth == 6)
                missingFuncFlag = 1;
            else
                missingFuncFlag = 1;

            break;
        }
        case 37:
        {
            test_case_name = "crop";

            // Uncomment to run test case with an xywhROI override
            for (i = 0; i < images; i++)
            {
                roiTensorPtrSrc[i].xywhROI.xy.x = 0;
                roiTensorPtrSrc[i].xywhROI.xy.y = 0;
                roiTensorPtrSrc[i].xywhROI.roiWidth = 100;
                roiTensorPtrSrc[i].xywhROI.roiHeight = 180;
            }

            // Uncomment to run test case with an ltrbROI override
            /*for (i = 0; i < images; i++)
                roiTensorPtrSrc[i].ltrbROI.lt.x = 50;
                roiTensorPtrSrc[i].ltrbROI.lt.y = 30;
                roiTensorPtrSrc[i].ltrbROI.rb.x = 210;
                roiTensorPtrSrc[i].ltrbROI.rb.y = 210;
            }
            roiTypeSrc = RpptRoiType::LTRB;
            roiTypeDst = RpptRoiType::LTRB;*/

            hipMemcpy(d_roiTensorPtrSrc, roiTensorPtrSrc, images * sizeof(RpptROI), hipMemcpyHostToDevice);

            start = clock();

            if (ip_bitDepth == 0)
                rppt_crop_gpu(d_input, srcDescPtr, d_output, dstDescPtr, d_roiTensorPtrSrc, roiTypeSrc, handle);
            else if (ip_bitDepth == 1)
                rppt_crop_gpu(d_inputf16, srcDescPtr, d_outputf16, dstDescPtr, d_roiTensorPtrSrc, roiTypeSrc, handle);
            else if (ip_bitDepth == 2)
                rppt_crop_gpu(d_inputf32, srcDescPtr, d_outputf32, dstDescPtr, d_roiTensorPtrSrc, roiTypeSrc, handle);
            else if (ip_bitDepth == 3)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 4)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 5)
                rppt_crop_gpu(d_inputi8, srcDescPtr, d_outputi8, dstDescPtr, d_roiTensorPtrSrc, roiTypeSrc, handle);
            else if (ip_bitDepth == 6)
                missingFuncFlag = 1;
            else
                missingFuncFlag = 1;

            break;
        }
        case 40:
        {
            test_case_name = "erode";

            Rpp32u kernelSize = additionalParam;

            // Uncomment to run test case with an xywhROI override
            /*for (i = 0; i < images; i++)
            {
                roiTensorPtrSrc[i].xywhROI.xy.x = 0;
                roiTensorPtrSrc[i].xywhROI.xy.y = 0;
                roiTensorPtrSrc[i].xywhROI.roiWidth = 100;
                roiTensorPtrSrc[i].xywhROI.roiHeight = 180;
            }*/

            // Uncomment to run test case with an ltrbROI override
            /*for (i = 0; i < images; i++)
                roiTensorPtrSrc[i].ltrbROI.lt.x = 50;
                roiTensorPtrSrc[i].ltrbROI.lt.y = 30;
                roiTensorPtrSrc[i].ltrbROI.rb.x = 210;
                roiTensorPtrSrc[i].ltrbROI.rb.y = 210;
            }
            roiTypeSrc = RpptRoiType::LTRB;
            roiTypeDst = RpptRoiType::LTRB;*/

            hipMemcpy(d_roiTensorPtrSrc, roiTensorPtrSrc, images * sizeof(RpptROI), hipMemcpyHostToDevice);

            start = clock();

            if (ip_bitDepth == 0)
                rppt_erode_gpu(d_input, srcDescPtr, d_output, dstDescPtr, kernelSize, d_roiTensorPtrSrc, roiTypeSrc, handle);
            else if (ip_bitDepth == 1)
                rppt_erode_gpu(d_inputf16, srcDescPtr, d_outputf16, dstDescPtr, kernelSize, d_roiTensorPtrSrc, roiTypeSrc, handle);
            else if (ip_bitDepth == 2)
                rppt_erode_gpu(d_inputf32, srcDescPtr, d_outputf32, dstDescPtr, kernelSize, d_roiTensorPtrSrc, roiTypeSrc, handle);
            else if (ip_bitDepth == 3)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 4)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 5)
                rppt_erode_gpu(d_inputi8, srcDescPtr, d_outputi8, dstDescPtr, kernelSize, d_roiTensorPtrSrc, roiTypeSrc, handle);
            else if (ip_bitDepth == 6)
                missingFuncFlag = 1;
            else
                missingFuncFlag = 1;

            break;
        }
        case 41:
        {
            test_case_name = "dilate";

            Rpp32u kernelSize = additionalParam;

            // Uncomment to run test case with an xywhROI override
            /*for (i = 0; i < images; i++)
            {
                roiTensorPtrSrc[i].xywhROI.xy.x = 0;
                roiTensorPtrSrc[i].xywhROI.xy.y = 0;
                roiTensorPtrSrc[i].xywhROI.roiWidth = 100;
                roiTensorPtrSrc[i].xywhROI.roiHeight = 180;
            }*/

            // Uncomment to run test case with an ltrbROI override
            /*for (i = 0; i < images; i++)
                roiTensorPtrSrc[i].ltrbROI.lt.x = 50;
                roiTensorPtrSrc[i].ltrbROI.lt.y = 30;
                roiTensorPtrSrc[i].ltrbROI.rb.x = 210;
                roiTensorPtrSrc[i].ltrbROI.rb.y = 210;
            }
            roiTypeSrc = RpptRoiType::LTRB;
            roiTypeDst = RpptRoiType::LTRB;*/

            hipMemcpy(d_roiTensorPtrSrc, roiTensorPtrSrc, images * sizeof(RpptROI), hipMemcpyHostToDevice);

            start = clock();

            if (ip_bitDepth == 0)
                rppt_dilate_gpu(d_input, srcDescPtr, d_output, dstDescPtr, kernelSize, d_roiTensorPtrSrc, roiTypeSrc, handle);
            else if (ip_bitDepth == 1)
                rppt_dilate_gpu(d_inputf16, srcDescPtr, d_outputf16, dstDescPtr, kernelSize, d_roiTensorPtrSrc, roiTypeSrc, handle);
            else if (ip_bitDepth == 2)
                rppt_dilate_gpu(d_inputf32, srcDescPtr, d_outputf32, dstDescPtr, kernelSize, d_roiTensorPtrSrc, roiTypeSrc, handle);
            else if (ip_bitDepth == 3)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 4)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 5)
                rppt_dilate_gpu(d_inputi8, srcDescPtr, d_outputi8, dstDescPtr, kernelSize, d_roiTensorPtrSrc, roiTypeSrc, handle);
            else if (ip_bitDepth == 6)
                missingFuncFlag = 1;
            else
                missingFuncFlag = 1;

            break;
        }
        case 49:
        {
            test_case_name = "box_filter";

            Rpp32u kernelSize = additionalParam;

            // Uncomment to run test case with an xywhROI override
            /*for (i = 0; i < images; i++)
            {
                roiTensorPtrSrc[i].xywhROI.xy.x = 0;
                roiTensorPtrSrc[i].xywhROI.xy.y = 0;
                roiTensorPtrSrc[i].xywhROI.roiWidth = 100;
                roiTensorPtrSrc[i].xywhROI.roiHeight = 180;
            }*/

            // Uncomment to run test case with an ltrbROI override
            /*for (i = 0; i < images; i++)
                roiTensorPtrSrc[i].ltrbROI.lt.x = 50;
                roiTensorPtrSrc[i].ltrbROI.lt.y = 30;
                roiTensorPtrSrc[i].ltrbROI.rb.x = 210;
                roiTensorPtrSrc[i].ltrbROI.rb.y = 210;
            }
            roiTypeSrc = RpptRoiType::LTRB;
            roiTypeDst = RpptRoiType::LTRB;*/

            hipMemcpy(d_roiTensorPtrSrc, roiTensorPtrSrc, images * sizeof(RpptROI), hipMemcpyHostToDevice);

            start = clock();

            if (ip_bitDepth == 0)
                rppt_box_filter_gpu(d_input, srcDescPtr, d_output, dstDescPtr, kernelSize, d_roiTensorPtrSrc, roiTypeSrc, handle);
            else if (ip_bitDepth == 1)
                rppt_box_filter_gpu(d_inputf16, srcDescPtr, d_outputf16, dstDescPtr, kernelSize, d_roiTensorPtrSrc, roiTypeSrc, handle);
            else if (ip_bitDepth == 2)
                rppt_box_filter_gpu(d_inputf32, srcDescPtr, d_outputf32, dstDescPtr, kernelSize, d_roiTensorPtrSrc, roiTypeSrc, handle);
            else if (ip_bitDepth == 3)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 4)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 5)
                rppt_box_filter_gpu(d_inputi8, srcDescPtr, d_outputi8, dstDescPtr, kernelSize, d_roiTensorPtrSrc, roiTypeSrc, handle);
            else if (ip_bitDepth == 6)
                missingFuncFlag = 1;
            else
                missingFuncFlag = 1;

            break;
        }
        case 82:
        {
            test_case_name = "ricap";
            float _beta_param = 0.3;
            std::random_device rd;
            std::mt19937 gen(rd());
            static std::uniform_real_distribution<double> unif(0.3, 0.7);
            double p = unif(gen);
            double randFromDist = boost::math::ibeta_inv(_beta_param, _beta_param, p);

            std::random_device rd1;
            std::mt19937 gen1(rd1());
            static std::uniform_real_distribution<double> unif1(0.3, 0.7);
            double p1 = unif1(gen1);
            double randFromDist1 = boost::math::ibeta_inv(_beta_param, _beta_param, p1);

            uint32_t iX = maxDstWidth;
            uint32_t iY = maxDstHeight;
            srcDescPtr->w = maxDstWidth;
            dstDescPtr->w = maxDstHeight;

            Rpp32u initialPermuteArray[images], permutedArray[images * 4], permutationTensor[images * 4];
            for (uint i = 0; i < images; i++)
            {
                initialPermuteArray[i] = i;
            }
            randomize(initialPermuteArray, images);
            memcpy(permutedArray, initialPermuteArray, images * sizeof(Rpp32u));
            randomize(initialPermuteArray, images);
            memcpy(permutedArray + images, initialPermuteArray, images * sizeof(Rpp32u));
            randomize(initialPermuteArray, images);
            memcpy(permutedArray + (images * 2), initialPermuteArray, images * sizeof(Rpp32u));
            randomize(initialPermuteArray, images);
            memcpy(permutedArray + (images * 3), initialPermuteArray, images * sizeof(Rpp32u));

            for (uint i = 0, j = 0; i < images && j < images * 4; i++, j += 4)
            {
                permutationTensor[j] = permutedArray[i];
                permutationTensor[j + 1] = permutedArray[i + images];
                permutationTensor[j + 2] = permutedArray[i + (images * 2)];
                permutationTensor[j + 3] = permutedArray[i + (images * 3)];
            }

            RpptROI roiPtrInputCropRegion[4];
            RpptROI *d_roiPtrInputCropRegion;
            hipMalloc(&d_roiPtrInputCropRegion, 4 * sizeof(RpptROI));


            // Uncomment to run test case with an xywhROI override
            // /*
            roiPtrInputCropRegion[0].xywhROI.roiWidth = std::round(randFromDist * iX);                    // w1
            roiPtrInputCropRegion[0].xywhROI.roiHeight = std::round(randFromDist1 * iY);                  // h1
            roiPtrInputCropRegion[1].xywhROI.roiWidth = iX - roiPtrInputCropRegion[0].xywhROI.roiWidth;   // w2
            roiPtrInputCropRegion[1].xywhROI.roiHeight = roiPtrInputCropRegion[0].xywhROI.roiHeight;      // h2
            roiPtrInputCropRegion[2].xywhROI.roiWidth = roiPtrInputCropRegion[0].xywhROI.roiWidth;        // w3
            roiPtrInputCropRegion[2].xywhROI.roiHeight = iY - roiPtrInputCropRegion[0].xywhROI.roiHeight; // h3
            roiPtrInputCropRegion[3].xywhROI.roiWidth = iX - roiPtrInputCropRegion[0].xywhROI.roiWidth;   // w4
            roiPtrInputCropRegion[3].xywhROI.roiHeight = iY - roiPtrInputCropRegion[0].xywhROI.roiHeight; // h4
            roiPtrInputCropRegion[0].xywhROI.xy.x = random_val(0, iX - roiPtrInputCropRegion[0].xywhROI.roiWidth);  // x1
            roiPtrInputCropRegion[0].xywhROI.xy.y = random_val(0, iY - roiPtrInputCropRegion[0].xywhROI.roiHeight); // y1
            roiPtrInputCropRegion[1].xywhROI.xy.x = random_val(0, iX - roiPtrInputCropRegion[1].xywhROI.roiWidth);  // x2
            roiPtrInputCropRegion[1].xywhROI.xy.y = random_val(0, iY - roiPtrInputCropRegion[1].xywhROI.roiHeight); // y2
            roiPtrInputCropRegion[2].xywhROI.xy.x = random_val(0, iX - roiPtrInputCropRegion[2].xywhROI.roiWidth);  // x3
            roiPtrInputCropRegion[2].xywhROI.xy.y = random_val(0, iY - roiPtrInputCropRegion[2].xywhROI.roiHeight); // y3
            roiPtrInputCropRegion[3].xywhROI.xy.x = random_val(0, iX - roiPtrInputCropRegion[3].xywhROI.roiWidth);  // x4
            roiPtrInputCropRegion[3].xywhROI.xy.y = random_val(0, iY - roiPtrInputCropRegion[3].xywhROI.roiHeight); // y4
            // */

            // Uncomment to run test case with an ltrbROI override
            /*
            double w1, h1, w2, h2, w3, h3, w4, h4;
            double r1, b1, r2, b2, r3, b3, r4, b4;
            w1 = std::round(randFromDist * iX);                             // w1
            h1 = std::round(randFromDist1 * iY);                            // h1
            w2 = iX - w1;                                                   // w2
            h2 = h1;                                                        // h2
            w3 = w1;                                                        // w3
            h3 = iY - h1;                                                   // h3
            w4 = iX - w1;                                                   // w4
            h4 = iY - h1;                                                   // h4
            roiPtrInputCropRegion[0].ltrbROI.lt.x = random_val(0, iX - w1); // l1
            roiPtrInputCropRegion[0].ltrbROI.lt.y = random_val(0, iY - h1); // t1
            roiPtrInputCropRegion[1].ltrbROI.lt.x = random_val(0, iX - w2); // l2
            roiPtrInputCropRegion[1].ltrbROI.lt.y = random_val(0, iY - h2); // t2
            roiPtrInputCropRegion[2].ltrbROI.lt.x = random_val(0, iX - w3); // l3
            roiPtrInputCropRegion[2].ltrbROI.lt.y = random_val(0, iY - h3); // t3
            roiPtrInputCropRegion[3].ltrbROI.lt.x = random_val(0, iX - w4); // l4
            roiPtrInputCropRegion[3].ltrbROI.lt.y = random_val(0, iY - h4); // t4
            roiPtrInputCropRegion[0].ltrbROI.rb.x = roiPtrInputCropRegion[0].ltrbROI.lt.x + w1 - 1; // r1
            roiPtrInputCropRegion[0].ltrbROI.rb.y = roiPtrInputCropRegion[0].ltrbROI.lt.y + h1 - 1; // b1
            roiPtrInputCropRegion[1].ltrbROI.rb.x = roiPtrInputCropRegion[1].ltrbROI.lt.x + w2 - 1; // r2
            roiPtrInputCropRegion[1].ltrbROI.rb.y = roiPtrInputCropRegion[1].ltrbROI.lt.y + h2 - 1; // b2
            roiPtrInputCropRegion[2].ltrbROI.rb.x = roiPtrInputCropRegion[2].ltrbROI.lt.x + w3 - 1; // r3
            roiPtrInputCropRegion[2].ltrbROI.rb.y = roiPtrInputCropRegion[2].ltrbROI.lt.y + h3 - 1; // b3
            roiPtrInputCropRegion[3].ltrbROI.rb.x = roiPtrInputCropRegion[3].ltrbROI.lt.x + w4 - 1; // r4
            roiPtrInputCropRegion[3].ltrbROI.rb.y = roiPtrInputCropRegion[3].ltrbROI.lt.y + h4 - 1; // b4
            roiTypeSrc = RpptRoiType::LTRB;
            roiTypeDst = RpptRoiType::LTRB;
            */

            hipError_t err;
            err = hipMemcpy(d_roiPtrInputCropRegion, roiPtrInputCropRegion, 4 * sizeof(RpptROI), hipMemcpyHostToDevice);
            if (err)
            {
                std::cerr<<"\n RICAP roiPtrInputCropRegion hipMemcpy failed with err "<<err;
                exit(0);
            }

            start = clock();

            if (ip_bitDepth == 0)
                rppt_ricap_gpu(d_input, srcDescPtr, d_output, dstDescPtr, permutationTensor, d_roiPtrInputCropRegion, roiTypeSrc, handle);
            else if (ip_bitDepth == 1)
                rppt_ricap_gpu(d_inputf16, srcDescPtr, d_outputf16, dstDescPtr, permutationTensor, d_roiPtrInputCropRegion, roiTypeSrc, handle);
            else if (ip_bitDepth == 2)
                rppt_ricap_gpu(d_inputf32, srcDescPtr, d_outputf32, dstDescPtr, permutationTensor, d_roiPtrInputCropRegion, roiTypeSrc, handle);
             else if (ip_bitDepth == 3)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 4)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 5)
                rppt_ricap_gpu(d_inputi8, srcDescPtr, d_outputi8, dstDescPtr, permutationTensor, d_roiPtrInputCropRegion, roiTypeSrc, handle);
            else if (ip_bitDepth == 6)
                missingFuncFlag = 1;
            else
                missingFuncFlag = 1;

            hipDeviceSynchronize();

            end = clock();

            break;
        }
        case 83:
        {
            test_case_name = "gridmask";

            Rpp32u tileWidth = 40;
            Rpp32f gridRatio = 0.6;
            Rpp32f gridAngle = 0.5;
            RpptUintVector2D translateVector;
            translateVector.x = 0.0;
            translateVector.y = 0.0;

            // Uncomment to run test case with an xywhROI override
            /*for (i = 0; i < images; i++)
            {
                roiTensorPtrSrc[i].xywhROI.xy.x = 0;
                roiTensorPtrSrc[i].xywhROI.xy.y = 0;
                roiTensorPtrSrc[i].xywhROI.roiWidth = 100;
                roiTensorPtrSrc[i].xywhROI.roiHeight = 180;
            }*/

            // Uncomment to run test case with an ltrbROI override
            /*for (i = 0; i < images; i++)
                roiTensorPtrSrc[i].ltrbROI.lt.x = 50;
                roiTensorPtrSrc[i].ltrbROI.lt.y = 30;
                roiTensorPtrSrc[i].ltrbROI.rb.x = 210;
                roiTensorPtrSrc[i].ltrbROI.rb.y = 210;
            }
            roiTypeSrc = RpptRoiType::LTRB;
            roiTypeDst = RpptRoiType::LTRB;*/


            hipMemcpy(d_roiTensorPtrSrc, roiTensorPtrSrc, images * sizeof(RpptROI), hipMemcpyHostToDevice);

            start = clock();

            if (ip_bitDepth == 0)
                rppt_gridmask_gpu(d_input, srcDescPtr, d_output, dstDescPtr, tileWidth, gridRatio, gridAngle, translateVector, d_roiTensorPtrSrc, roiTypeSrc, handle);
            else if (ip_bitDepth == 1)
                rppt_gridmask_gpu(d_inputf16, srcDescPtr, d_outputf16, dstDescPtr, tileWidth, gridRatio, gridAngle, translateVector, d_roiTensorPtrSrc, roiTypeSrc, handle);
            else if (ip_bitDepth == 2)
                rppt_gridmask_gpu(d_inputf32, srcDescPtr, d_outputf32, dstDescPtr, tileWidth, gridRatio, gridAngle, translateVector, d_roiTensorPtrSrc, roiTypeSrc, handle);
            else if (ip_bitDepth == 3)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 4)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 5)
                rppt_gridmask_gpu(d_inputi8, srcDescPtr, d_outputi8, dstDescPtr, tileWidth, gridRatio, gridAngle, translateVector, d_roiTensorPtrSrc, roiTypeSrc, handle);
            else if (ip_bitDepth == 6)
                missingFuncFlag = 1;
            else
                missingFuncFlag = 1;

            break;
        }
        case 84:
        {
            test_case_name = "spatter";

            RpptRGB spatterColor;

            // Mud Spatter
            spatterColor.R = 65;
            spatterColor.G = 50;
            spatterColor.B = 23;

            // Blood Spatter
            // spatterColor.R = 98;
            // spatterColor.G = 3;
            // spatterColor.B = 3;

            // Ink Spatter
            // spatterColor.R = 5;
            // spatterColor.G = 20;
            // spatterColor.B = 64;

            // Uncomment to run test case with an xywhROI override
            /*for (i = 0; i < images; i++)
            {
                roiTensorPtrSrc[i].xywhROI.xy.x = 0;
                roiTensorPtrSrc[i].xywhROI.xy.y = 0;
                roiTensorPtrSrc[i].xywhROI.roiWidth = 100;
                roiTensorPtrSrc[i].xywhROI.roiHeight = 180;
            }*/

            // Uncomment to run test case with an ltrbROI override
            /*for (i = 0; i < images; i++)
                roiTensorPtrSrc[i].ltrbROI.lt.x = 50;
                roiTensorPtrSrc[i].ltrbROI.lt.y = 30;
                roiTensorPtrSrc[i].ltrbROI.rb.x = 210;
                roiTensorPtrSrc[i].ltrbROI.rb.y = 210;
            }
            roiTypeSrc = RpptRoiType::LTRB;
            roiTypeDst = RpptRoiType::LTRB;*/


            hipMemcpy(d_roiTensorPtrSrc, roiTensorPtrSrc, images * sizeof(RpptROI), hipMemcpyHostToDevice);

            start = clock();

            if (ip_bitDepth == 0)
                rppt_spatter_gpu(d_input, srcDescPtr, d_output, dstDescPtr, spatterColor, d_roiTensorPtrSrc, roiTypeSrc, handle);
            else if (ip_bitDepth == 1)
                rppt_spatter_gpu(d_inputf16, srcDescPtr, d_outputf16, dstDescPtr, spatterColor, d_roiTensorPtrSrc, roiTypeSrc, handle);
            else if (ip_bitDepth == 2)
                rppt_spatter_gpu(d_inputf32, srcDescPtr, d_outputf32, dstDescPtr, spatterColor, d_roiTensorPtrSrc, roiTypeSrc, handle);
            else if (ip_bitDepth == 3)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 4)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 5)
                rppt_spatter_gpu(d_inputi8, srcDescPtr, d_outputi8, dstDescPtr, spatterColor, d_roiTensorPtrSrc, roiTypeSrc, handle);
            else if (ip_bitDepth == 6)
                missingFuncFlag = 1;
            else
                missingFuncFlag = 1;

            hipDeviceSynchronize();

            end = clock();

            break;
        }
        default:
            missingFuncFlag = 1;
            break;
        }

        hipDeviceSynchronize();
        end = clock();

        if (missingFuncFlag == 1)
        {
            printf("\nThe functionality %s doesn't yet exist in RPP\n", func);
            return -1;
        }

        // Display measured times

        gpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
        if (gpu_time_used > max_time_used)
            max_time_used = gpu_time_used;
        if (gpu_time_used < min_time_used)
            min_time_used = gpu_time_used;
        avg_time_used += gpu_time_used;
    }

    avg_time_used /= 100;
    cout << fixed << "\nmax,min,avg = " << max_time_used << "," << min_time_used << "," << avg_time_used << endl;

    rppDestroyGPU(handle);

    // Free memory

    free(roiTensorPtrSrc);
    free(roiTensorPtrDst);
    hipFree(d_roiTensorPtrSrc);
    hipFree(d_roiTensorPtrDst);
    free(input);
    free(input_second);
    free(output);

    if (ip_bitDepth == 0)
    {
        hipFree(d_input);
        hipFree(d_input_second);
        hipFree(d_output);
    }
    else if (ip_bitDepth == 1)
    {
        free(inputf16);
        free(inputf16_second);
        free(outputf16);
        hipFree(d_inputf16);
        hipFree(d_inputf16_second);
        hipFree(d_outputf16);
    }
    else if (ip_bitDepth == 2)
    {
        free(inputf32);
        free(inputf32_second);
        free(outputf32);
        hipFree(d_inputf32);
        hipFree(d_inputf32_second);
        hipFree(d_outputf32);
    }
    else if (ip_bitDepth == 3)
    {
        free(outputf16);
        hipFree(d_input);
        hipFree(d_input_second);
        hipFree(d_outputf16);
    }
    else if (ip_bitDepth == 4)
    {
        free(outputf32);
        hipFree(d_input);
        hipFree(d_input_second);
        hipFree(d_outputf32);
    }
    else if (ip_bitDepth == 5)
    {
        free(inputi8);
        free(inputi8_second);
        free(outputi8);
        hipFree(d_inputi8);
        hipFree(d_inputi8_second);
        hipFree(d_outputi8);
    }
    else if (ip_bitDepth == 6)
    {
        free(outputi8);
        hipFree(d_input);
        hipFree(d_input_second);
        hipFree(d_outputi8);
    }

    return 0;
}
