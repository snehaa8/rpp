#include <stdio.h>
#include <dirent.h>
#include <string.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include "rpp.h"
#include "../rpp_test_suite_common.h"
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <time.h>
#include <omp.h>
#include <half/half.hpp>
#include <fstream>

using namespace cv;
using namespace std;

using half_float::half;
typedef half Rpp16f;

inline size_t get_size_of_data_type(RpptDataType dataType)
{
    if(dataType == RpptDataType::U8)
        return sizeof(Rpp8u);
    else if(dataType == RpptDataType::I8)
        return sizeof(Rpp8s);
    else if(dataType == RpptDataType::F16)
        return sizeof(Rpp16f);
    else if(dataType == RpptDataType::F32)
        return sizeof(Rpp32f);
    else
        return 0;
}

int main(int argc, char **argv)
{
    // Handle inputs
    const int MIN_ARG_COUNT = 11;

    char *src = argv[1];
    char *srcSecond = argv[2];
    string dst = argv[3];

    int inputBitDepth = atoi(argv[4]);
    unsigned int outputFormatToggle = atoi(argv[5]);
    int testCase = atoi(argv[6]);
    int numIterations = atoi(argv[8]);
    int testType = atoi(argv[9]);     // 0 for unit and 1 for performance test
    int layoutType = atoi(argv[10]); // 0 for pkd3 / 1 for pln3 / 2 for pln1

    bool additionalParamCase = (testCase == 8 || testCase == 21 || testCase == 23 || testCase == 24);
    bool kernelSizeCase = false;
    bool interpolationTypeCase = (testCase == 21 || testCase == 23 || testCase == 24);
    bool noiseTypeCase = (testCase == 8);
    bool pln1OutTypeCase = (testCase == 86);
    unsigned int verbosity = atoi(argv[11]);
    unsigned int additionalParam = additionalParamCase ? atoi(argv[7]) : 1;

    if (verbosity == 1)
    {
        printf("\nInputs for this test case are:");
        printf("\nsrc1 = %s", argv[1]);
        printf("\nsrc2 = %s", argv[2]);
        if (testType == 0)
            printf("\ndst = %s", argv[3]);
        printf("\nu8 / f16 / f32 / u8->f16 / u8->f32 / i8 / u8->i8 (0/1/2/3/4/5/6) = %s", argv[4]);
        printf("\noutputFormatToggle (pkd->pkd = 0 / pkd->pln = 1) = %s", argv[5]);
        printf("\ncase number (0:86) = %s", argv[6]);
        printf("\nNumber of times to run = %s", argv[8]);
        printf("\nUnit test/Performance test - 0/1 = %s", argv[9]);
    }

    if (argc < MIN_ARG_COUNT)
    {
        printf("\nImproper Usage! Needs all arguments!\n");
        if (testType == 0)
        {
            printf("\nUsage: <src1 folder> <src2 folder (place same as src1 folder for single image functionalities)> <dst folder> <u8 = 0 / f16 = 1 / f32 = 2 / u8->f16 = 3 / u8->f32 = 4 / i8 = 5 / u8->i8 = 6> <outputFormatToggle (pkd->pkd = 0 / pkd->pln = 1)> <case number = 0:86> <number of iterations > 0> <verbosity = 0/1>>\n");
            if (layoutType == 2)
            {
                if (atoi(argv[5]) != 0)
                {
                    printf("\nPLN1 cases don't have outputFormatToggle! Please input outputFormatToggle = 0\n");
                    return -1;
                }
            }
        }
    }

    int inputChannels;
    string funcType;

    // Initialize tensor descriptors
    RpptDesc srcDesc, dstDesc;
    RpptDescPtr srcDescPtr, dstDescPtr;
    srcDescPtr = &srcDesc;
    dstDescPtr = &dstDesc;

    if(layoutType == 0)
    {
        inputChannels = 3;
        funcType = "Tensor_HOST_PKD3";
        srcDescPtr->layout = RpptLayout::NHWC;

        // Set src/dst layouts in tensor descriptors
        if (pln1OutTypeCase)
        {
            funcType += "_toPLN1";
            dstDescPtr->layout = RpptLayout::NCHW;
        }
        else
        {
            if (outputFormatToggle == 0)
            {
                funcType += "_toPKD3";
                dstDescPtr->layout = RpptLayout::NHWC;
            }
            else if (outputFormatToggle == 1)
            {
                funcType += "_toPLN3";
                dstDescPtr->layout = RpptLayout::NCHW;
            }
        }
    }
    else if(layoutType == 1)
    {
        inputChannels = 3;
        funcType = "Tensor_HOST_PLN3";
        srcDescPtr->layout = RpptLayout::NCHW;

        // Set src/dst layouts in tensor descriptors
        if (pln1OutTypeCase)
        {
            funcType += "_toPLN1";
            dstDescPtr->layout = RpptLayout::NCHW;
        }
        else
        {
            if (outputFormatToggle == 0)
            {
                funcType += "_toPLN3";
                dstDescPtr->layout = RpptLayout::NCHW;
            }
            else if (outputFormatToggle == 1)
            {
                funcType += "_toPKD3";
                dstDescPtr->layout = RpptLayout::NHWC;
            }
        }
    }
    else
    {
        inputChannels = 1;
        funcType = "Tensor_HOST_PLN1";
        funcType += "_toPLN1";

        // Set src/dst layouts in tensor descriptors
        srcDescPtr->layout = RpptLayout::NCHW;
        dstDescPtr->layout = RpptLayout::NCHW;
    }

    // Get function name
    string funcName = augmentationMap[testCase];
    funcName = (funcName.empty()) ? "testCase" : funcName;

    // Set src/dst data types in tensor descriptors
    set_data_type(inputBitDepth, funcName, srcDescPtr, dstDescPtr);

    // Other initializations
    int missingFuncFlag = 0;
    int i = 0, j = 0;
    int maxHeight = 0, maxWidth = 0;
    int maxDstHeight = 0, maxDstWidth = 0;
    Rpp64u count = 0;
    Rpp64u ioBufferSize = 0;
    Rpp64u oBufferSize = 0;
    static int noOfImages = 0;
    Mat image, imageSecond;

    // String ops on function name
    string src1 = "";
    src1 = src;
    src1 += "/";
    string src1Second = "";
    src1Second = srcSecond;
    src1Second += "/";

    string func="";
    func = funcName;
    func += funcType;

    RpptInterpolationType interpolationType = RpptInterpolationType::BILINEAR;
    if (kernelSizeCase)
    {
        char additionalParam_char[2];
        std::sprintf(additionalParam_char, "%u", additionalParam);
        func += "_kSize";
        func += additionalParam_char;
    }
    else if (interpolationTypeCase)
    {
        std::string interpolationTypeName;
        interpolationTypeName = get_interpolation_type(additionalParam, interpolationType);
        func += "_interpolationType";
        func += interpolationTypeName.c_str();
    }
    else if (noiseTypeCase)
    {
        std::string noiseTypeName;
        noiseTypeName = get_noise_type(additionalParam);
        func += "_noiseType";
        func += noiseTypeName.c_str();
    }
    printf("\nRunning %s...", func.c_str());
    dst += "/";
    dst += func;

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

    if(!noOfImages)
    {
        std::cerr<<"Not able to find any images in the folder specified. Please check the input path";
        exit(0);
    }

    // Initialize ROI tensors for src/dst
    RpptROI *roiTensorPtrSrc = (RpptROI *)calloc(noOfImages, sizeof(RpptROI));
    RpptROI *roiTensorPtrDst = (RpptROI *)calloc(noOfImages, sizeof(RpptROI));

    // Initialize the ImagePatch for source and destination
    RpptImagePatch *srcImgSizes = (RpptImagePatch *)calloc(noOfImages, sizeof(RpptImagePatch));
    RpptImagePatch *dstImgSizes = (RpptImagePatch *)calloc(noOfImages, sizeof(RpptImagePatch));

    // Set ROI tensors types for src/dst
    RpptRoiType roiTypeSrc, roiTypeDst;
    roiTypeSrc = RpptRoiType::XYWH;
    roiTypeDst = RpptRoiType::XYWH;

    // Initialize roi that can be updated in case-wise augmentations if needed
    RpptROI roi;

    // Set maxHeight, maxWidth and ROIs for src/dst
    const int images = noOfImages;
    string imageNames[images];

    DIR *dr1 = opendir(src);
    while ((de = readdir(dr1)) != NULL)
    {
        if (strcmp(de->d_name, ".") == 0 || strcmp(de->d_name, "..") == 0)
            continue;
        imageNames[count] = de->d_name;
        string temp = src1;
        temp += imageNames[count];
        if (layoutType == 0 || layoutType == 1)
            image = imread(temp, 1);

        else
            image = imread(temp, 0);

        roiTensorPtrSrc[count].xywhROI.xy.x = 0;
        roiTensorPtrSrc[count].xywhROI.xy.y = 0;
        roiTensorPtrSrc[count].xywhROI.roiWidth = image.cols;
        roiTensorPtrSrc[count].xywhROI.roiHeight = image.rows;

        roiTensorPtrDst[count].xywhROI.xy.x = 0;
        roiTensorPtrDst[count].xywhROI.xy.y = 0;
        roiTensorPtrDst[count].xywhROI.roiWidth = image.cols;
        roiTensorPtrDst[count].xywhROI.roiHeight = image.rows;

        srcImgSizes[count].width = roiTensorPtrSrc[count].xywhROI.roiWidth;
        srcImgSizes[count].height = roiTensorPtrSrc[count].xywhROI.roiHeight;
        dstImgSizes[count].width = roiTensorPtrDst[count].xywhROI.roiWidth;
        dstImgSizes[count].height = roiTensorPtrDst[count].xywhROI.roiHeight;

        maxHeight = std::max(maxHeight, roiTensorPtrSrc[count].xywhROI.roiHeight);
        maxWidth = std::max(maxWidth, roiTensorPtrSrc[count].xywhROI.roiWidth);
        maxDstHeight = std::max(maxDstHeight, roiTensorPtrDst[count].xywhROI.roiHeight);
        maxDstWidth = std::max(maxDstWidth, roiTensorPtrDst[count].xywhROI.roiWidth);

        count++;
    }
    closedir(dr1);

    // Check if any of maxWidth and maxHeight is less than or equal to 0
    if(maxHeight <= 0 || maxWidth <= 0)
    {
        std::cerr<<"Unable to read images properly.Please check the input path of the files specified";
        exit(0);
    }

    Rpp32u outputChannels = inputChannels;
    if(pln1OutTypeCase)
        outputChannels = 1;
    Rpp32u offsetInBytes = 0;

    // Set numDims, offset, n/c/h/w values, strides for src/dst
    set_description_ptr_dims_and_strides(srcDescPtr, noOfImages, maxHeight, maxWidth, inputChannels, offsetInBytes);
    set_description_ptr_dims_and_strides(dstDescPtr, noOfImages, maxDstHeight, maxDstWidth, outputChannels, offsetInBytes);

    // Set buffer sizes for src/dst
    ioBufferSize = (Rpp64u)srcDescPtr->h * (Rpp64u)srcDescPtr->w * (Rpp64u)srcDescPtr->c * (Rpp64u)noOfImages;
    oBufferSize = (Rpp64u)dstDescPtr->h * (Rpp64u)dstDescPtr->w * (Rpp64u)dstDescPtr->c * (Rpp64u)noOfImages;

    // Set buffer sizes in bytes for src/dst (including offsets)
    Rpp64u ioBufferSizeInBytes_u8 = ioBufferSize + srcDescPtr->offsetInBytes;
    Rpp64u oBufferSizeInBytes_u8 = oBufferSize + dstDescPtr->offsetInBytes;
    Rpp64u inputBufferSize = ioBufferSize * get_size_of_data_type(srcDescPtr->dataType) + srcDescPtr->offsetInBytes;
    Rpp64u outputBufferSize = oBufferSize * get_size_of_data_type(dstDescPtr->dataType) + dstDescPtr->offsetInBytes;

    // Initialize 8u host buffers for src/dst
    Rpp8u *inputu8 = (Rpp8u *)calloc(ioBufferSizeInBytes_u8, 1);
    Rpp8u *inputu8Second = (Rpp8u *)calloc(ioBufferSizeInBytes_u8, 1);
    Rpp8u *outputu8 = (Rpp8u *)calloc(oBufferSizeInBytes_u8, 1);

    // Set 8u host buffers for src/dst
    DIR *dr2 = opendir(src);
    DIR *dr2Second = opendir(srcSecond);
    count = 0;
    i = 0;

    Rpp8u *offsettedInput, *offsettedInputSecond;
    offsettedInput = inputu8 + srcDescPtr->offsetInBytes;
    offsettedInputSecond = inputu8Second + srcDescPtr->offsetInBytes;
    string imageNamesPath[images];
    string imageNamesPathSecond[images];
    for(int i = 0; i < images; i++)
    {
        imageNamesPath[i] = src1 + "/" + imageNames[i];
        imageNamesPathSecond[i] = src1Second + "/" + imageNames[i];
    }

    // Read image from opencv to input buffer
    read_image_batch_opencv(inputu8, srcDescPtr, imageNamesPath);
    read_image_batch_opencv(inputu8Second, srcDescPtr, imageNamesPathSecond);

    // Convert inputs to test various other bit depths and copy to hip buffers
    void *input, *input_second, *output;
    if (layoutType == 1)
    {
        // Convert default OpenCV PKD3 to PLN3 for first and second input batch
        convert_pkd3_to_pln3(inputu8, srcDescPtr);
        convert_pkd3_to_pln3(inputu8Second, srcDescPtr);
    }

    // Factors to convert U8 data to F32, F16 data to 0-1 range and reconvert them back to 0 -255 range
    Rpp32f conversionFactor = 1.0f / 255.0;
    if(testCase == 38)
        conversionFactor = 1.0;
    Rpp32f invConversionFactor = 1.0f / conversionFactor;

    // Convert inputs to test various other bit depths
    input = (Rpp8u *)calloc(inputBufferSize, 1);
    input_second = (Rpp8u *)calloc(inputBufferSize, 1);
    output = (Rpp8u *)calloc(outputBufferSize, 1);

    if (inputBitDepth == 0)
    {
        memcpy(input, inputu8, inputBufferSize);
        memcpy(input_second, inputu8Second, inputBufferSize);
    }
    else if (inputBitDepth == 1)
    {
        Rpp8u *inputTemp, *inputSecondTemp;
        half *inputf16Temp, *inputf16SecondTemp;
        inputTemp = inputu8 + srcDescPtr->offsetInBytes;
        inputSecondTemp = inputu8Second + srcDescPtr->offsetInBytes;
        inputf16Temp = (half *)((Rpp8u *)input + srcDescPtr->offsetInBytes);
        inputf16SecondTemp = (half *)((Rpp8u *)input + srcDescPtr->offsetInBytes);

        for (int i = 0; i < ioBufferSize; i++)
        {
            *inputf16Temp = (half)(((float)*inputTemp) * conversionFactor);
            *inputf16SecondTemp = (half)(((float)*inputSecondTemp) * conversionFactor);
            inputTemp++;
            inputf16Temp++;
            inputSecondTemp++;
            inputf16SecondTemp++;
        }
    }
    else if (inputBitDepth == 2)
    {
        Rpp8u *inputTemp, *inputSecondTemp;
        Rpp32f *inputf32Temp, *inputf32SecondTemp;
        inputTemp = inputu8 + srcDescPtr->offsetInBytes;
        inputSecondTemp = inputu8Second + srcDescPtr->offsetInBytes;
        inputf32Temp = (Rpp32f *)((Rpp8u *)input + srcDescPtr->offsetInBytes);
        inputf32SecondTemp = (Rpp32f *)((Rpp8u *)input + srcDescPtr->offsetInBytes);

        for (int i = 0; i < ioBufferSize; i++)
        {
            *inputf32Temp = ((Rpp32f)*inputTemp) * conversionFactor;
            *inputf32SecondTemp = ((Rpp32f)*inputSecondTemp) * conversionFactor;
            inputTemp++;
            inputf32Temp++;
            inputSecondTemp++;
            inputf32SecondTemp++;
        }
    }
    else if (inputBitDepth == 5)
    {
        Rpp8u *inputTemp, *inputSecondTemp;
        Rpp8s *inputi8Temp, *inputi8SecondTemp;

        inputTemp = inputu8 + srcDescPtr->offsetInBytes;
        inputSecondTemp = inputu8Second + srcDescPtr->offsetInBytes;
        inputi8Temp = (Rpp8s *)input + srcDescPtr->offsetInBytes;
        inputi8SecondTemp = (Rpp8s *)input_second + srcDescPtr->offsetInBytes;

        for (int i = 0; i < ioBufferSize; i++)
        {
            *inputi8Temp = (Rpp8s) (((Rpp32s) *inputTemp) - 128);
            *inputi8SecondTemp = (Rpp8s) (((Rpp32s) *inputSecondTemp) - 128);
            inputTemp++;
            inputi8Temp++;
            inputSecondTemp++;
            inputi8SecondTemp++;
        }
    }

    // Run case-wise RPP API and measure time
    rppHandle_t handle;
    rppCreateWithBatchSize(&handle, noOfImages);

    double maxWallTime = 0, minWallTime = 500, avgWallTime = 0;
    double cpuTime, wallTime;
    string testCaseName;

    // case-wise RPP API and measure time script for Unit and Performance test
    printf("\nRunning %s %d times (each time with a batch size of %d images) and computing mean statistics...", func.c_str(), numIterations, noOfImages);
    for (int perfRunCount = 0; perfRunCount < numIterations; perfRunCount++)
    {
        clock_t startCpuTime, endCpuTime;
        double startWallTime, endWallTime;
        switch (testCase)
        {
            case 0:
            {
                testCaseName = "brightness";

                Rpp32f alpha[images];
                Rpp32f beta[images];
                for (i = 0; i < images; i++)
                {
                    alpha[i] = 1.75;
                    beta[i] = 50;
                }

                // Uncomment to run test case with an xywhROI override
                // roi.xywhROI = {0, 0, 100, 180};
                // set_roi_values(&roi, roiTensorPtrSrc, roiTypeSrc, images);
                // update_dst_sizes_with_roi(roiTensorPtrSrc, dstImgSizes, roiTypeSrc, images);

                // Uncomment to run test case with an ltrbROI override
                // roiTypeSrc = RpptRoiType::LTRB;
                // roi.ltrbROI = {50, 30, 210, 210};
                // set_roi_values(&roi, roiTensorPtrSrc, roiTypeSrc, images);
                // update_dst_sizes_with_roi(roiTensorPtrSrc, dstImgSizes, roiTypeSrc, images);

                startWallTime = omp_get_wtime();
                startCpuTime = clock();
                if (inputBitDepth == 0 || inputBitDepth == 1 || inputBitDepth == 2 || inputBitDepth == 5)
                    rppt_brightness_host(input, srcDescPtr, output, dstDescPtr, alpha, beta, roiTensorPtrSrc, roiTypeSrc, handle);
                else
                    missingFuncFlag = 1;

                break;
            }
            case 1:
            {
                testCaseName = "gamma_correction";

                Rpp32f gammaVal[images];
                for (i = 0; i < images; i++)
                {
                    gammaVal[i] = 1.9;
                }

                // Uncomment to run test case with an xywhROI override
                // roi.xywhROI = {0, 0, 100, 180};
                // set_roi_values(&roi, roiTensorPtrSrc, roiTypeSrc, images);
                // update_dst_sizes_with_roi(roiTensorPtrSrc, dstImgSizes, roiTypeSrc, images);

                // Uncomment to run test case with an ltrbROI override
                // roiTypeSrc = RpptRoiType::LTRB;
                // roi.ltrbROI = {50, 30, 210, 210};
                // set_roi_values(&roi, roiTensorPtrSrc, roiTypeSrc, images);
                // update_dst_sizes_with_roi(roiTensorPtrSrc, dstImgSizes, roiTypeSrc, images);

                startWallTime = omp_get_wtime();
                startCpuTime = clock();
                if (inputBitDepth == 0 || inputBitDepth == 1 || inputBitDepth == 2 || inputBitDepth == 5)
                    rppt_gamma_correction_host(input, srcDescPtr, output, dstDescPtr, gammaVal, roiTensorPtrSrc, roiTypeSrc, handle);
                else
                    missingFuncFlag = 1;

                break;
            }
            case 2:
            {
                testCaseName = "blend";

                Rpp32f alpha[images];
                for (i = 0; i < images; i++)
                {
                    alpha[i] = 0.4;
                }

                // Uncomment to run test case with an xywhROI override
                // roi.xywhROI = {0, 0, 100, 180};
                // set_roi_values(&roi, roiTensorPtrSrc, roiTypeSrc, images);
                // update_dst_sizes_with_roi(roiTensorPtrSrc, dstImgSizes, roiTypeSrc, images);

                // Uncomment to run test case with an ltrbROI override
                // roiTypeSrc = RpptRoiType::LTRB;
                // roi.ltrbROI = {50, 30, 210, 210};
                // set_roi_values(&roi, roiTensorPtrSrc, roiTypeSrc, images);
                // update_dst_sizes_with_roi(roiTensorPtrSrc, dstImgSizes, roiTypeSrc, images);

                startWallTime = omp_get_wtime();
                startCpuTime = clock();
                if (inputBitDepth == 0 || inputBitDepth == 1 || inputBitDepth == 2 || inputBitDepth == 5)
                    rppt_blend_host(input, input_second, srcDescPtr, output, dstDescPtr, alpha, roiTensorPtrSrc, roiTypeSrc, handle);
                else
                    missingFuncFlag = 1;

                break;
            }
            case 4:
            {
                testCaseName = "contrast";

                Rpp32f contrastFactor[images];
                Rpp32f contrastCenter[images];
                for (i = 0; i < images; i++)
                {
                    contrastFactor[i] = 2.96;
                    contrastCenter[i] = 128;
                }

                // Uncomment to run test case with an xywhROI override
                // roi.xywhROI = {0, 0, 100, 180};
                // set_roi_values(&roi, roiTensorPtrSrc, roiTypeSrc, images);
                // update_dst_sizes_with_roi(roiTensorPtrSrc, dstImgSizes, roiTypeSrc, images);

                // Uncomment to run test case with an ltrbROI override
                // roiTypeSrc = RpptRoiType::LTRB;
                // roi.ltrbROI = {50, 30, 210, 210};
                // set_roi_values(&roi, roiTensorPtrSrc, roiTypeSrc, images);
                // update_dst_sizes_with_roi(roiTensorPtrSrc, dstImgSizes, roiTypeSrc, images);

                startWallTime = omp_get_wtime();
                startCpuTime = clock();
                if (inputBitDepth == 0 || inputBitDepth == 1 || inputBitDepth == 2 || inputBitDepth == 5)
                    rppt_contrast_host(input, srcDescPtr, output, dstDescPtr, contrastFactor, contrastCenter, roiTensorPtrSrc, roiTypeSrc, handle);
                else
                    missingFuncFlag = 1;

                break;
            }
            case 8:
            {
                testCaseName = "noise";

                switch(additionalParam)
                {
                    case 0:
                    {
                        Rpp32f noiseProbabilityTensor[images];
                        Rpp32f saltProbabilityTensor[images];
                        Rpp32f saltValueTensor[images];
                        Rpp32f pepperValueTensor[images];
                        Rpp32u seed = 1255459;
                        for (i = 0; i < images; i++)
                        {
                            noiseProbabilityTensor[i] = 0.1f;
                            saltProbabilityTensor[i] = 0.5f;
                            saltValueTensor[i] = 1.0f;
                            pepperValueTensor[i] = 0.0f;
                        }

                        // Uncomment to run test case with an xywhROI override
                        // roi.xywhROI = {0, 0, 100, 180};
                        // set_roi_values(&roi, roiTensorPtrSrc, roiTypeSrc, images);
                        // update_dst_sizes_with_roi(roiTensorPtrSrc, dstImgSizes, roiTypeSrc, images);

                        // Uncomment to run test case with an ltrbROI override
                        // roiTypeSrc = RpptRoiType::LTRB;
                        // roi.ltrbROI = {50, 30, 210, 210};
                        // set_roi_values(&roi, roiTensorPtrSrc, roiTypeSrc, images);
                        // update_dst_sizes_with_roi(roiTensorPtrSrc, dstImgSizes, roiTypeSrc, images);

                        startWallTime = omp_get_wtime();
                        startCpuTime = clock();
                        if (inputBitDepth == 0 || inputBitDepth == 1 || inputBitDepth == 2 || inputBitDepth == 5)
                            rppt_salt_and_pepper_noise_host(input, srcDescPtr, output, dstDescPtr, noiseProbabilityTensor, saltProbabilityTensor, saltValueTensor, pepperValueTensor, seed, roiTensorPtrSrc, roiTypeSrc, handle);
                        else
                            missingFuncFlag = 1;

                        break;
                    }
                    case 1:
                    {
                        Rpp32f meanTensor[images];
                        Rpp32f stdDevTensor[images];
                        Rpp32u seed = 1255459;
                        for (i = 0; i < images; i++)
                        {
                            meanTensor[i] = 0.0f;
                            stdDevTensor[i] = 0.2f;
                        }

                        // Uncomment to run test case with an xywhROI override
                        // roi.xywhROI = {0, 0, 100, 180};
                        // set_roi_values(&roi, roiTensorPtrSrc, roiTypeSrc, images);
                        // update_dst_sizes_with_roi(roiTensorPtrSrc, dstImgSizes, roiTypeSrc, images);

                        // Uncomment to run test case with an ltrbROI override
                        // roiTypeSrc = RpptRoiType::LTRB;
                        // roi.ltrbROI = {50, 30, 210, 210};
                        // set_roi_values(&roi, roiTensorPtrSrc, roiTypeSrc, images);
                        // update_dst_sizes_with_roi(roiTensorPtrSrc, dstImgSizes, roiTypeSrc, images);

                        startWallTime = omp_get_wtime();
                        startCpuTime = clock();
                        if (inputBitDepth == 0 || inputBitDepth == 1 || inputBitDepth == 2 || inputBitDepth == 5)
                            rppt_gaussian_noise_host(input, srcDescPtr, output, dstDescPtr, meanTensor, stdDevTensor, seed, roiTensorPtrSrc, roiTypeSrc, handle);
                        else
                            missingFuncFlag = 1;

                        break;
                    }
                    case 2:
                    {
                        Rpp32f shotNoiseFactorTensor[images];
                        Rpp32u seed = 1255459;
                        for (i = 0; i < images; i++)
                        {
                            shotNoiseFactorTensor[i] = 80.0f;
                        }

                        // Uncomment to run test case with an xywhROI override
                        // roi.xywhROI = {0, 0, 100, 180};
                        // set_roi_values(&roi, roiTensorPtrSrc, roiTypeSrc, images);
                        // update_dst_sizes_with_roi(roiTensorPtrSrc, dstImgSizes, roiTypeSrc, images);

                        // Uncomment to run test case with an ltrbROI override
                        // roiTypeSrc = RpptRoiType::LTRB;
                        // roi.ltrbROI = {50, 30, 210, 210};
                        // set_roi_values(&roi, roiTensorPtrSrc, roiTypeSrc, images);
                        // update_dst_sizes_with_roi(roiTensorPtrSrc, dstImgSizes, roiTypeSrc, images);

                        startWallTime = omp_get_wtime();
                        startCpuTime = clock();
                        if (inputBitDepth == 0 || inputBitDepth == 1 || inputBitDepth == 2 || inputBitDepth == 5)
                            rppt_shot_noise_host(input, srcDescPtr, output, dstDescPtr, shotNoiseFactorTensor, seed, roiTensorPtrSrc, roiTypeSrc, handle);
                        else
                            missingFuncFlag = 1;

                        break;
                    }
                    default:
                    {
                        missingFuncFlag = 1;
                        break;
                    }
                }

                break;
            }
            case 13:
            {
                testCaseName = "exposure";

                Rpp32f exposureFactor[images];
                for (i = 0; i < images; i++)
                {
                    exposureFactor[i] = 1.4;
                }

                // Uncomment to run test case with an xywhROI override
                // roi.xywhROI = {0, 0, 100, 180};
                // set_roi_values(&roi, roiTensorPtrSrc, roiTypeSrc, images);
                // update_dst_sizes_with_roi(roiTensorPtrSrc, dstImgSizes, roiTypeSrc, images);

                // Uncomment to run test case with an ltrbROI override
                // roiTypeSrc = RpptRoiType::LTRB;
                // roi.ltrbROI = {50, 30, 210, 210};
                // set_roi_values(&roi, roiTensorPtrSrc, roiTypeSrc, images);
                // update_dst_sizes_with_roi(roiTensorPtrSrc, dstImgSizes, roiTypeSrc, images);

                startWallTime = omp_get_wtime();
                startCpuTime = clock();
                if (inputBitDepth == 0 || inputBitDepth == 1 || inputBitDepth == 2 || inputBitDepth == 5)
                    rppt_exposure_host(input, srcDescPtr, output, dstDescPtr, exposureFactor, roiTensorPtrSrc, roiTypeSrc, handle);
                else
                    missingFuncFlag = 1;

                break;
            }
            case 20:
            {
                testCaseName = "flip";

                Rpp32u horizontalFlag[images];
                Rpp32u verticalFlag[images];
                for (i = 0; i < images; i++)
                {
                    horizontalFlag[i] = 1;
                    verticalFlag[i] = 0;
                }

                startWallTime = omp_get_wtime();
                startCpuTime = clock();
                if (inputBitDepth == 0 || inputBitDepth == 1 || inputBitDepth == 2 || inputBitDepth == 5)
                    rppt_flip_host(input, srcDescPtr, output, dstDescPtr, horizontalFlag, verticalFlag, roiTensorPtrSrc, roiTypeSrc, handle);
                else
                    missingFuncFlag = 1;

                break;
            }
            case 21:
            {
                testCaseName = "resize";

                for (i = 0; i < images; i++)
                {
                    dstImgSizes[i].width = roiTensorPtrDst[i].xywhROI.roiWidth = roiTensorPtrSrc[i].xywhROI.roiWidth / 1.1;
                    dstImgSizes[i].height = roiTensorPtrDst[i].xywhROI.roiHeight = roiTensorPtrSrc[i].xywhROI.roiHeight / 3;
                }

                // Uncomment to run test case with an xywhROI override
                // roi.xywhROI = {0, 0, 100, 180};
                // set_roi_values(&roi, roiTensorPtrSrc, roiTypeSrc, images);
                // update_dst_sizes_with_roi(roiTensorPtrSrc, dstImgSizes, roiTypeSrc, images);

                // Uncomment to run test case with an ltrbROI override
                // roiTypeSrc = RpptRoiType::LTRB;
                // roi.ltrbROI = {50, 30, 210, 210};
                // set_roi_values(&roi, roiTensorPtrSrc, roiTypeSrc, images);
                // update_dst_sizes_with_roi(roiTensorPtrSrc, dstImgSizes, roiTypeSrc, images);

                startWallTime = omp_get_wtime();
                startCpuTime = clock();
                if (inputBitDepth == 0 || inputBitDepth == 1 || inputBitDepth == 2 || inputBitDepth == 5)
                    rppt_resize_host(input, srcDescPtr, output, dstDescPtr, dstImgSizes, interpolationType, roiTensorPtrSrc, roiTypeSrc, handle);
                else
                    missingFuncFlag = 1;

                break;
            }
            case 24:
            {
                testCaseName = "warp_affine";

                if ((interpolationType != RpptInterpolationType::BILINEAR) && (interpolationType != RpptInterpolationType::NEAREST_NEIGHBOR))
                {
                    missingFuncFlag = 1;
                    break;
                }

                Rpp32f6 affineTensor_f6[images];
                Rpp32f *affineTensor = (Rpp32f *)affineTensor_f6;
                for (i = 0; i < images; i++)
                {
                    affineTensor_f6[i].data[0] = 1.23;
                    affineTensor_f6[i].data[1] = 0.5;
                    affineTensor_f6[i].data[2] = 0;
                    affineTensor_f6[i].data[3] = -0.8;
                    affineTensor_f6[i].data[4] = 0.83;
                    affineTensor_f6[i].data[5] = 0;
                }

                // Uncomment to run test case with an xywhROI override
                // roi.xywhROI = {0, 0, 100, 180};
                // set_roi_values(&roi, roiTensorPtrSrc, roiTypeSrc, images);
                // update_dst_sizes_with_roi(roiTensorPtrSrc, dstImgSizes, roiTypeSrc, images);

                // Uncomment to run test case with an ltrbROI override
                // roiTypeSrc = RpptRoiType::LTRB;
                // roi.ltrbROI = {50, 30, 210, 210};
                // set_roi_values(&roi, roiTensorPtrSrc, roiTypeSrc, images);
                // update_dst_sizes_with_roi(roiTensorPtrSrc, dstImgSizes, roiTypeSrc, images);

                startWallTime = omp_get_wtime();
                startCpuTime = clock();
                if (inputBitDepth == 0 || inputBitDepth == 1 || inputBitDepth == 2 || inputBitDepth == 5)
                    rppt_warp_affine_host(input, srcDescPtr, output, dstDescPtr, affineTensor, interpolationType, roiTensorPtrSrc, roiTypeSrc, handle);
                else
                    missingFuncFlag = 1;

                break;
            }
            case 30:
            {
                testCaseName = "non_linear_blend";

                Rpp32f stdDev[images];
                for (i = 0; i < images; i++)
                {
                    stdDev[i] = 50.0;
                }

                // Uncomment to run test case with an xywhROI override
                // roi.xywhROI = {0, 0, 100, 180};
                // set_roi_values(&roi, roiTensorPtrSrc, roiTypeSrc, images);
                // update_dst_sizes_with_roi(roiTensorPtrSrc, dstImgSizes, roiTypeSrc, images);

                // Uncomment to run test case with an ltrbROI override
                // roiTypeSrc = RpptRoiType::LTRB;
                // roi.ltrbROI = {50, 30, 210, 210};
                // set_roi_values(&roi, roiTensorPtrSrc, roiTypeSrc, images);
                // update_dst_sizes_with_roi(roiTensorPtrSrc, dstImgSizes, roiTypeSrc, images);

                startWallTime = omp_get_wtime();
                startCpuTime = clock();
                if (inputBitDepth == 0 || inputBitDepth == 1 || inputBitDepth == 2 || inputBitDepth == 5)
                    rppt_non_linear_blend_host(input, input_second, srcDescPtr, output, dstDescPtr, stdDev, roiTensorPtrSrc, roiTypeSrc, handle);
                else
                    missingFuncFlag = 1;

                break;
            }
            case 31:
            {
                testCaseName = "color_cast";

                RpptRGB rgbTensor[images];
                Rpp32f alphaTensor[images];
                for (i = 0; i < images; i++)
                {
                    rgbTensor[i].R = 0;
                    rgbTensor[i].G = 0;
                    rgbTensor[i].B = 100;
                    alphaTensor[i] = 0.5;
                }

                // Uncomment to run test case with an xywhROI override
                // roi.xywhROI = {0, 0, 100, 180};
                // set_roi_values(&roi, roiTensorPtrSrc, roiTypeSrc, images);
                // update_dst_sizes_with_roi(roiTensorPtrSrc, dstImgSizes, roiTypeSrc, images);

                // Uncomment to run test case with an ltrbROI override
                // roiTypeSrc = RpptRoiType::LTRB;
                // roi.ltrbROI = {50, 30, 210, 210};
                // set_roi_values(&roi, roiTensorPtrSrc, roiTypeSrc, images);
                // update_dst_sizes_with_roi(roiTensorPtrSrc, dstImgSizes, roiTypeSrc, images);

                startWallTime = omp_get_wtime();
                startCpuTime = clock();
                if (inputBitDepth == 0 || inputBitDepth == 1 || inputBitDepth == 2 || inputBitDepth == 5)
                    rppt_color_cast_host(input, srcDescPtr, output, dstDescPtr, rgbTensor, alphaTensor, roiTensorPtrSrc, roiTypeSrc, handle);
                else
                    missingFuncFlag = 1;

                break;
            }
            case 36:
            {
                testCaseName = "color_twist";

                Rpp32f brightness[images];
                Rpp32f contrast[images];
                Rpp32f hue[images];
                Rpp32f saturation[images];
                for (i = 0; i < images; i++)
                {
                    brightness[i] = 1.4;
                    contrast[i] = 0.0;
                    hue[i] = 60.0;
                    saturation[i] = 1.9;
                }

                // Uncomment to run test case with an xywhROI override
                // roi.xywhROI = {0, 0, 100, 180};
                // set_roi_values(&roi, roiTensorPtrSrc, roiTypeSrc, images);
                // update_dst_sizes_with_roi(roiTensorPtrSrc, dstImgSizes, roiTypeSrc, images);

                // Uncomment to run test case with an ltrbROI override
                // roiTypeSrc = RpptRoiType::LTRB;
                // roi.ltrbROI = {50, 30, 210, 210};
                // set_roi_values(&roi, roiTensorPtrSrc, roiTypeSrc, images);
                // update_dst_sizes_with_roi(roiTensorPtrSrc, dstImgSizes, roiTypeSrc, images);

                startWallTime = omp_get_wtime();
                startCpuTime = clock();
                if (inputBitDepth == 0 || inputBitDepth == 1 || inputBitDepth == 2 || inputBitDepth == 5)
                    rppt_color_twist_host(input, srcDescPtr, output, dstDescPtr, brightness, contrast, hue, saturation, roiTensorPtrSrc, roiTypeSrc, handle);
                else
                    missingFuncFlag = 1;

                break;
            }
            case 37:
            {
                testCaseName = "crop";

                // Uncomment to run test case with an xywhROI override
                for (i = 0; i < images; i++)
                {
                    roiTensorPtrSrc[i].xywhROI.xy.x = 0;
                    roiTensorPtrSrc[i].xywhROI.xy.y = 0;
                    dstImgSizes[i].width = roiTensorPtrSrc[i].xywhROI.roiWidth = 150;
                    dstImgSizes[i].height = roiTensorPtrSrc[i].xywhROI.roiHeight = 150;
                }

                // Uncomment to run test case with an xywhROI override
                // roi.xywhROI = {0, 0, 100, 180};
                // set_roi_values(&roi, roiTensorPtrSrc, roiTypeSrc, images);
                // update_dst_sizes_with_roi(roiTensorPtrSrc, dstImgSizes, roiTypeSrc, images);

                // Uncomment to run test case with an ltrbROI override
                // roiTypeSrc = RpptRoiType::LTRB;
                // roi.ltrbROI = {50, 30, 210, 210};
                // set_roi_values(&roi, roiTensorPtrSrc, roiTypeSrc, images);
                // update_dst_sizes_with_roi(roiTensorPtrSrc, dstImgSizes, roiTypeSrc, images);

                startWallTime = omp_get_wtime();
                startCpuTime = clock();
                if (inputBitDepth == 0 || inputBitDepth == 1 || inputBitDepth == 2 || inputBitDepth == 5)
                    rppt_crop_host(input, srcDescPtr, output, dstDescPtr, roiTensorPtrSrc, roiTypeSrc, handle);
                else
                    missingFuncFlag = 1;

                break;
            }
            case 38:
            {
                testCaseName = "crop_mirror_normalize";
                Rpp32f multiplier[images * srcDescPtr->c];
                Rpp32f offset[images * srcDescPtr->c];
                Rpp32u mirror[images];
                if (srcDescPtr->c == 3)
                {
                    Rpp32f meanParam[3] = { 60.0f, 80.0f, 100.0f };
                    Rpp32f stdDevParam[3] = { 0.9f, 0.9f, 0.9f };
                    Rpp32f offsetParam[3] = { - meanParam[0] / stdDevParam[0], - meanParam[1] / stdDevParam[1], - meanParam[2] / stdDevParam[2] };
                    Rpp32f multiplierParam[3] = {  1.0f / stdDevParam[0], 1.0f / stdDevParam[1], 1.0f / stdDevParam[2] };

                    for (i = 0, j = 0; i < images; i++, j += 3)
                    {
                        multiplier[j] = multiplierParam[0];
                        offset[j] = offsetParam[0];
                        multiplier[j + 1] = multiplierParam[1];
                        offset[j + 1] = offsetParam[1];
                        multiplier[j + 2] = multiplierParam[2];
                        offset[j + 2] = offsetParam[2];
                        mirror[i] = 1;
                    }
                }
                else if(srcDescPtr->c == 1)
                {
                    Rpp32f meanParam = 100.0f;
                    Rpp32f stdDevParam = 0.9f;
                    Rpp32f offsetParam = - meanParam / stdDevParam;
                    Rpp32f multiplierParam = 1.0f / stdDevParam;

                    for (i = 0; i < images; i++)
                    {
                        multiplier[i] = multiplierParam;
                        offset[i] = offsetParam;
                        mirror[i] = 1;
                    }
                }

                // Uncomment to run test case with an xywhROI override
                for (i = 0; i < images; i++)
                {
                    roiTensorPtrSrc[i].xywhROI.xy.x = 10;
                    roiTensorPtrSrc[i].xywhROI.xy.y = 10;
                    dstImgSizes[i].width = roiTensorPtrSrc[i].xywhROI.roiWidth = roiTensorPtrSrc[i].xywhROI.roiWidth / 2;
                    dstImgSizes[i].height = roiTensorPtrSrc[i].xywhROI.roiHeight = roiTensorPtrSrc[i].xywhROI.roiHeight / 2;
                }

                // Uncomment to run test case with an xywhROI override
                // roi.xywhROI = {0, 0, 100, 180};
                // set_roi_values(&roi, roiTensorPtrSrc, roiTypeSrc, images);
                // update_dst_sizes_with_roi(roiTensorPtrSrc, dstImgSizes, roiTypeSrc, images);

                // Uncomment to run test case with an ltrbROI override
                // roiTypeSrc = RpptRoiType::LTRB;
                // roi.ltrbROI = {50, 30, 210, 210};
                // set_roi_values(&roi, roiTensorPtrSrc, roiTypeSrc, images);
                // update_dst_sizes_with_roi(roiTensorPtrSrc, dstImgSizes, roiTypeSrc, images);

                startWallTime = omp_get_wtime();
                startCpuTime = clock();
                if (inputBitDepth == 0 || inputBitDepth == 1 || inputBitDepth == 2 || inputBitDepth == 3 || inputBitDepth == 4 || inputBitDepth == 5)
                    rppt_crop_mirror_normalize_host(input, srcDescPtr, output, dstDescPtr, offset, multiplier, mirror, roiTensorPtrSrc, roiTypeSrc, handle);
                else
                    missingFuncFlag = 1;

                break;
            }
            case 39:
            {
                testCaseName = "resize_crop_mirror";

                if (interpolationType != RpptInterpolationType::BILINEAR)
                {
                    missingFuncFlag = 1;
                    break;
                }

                Rpp32u mirror[images];
                for (i = 0; i < images; i++)
                {
                    mirror[i] = 1;
                }

                for (i = 0; i < images; i++)
                {
                    roiTensorPtrSrc[i].xywhROI.xy.x = 0;
                    roiTensorPtrSrc[i].xywhROI.xy.y = 0;
                    dstImgSizes[i].width = roiTensorPtrSrc[i].xywhROI.roiWidth / 1.1;
                    dstImgSizes[i].height = roiTensorPtrSrc[i].xywhROI.roiHeight / 3;
                    roiTensorPtrSrc[i].xywhROI.roiWidth = 50;
                    roiTensorPtrSrc[i].xywhROI.roiHeight = 50;
                }

                // Uncomment to run test case with an xywhROI override
                // roi.xywhROI = {0, 0, 100, 180};
                // set_roi_values(&roi, roiTensorPtrSrc, roiTypeSrc, images);
                // update_dst_sizes_with_roi(roiTensorPtrSrc, dstImgSizes, roiTypeSrc, images);

                // Uncomment to run test case with an ltrbROI override
                // roiTypeSrc = RpptRoiType::LTRB;
                // roi.ltrbROI = {50, 30, 210, 210};
                // set_roi_values(&roi, roiTensorPtrSrc, roiTypeSrc, images);
                // update_dst_sizes_with_roi(roiTensorPtrSrc, dstImgSizes, roiTypeSrc, images);

                startWallTime = omp_get_wtime();
                startCpuTime = clock();
                if (inputBitDepth == 0 || inputBitDepth == 1 || inputBitDepth == 2 || inputBitDepth == 5)
                    rppt_resize_crop_mirror_host(input, srcDescPtr, output, dstDescPtr, dstImgSizes, interpolationType, mirror, roiTensorPtrSrc, roiTypeSrc, handle);
                else
                    missingFuncFlag = 1;

                break;
            }
            case 70:
            {
                testCaseName = "copy";

                startWallTime = omp_get_wtime();
                startCpuTime = clock();
                if (inputBitDepth == 0 || inputBitDepth == 1 || inputBitDepth == 2 || inputBitDepth == 5)
                    rppt_copy_host(input, srcDescPtr, output, dstDescPtr, handle);
                else
                    missingFuncFlag = 1;

                break;
            }
            case 80:
            {
                testCaseName = "resize_mirror_normalize";

                if (interpolationType != RpptInterpolationType::BILINEAR)
                {
                    missingFuncFlag = 1;
                    break;
                }

                for (i = 0; i < images; i++)
                {
                    dstImgSizes[i].width = roiTensorPtrDst[i].xywhROI.roiWidth = roiTensorPtrSrc[i].xywhROI.roiWidth / 1.1;
                    dstImgSizes[i].height = roiTensorPtrDst[i].xywhROI.roiHeight = roiTensorPtrSrc[i].xywhROI.roiHeight / 3;
                }

                Rpp32f mean[images * srcDescPtr->c];
                Rpp32f stdDev[images * srcDescPtr->c];
                Rpp32u mirror[images];
                if(srcDescPtr->c == 3)
                {
                    for (i = 0, j = 0; i < images; i++, j += 3)
                    {
                        mean[j] = 60.0;
                        stdDev[j] = 1.0;
                        mean[j + 1] = 80.0;
                        stdDev[j + 1] = 1.0;
                        mean[j + 2] = 100.0;
                        stdDev[j + 2] = 1.0;
                        mirror[i] = 1;
                    }
                }
                else
                {
                    for (i = 0; i < images; i++)
                    {
                        mean[i] = 100.0;
                        stdDev[i] = 1.0;
                        mirror[i] = 1;
                    }
                }

                // Uncomment to run test case with an xywhROI override
                // roi.xywhROI = {0, 0, 100, 180};
                // set_roi_values(&roi, roiTensorPtrSrc, roiTypeSrc, images);
                // update_dst_sizes_with_roi(roiTensorPtrSrc, dstImgSizes, roiTypeSrc, images);

                // Uncomment to run test case with an ltrbROI override
                // roiTypeSrc = RpptRoiType::LTRB;
                // roi.ltrbROI = {50, 30, 210, 210};
                // set_roi_values(&roi, roiTensorPtrSrc, roiTypeSrc, images);
                // update_dst_sizes_with_roi(roiTensorPtrSrc, dstImgSizes, roiTypeSrc, images);

                startWallTime = omp_get_wtime();
                startCpuTime = clock();
                if (inputBitDepth == 0 || inputBitDepth == 1 || inputBitDepth == 2 || inputBitDepth == 3 || inputBitDepth == 4 || inputBitDepth == 5)
                    rppt_resize_mirror_normalize_host(input, srcDescPtr, output, dstDescPtr, dstImgSizes, interpolationType, mean, stdDev, mirror, roiTensorPtrSrc, roiTypeSrc, handle);
                else
                    missingFuncFlag = 1;

                break;
            }
            case 81:
            {
                testCaseName = "color_jitter";

                Rpp32f brightness[images];
                Rpp32f contrast[images];
                Rpp32f hue[images];
                Rpp32f saturation[images];
                for (i = 0; i < images; i++)
                {
                    brightness[i] = 1.02;
                    contrast[i] = 1.1;
                    hue[i] = 0.02;
                    saturation[i] = 1.3;
                }

                // Uncomment to run test case with an xywhROI override
                // roi.xywhROI = {0, 0, 100, 180};
                // set_roi_values(&roi, roiTensorPtrSrc, roiTypeSrc, images);
                // update_dst_sizes_with_roi(roiTensorPtrSrc, dstImgSizes, roiTypeSrc, images);

                // Uncomment to run test case with an ltrbROI override
                // roiTypeSrc = RpptRoiType::LTRB;
                // roi.ltrbROI = {50, 30, 210, 210};
                // set_roi_values(&roi, roiTensorPtrSrc, roiTypeSrc, images);
                // update_dst_sizes_with_roi(roiTensorPtrSrc, dstImgSizes, roiTypeSrc, images);

                startWallTime = omp_get_wtime();
                startCpuTime = clock();
                if (inputBitDepth == 0 || inputBitDepth == 1 || inputBitDepth == 2 || inputBitDepth == 5)
                    rppt_color_jitter_host(input, srcDescPtr, output, dstDescPtr, brightness, contrast, hue, saturation, roiTensorPtrSrc, roiTypeSrc, handle);
                else
                    missingFuncFlag = 1;

                break;
            }
            case 83:
            {
                testCaseName = "gridmask";

                Rpp32u tileWidth = 40;
                Rpp32f gridRatio = 0.6;
                Rpp32f gridAngle = 0.5;
                RpptUintVector2D translateVector;
                translateVector.x = 0.0;
                translateVector.y = 0.0;

                // Uncomment to run test case with an xywhROI override
                // roi.xywhROI = {0, 0, 100, 180};
                // set_roi_values(&roi, roiTensorPtrSrc, roiTypeSrc, images);
                // update_dst_sizes_with_roi(roiTensorPtrSrc, dstImgSizes, roiTypeSrc, images);

                // Uncomment to run test case with an ltrbROI override
                // roiTypeSrc = RpptRoiType::LTRB;
                // roi.ltrbROI = {50, 30, 210, 210};
                // set_roi_values(&roi, roiTensorPtrSrc, roiTypeSrc, images);
                // update_dst_sizes_with_roi(roiTensorPtrSrc, dstImgSizes, roiTypeSrc, images);

                startWallTime = omp_get_wtime();
                startCpuTime = clock();
                if (inputBitDepth == 0 || inputBitDepth == 1 || inputBitDepth == 2 || inputBitDepth == 5)
                    rppt_gridmask_host(input, srcDescPtr, output, dstDescPtr, tileWidth, gridRatio, gridAngle, translateVector, roiTensorPtrSrc, roiTypeSrc, handle);
                else
                    missingFuncFlag = 1;

                break;
            }
            case 84:
            {
                testCaseName = "spatter";

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
                // roi.xywhROI = {0, 0, 100, 180};
                // set_roi_values(&roi, roiTensorPtrSrc, roiTypeSrc, images);
                // update_dst_sizes_with_roi(roiTensorPtrSrc, dstImgSizes, roiTypeSrc, images);

                // Uncomment to run test case with an ltrbROI override
                // roiTypeSrc = RpptRoiType::LTRB;
                // roi.ltrbROI = {50, 30, 210, 210};
                // set_roi_values(&roi, roiTensorPtrSrc, roiTypeSrc, images);
                // update_dst_sizes_with_roi(roiTensorPtrSrc, dstImgSizes, roiTypeSrc, images);

                startWallTime = omp_get_wtime();
                startCpuTime = clock();
                if (inputBitDepth == 0 || inputBitDepth == 1 || inputBitDepth == 2 || inputBitDepth == 5)
                    rppt_spatter_host(input, srcDescPtr, output, dstDescPtr, spatterColor, roiTensorPtrSrc, roiTypeSrc, handle);
                else
                    missingFuncFlag = 1;

                break;
            }
            case 85:
            {
                testCaseName = "swap_channels";

                startWallTime = omp_get_wtime();
                startCpuTime = clock();
                if (inputBitDepth == 0 || inputBitDepth == 1 || inputBitDepth == 2 || inputBitDepth == 5)
                    rppt_swap_channels_host(input, srcDescPtr, output, dstDescPtr, handle);
                else
                    missingFuncFlag = 1;

                break;
            }
            case 86:
            {
                testCaseName = "color_to_greyscale";

                RpptSubpixelLayout srcSubpixelLayout = RpptSubpixelLayout::RGBtype;

                startWallTime = omp_get_wtime();
                startCpuTime = clock();
                if (inputBitDepth == 0 || inputBitDepth == 1 || inputBitDepth == 2 || inputBitDepth == 5)
                    rppt_color_to_greyscale_host(input, srcDescPtr, output, dstDescPtr, srcSubpixelLayout, handle);
                else
                    missingFuncFlag = 1;

                break;
            }
            default:
                missingFuncFlag = 1;
                break;
        }

        endCpuTime = clock();
        endWallTime = omp_get_wtime();
        cpuTime = ((double)(endCpuTime - startCpuTime)) / CLOCKS_PER_SEC;
        wallTime = endWallTime - startWallTime;
        if (missingFuncFlag == 1)
        {
            printf("\nThe functionality %s doesn't yet exist in RPP\n", func.c_str());
            return -1;
        }

        maxWallTime = max(maxWallTime, wallTime);
        minWallTime = min(minWallTime, wallTime);
        avgWallTime += wallTime;
    }

    // converts units to milliseconds
    cpuTime *= 1000;
    wallTime *= 1000;
    maxWallTime *= 1000;
    minWallTime *= 1000;
    avgWallTime *= 1000;
    if (testType == 0)
    {
        cout <<"\n\n";
        cout <<"CPU Backend Clock Time: "<< cpuTime <<" ms"<< endl;
        cout <<"CPU Backend Wall Time: "<< wallTime <<" ms"<< endl;
    }
    else
    {
        // Display measured times
        avgWallTime /= numIterations;
        cout << fixed << "\n\nmax,min,avg in ms = " << maxWallTime << "," << minWallTime << "," << avgWallTime << endl;
    }

    if (testType == 0)
    {
       // Reconvert other bit depths to 8u for output display purposes
        if (inputBitDepth == 0)
        {
            memcpy(outputu8, output, outputBufferSize);
        }
        else if ((inputBitDepth == 1) || (inputBitDepth == 3))
        {
            Rpp8u *outputTemp = outputu8 + dstDescPtr->offsetInBytes;
            half *outputf16Temp = (half *)((Rpp8u *)output + dstDescPtr->offsetInBytes);
            for (int i = 0; i < oBufferSize; i++)
            {
                *outputTemp = (Rpp8u)validate_pixel_range((float)*outputf16Temp * invConversionFactor);
                outputf16Temp++;
                outputTemp++;
            }
        }
        else if ((inputBitDepth == 2) || (inputBitDepth == 4))
        {
            Rpp8u *outputTemp = outputu8 + dstDescPtr->offsetInBytes;
            Rpp32f *outputf32Temp = (Rpp32f *)((Rpp8u *)output + dstDescPtr->offsetInBytes);
            for (int i = 0; i < oBufferSize; i++)
            {
                *outputTemp = (Rpp8u)validate_pixel_range(*outputf32Temp * invConversionFactor);
                outputf32Temp++;
                outputTemp++;
            }
        }
        else if ((inputBitDepth == 5) || (inputBitDepth == 6))
        {
            Rpp8u *outputTemp = outputu8 + dstDescPtr->offsetInBytes;
            Rpp8s *outputi8Temp = (Rpp8s *)output + dstDescPtr->offsetInBytes;
            for (int i = 0; i < oBufferSize; i++)
            {
                *outputTemp = (Rpp8u) validate_pixel_range(((Rpp32s) *outputi8Temp) + 128);
                outputi8Temp++;
                outputTemp++;
            }
        }

        if(inputBitDepth == 0 && (srcDescPtr->layout == dstDescPtr->layout))
            compare_output<Rpp8u>(outputu8, testCaseName, srcDescPtr, dstDescPtr, roiTensorPtrDst, noOfImages);

        // Calculate exact dstROI in XYWH format for OpenCV dump
        if (roiTypeSrc == RpptRoiType::LTRB)
        {
            for (int i = 0; i < dstDescPtr->n; i++)
            {
                int ltX = roiTensorPtrSrc[i].ltrbROI.lt.x;
                int ltY = roiTensorPtrSrc[i].ltrbROI.lt.y;
                int rbX = roiTensorPtrSrc[i].ltrbROI.rb.x;
                int rbY = roiTensorPtrSrc[i].ltrbROI.rb.y;

                roiTensorPtrSrc[i].xywhROI.xy.x = ltX;
                roiTensorPtrSrc[i].xywhROI.xy.y = ltY;
                roiTensorPtrSrc[i].xywhROI.roiWidth = rbX - ltX + 1;
                roiTensorPtrSrc[i].xywhROI.roiHeight = rbY - ltY + 1;
            }
        }

        RpptROI roiDefault;
        RpptROIPtr roiPtrDefault;
        roiPtrDefault = &roiDefault;
        roiPtrDefault->xywhROI.xy.x = 0;
        roiPtrDefault->xywhROI.xy.y = 0;
        roiPtrDefault->xywhROI.roiWidth = dstDescPtr->w;
        roiPtrDefault->xywhROI.roiHeight = dstDescPtr->h;

        for (int i = 0; i < dstDescPtr->n; i++)
        {
            roiTensorPtrSrc[i].xywhROI.roiWidth = std::min(roiPtrDefault->xywhROI.roiWidth - roiTensorPtrSrc[i].xywhROI.xy.x, roiTensorPtrSrc[i].xywhROI.roiWidth);
            roiTensorPtrSrc[i].xywhROI.roiHeight = std::min(roiPtrDefault->xywhROI.roiHeight - roiTensorPtrSrc[i].xywhROI.xy.y, roiTensorPtrSrc[i].xywhROI.roiHeight);
            roiTensorPtrSrc[i].xywhROI.xy.x = std::max(roiPtrDefault->xywhROI.xy.x, roiTensorPtrSrc[i].xywhROI.xy.x);
            roiTensorPtrSrc[i].xywhROI.xy.y = std::max(roiPtrDefault->xywhROI.xy.y, roiTensorPtrSrc[i].xywhROI.xy.y);
        }

        // Convert any PLN3 outputs to the corresponding PKD3 version for OpenCV dump
        if (layoutType == 0 || layoutType == 1)
        {
            if ((dstDescPtr->c == 3) && (dstDescPtr->layout == RpptLayout::NCHW))
                convert_pln3_to_pkd3(outputu8, dstDescPtr);
        }
        rppDestroyHost(handle);

        // OpenCV dump (if testType is unit test)
        write_image_batch_opencv(dst, outputu8, dstDescPtr, imageNames, dstImgSizes);
    }

    // Free memory
    free(roiTensorPtrSrc);
    free(roiTensorPtrDst);
    free(input);
    free(input_second);
    free(output);
    free(inputu8);
    free(inputu8Second);
    free(outputu8);
    return 0;
}
