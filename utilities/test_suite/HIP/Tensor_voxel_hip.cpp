/*
Copyright (c) 2019 - 2023 Advanced Micro Devices, Inc. All rights reserved.

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

#include "../rpp_test_suite_voxel.h"

int main(int argc, char * argv[])
{
    int layoutType, testCase, testType, qaFlag, numRuns, batchSize;
    char *header_file, *data_file, *dst_path;

    if (argc < 7)
    {
        fprintf(stdout, "\nUsage: %s <header file> <data file> <layoutType = 0 - PKD3/ 1 - PLN3/ 2 - PLN1> <testCase = 0 to 1> <testType = 0 - unit test/ 1 - performance test>\n", argv[0]);
        exit(1);
    }

    header_file = argv[1];
    data_file = argv[2];
    dst_path = argv[3];
    layoutType = atoi(argv[4]); // 0 for PKD3 // 1 for PLN3 // 2 for PLN1
    testCase = atoi(argv[5]); // 0 to 1
    numRuns = atoi(argv[6]);
    testType = atoi(argv[7]); // 0 - unit test / 1 - performance test
    qaFlag = atoi(argv[8]); //0 - QA disabled / 1 - QA enabled
    batchSize = atoi(argv[9]);

    if ((layoutType < 0) || (layoutType > 2))
    {
        fprintf(stdout, "\nUsage: %s <header file> <data file> <layoutType = 0 - PKD3/ 1 - PLN3/ 2 - PLN1>\n", argv[0]);
        exit(1);
    }
    if ((testCase < 0) || (testCase > 4))
    {
        fprintf(stdout, "\nUsage: %s <header file> <data file> <layoutType = 0 for NCDHW / 1 for NDHWC>\n", argv[0]);
        exit(1);
    }

    if(batchSize > MAX_BATCH_SIZE)
    {
        std::cout << "\n Batchsize should be less than or equal to "<< MAX_BATCH_SIZE << " Aborting!";
        exit(0);
    }

    string func = augmentationMap[testCase];
    if (func.empty())
    {
        if (testType == 0)
            printf("\ncase %d is not supported\n", testCase);

        return -1;
    }

    int numChannels, offsetInBytes;
    int noOfFiles = 0, maxX = 0, maxY = 0, maxZ = 0;
    vector<string> headerNames, headerPath, dataFileNames, dataFilePath;
    search_nii_files(header_file, headerNames, headerPath);
    search_nii_files(data_file, dataFileNames, dataFilePath);
    noOfFiles = dataFileNames.size();
    if(noOfFiles < batchSize || ((noOfFiles % batchSize) != 0))
    {
        replicate_last_file_to_fill_batch(headerPath[noOfFiles - 1], headerPath, headerNames, headerNames[noOfFiles - 1], noOfFiles, batchSize);
        replicate_last_file_to_fill_batch(dataFilePath[noOfFiles - 1], dataFilePath, dataFileNames, dataFileNames[noOfFiles - 1], noOfFiles, batchSize);
        noOfFiles = dataFileNames.size();
    }

    // NIFTI_DATATYPE *niftiData = NULL;
    NIFTI_DATATYPE** niftiDataArray = (NIFTI_DATATYPE**)malloc(batchSize * sizeof(NIFTI_DATATYPE*));
    nifti_1_header* niftiHeader = (nifti_1_header*)malloc(noOfFiles * sizeof(nifti_1_header));

    // read nifti header file
    for(int i = 0; i < noOfFiles; i++)
    {
        read_nifti_header_file((char *)headerPath[i].c_str(), niftiHeader[i]);
        maxX = max(static_cast<int>(niftiHeader[i].dim[1]), maxX);
        maxY = max(static_cast<int>(niftiHeader[i].dim[2]), maxY);
        maxZ = max(static_cast<int>(niftiHeader[i].dim[3]), maxZ);
    }

    // Set ROI tensors types for src
    RpptRoi3DType roiTypeSrc;
    roiTypeSrc = RpptRoi3DType::XYZWHD;

    numChannels = (layoutType == 2) ? 1: 3;                    //Temporary value set to 3 for running pln3, the actual value should be obtained from niftiHeader.dim[4].
    offsetInBytes = 0;

    // optionally set maxX as a multiple of 8 for RPP optimal CPU/GPU processing
    maxX = ((maxX / 8) * 8) + 8;

    // set src/dst generic tensor descriptors
    RpptGenericDesc descriptor3D;
    RpptGenericDescPtr descriptorPtr3D = &descriptor3D;
    set_generic_descriptor(descriptorPtr3D, batchSize, maxX, maxY, maxZ, numChannels, offsetInBytes, layoutType);

    // set src/dst xyzwhd ROI tensors
    void *pinnedMemROI;
    hipHostMalloc(&pinnedMemROI, batchSize * sizeof(RpptRoiXyzwhd));
    RpptRoiXyzwhd *roiGenericSrcPtr = reinterpret_cast<RpptRoiXyzwhd *>(pinnedMemROI);

    // Set buffer sizes in pixels for src/dst
    Rpp64u iBufferSize = (Rpp64u)descriptorPtr3D->strides[0] * (Rpp64u)descriptorPtr3D->dims[0]; //  (d x h x w x c) x (n)
    Rpp64u oBufferSize = iBufferSize;   // User can provide a different oBufferSize

    // Set buffer sizes in bytes for src/dst (including offsets)
    Rpp64u iBufferSizeInBytes = iBufferSize * sizeof(Rpp32f) + descriptorPtr3D->offsetInBytes;
    Rpp64u oBufferSizeInBytes = iBufferSizeInBytes;

    // Allocate host memory in Rpp32f for RPP strided buffer
    Rpp32f *inputF32 = static_cast<Rpp32f *>(calloc(iBufferSizeInBytes, 1));
    Rpp32f *outputF32 = static_cast<Rpp32f *>(calloc(oBufferSizeInBytes, 1));

    // Allocate hip memory in float for RPP strided buffer
    void *d_inputF32, *d_outputF32;
    hipMalloc(&d_inputF32, iBufferSizeInBytes);
    hipMalloc(&d_outputF32, oBufferSizeInBytes);

    // set argument tensors
    void *pinnedMemArgs;
    pinnedMemArgs = calloc(2 * noOfFiles , sizeof(Rpp32f));

    // Set the number of threads to be used by OpenMP pragma for RPP batch processing on host.
    // If numThreads value passed is 0, number of OpenMP threads used by RPP will be set to batch size
    Rpp32u numThreads = 0;
    rppHandle_t handle;
    rppCreateWithBatchSize(&handle, noOfFiles, numThreads);

    // Run case-wise RPP API and measure time
    int missingFuncFlag = 0;
    double startWallTime, endWallTime, wallTime;
    double maxWallTime = 0, minWallTime = 5000, avgWallTime = 0;
    int noOfIterations = (int)noOfFiles / batchSize;
    string testCaseName;
    printf("\nRunning %s %d times (each time with a batch size of %d images) and computing mean statistics...", func.c_str(), numRuns, batchSize);
    for (int perfRunCount = 0; perfRunCount < numRuns; perfRunCount++)
    {
        for(int iterCount = 0; iterCount < noOfIterations; iterCount++)
        {
            vector<string>::const_iterator dataFilePathStart = dataFilePath.begin() + (iterCount * batchSize);
            vector<string>::const_iterator dataFilePathEnd = dataFilePathStart + batchSize;
            nifti_1_header *niftiHeaderTemp = niftiHeader + batchSize * iterCount;

            read_nifti_data(dataFilePathStart, dataFilePathEnd, niftiDataArray, niftiHeaderTemp);

            // optionally pick full image as ROI or a smaller slice of the 3D tensor in X/Y/Z dimensions
            for(int i = 0; i < batchSize; i++)
            {
                // option 1 - test using roi as the whole 3D image - not sliced (example for 240 x 240 x 155 x 1)
                roiGenericSrcPtr[i].xyz.x = 0;                                    // start X dim = 0
                roiGenericSrcPtr[i].xyz.y = 0;                                    // start Y dim = 0
                roiGenericSrcPtr[i].xyz.z = 0;                                    // start Z dim = 0
                roiGenericSrcPtr[i].roiWidth = niftiHeaderTemp[i].dim[1];         // length in X dim
                roiGenericSrcPtr[i].roiHeight = niftiHeaderTemp[i].dim[2];        // length in Y dim
                roiGenericSrcPtr[i].roiDepth = niftiHeaderTemp[i].dim[3];         // length in Z dim
                // option 2 - test using roi as a smaller 3D tensor slice - sliced in X, Y and Z dims (example for 240 x 240 x 155 x 1)
                // roiGenericSrcPtr[i].xyz.x = niftiHeader.dim[1] / 4;            // start X dim = 60
                // roiGenericSrcPtr[i].xyz.y = niftiHeader[i].dim[2] / 4;         // start Y dim = 60
                // roiGenericSrcPtr[i].xyz.z = niftiHeader[i].dim[3] / 3;         // start Z dim = 51
                // roiGenericSrcPtr[i].roiWidth = niftiHeader[i].dim[1] / 2;      // length in X dim = 120
                // roiGenericSrcPtr[i].roiHeight = niftiHeader[i].dim[2] / 2;     // length in Y dim = 120
                // roiGenericSrcPtr[i].roiDepth = niftiHeader[i].dim[3] / 3;      // length in Z dim = 51
                // option 3 - test using roi as a smaller 3D tensor slice - sliced in only Z dim (example for 240 x 240 x 155 x 1)
                // roiGenericSrcPtr[i].xyz.x = 0;                                 // start X dim = 0
                // roiGenericSrcPtr[i].xyz.y = 0;                                 // start Y dim = 0
                // roiGenericSrcPtr[i].xyz.z = niftiHeader[i].dim[3] / 3;         // start Z dim = 51
                // roiGenericSrcPtr[i].roiWidth = niftiHeader[i].dim[1];          // length in X dim = 240
                // roiGenericSrcPtr[i].roiHeight = niftiHeader[i].dim[2];         // length in Y dim = 240
                // roiGenericSrcPtr[i].roiDepth = niftiHeader[i].dim[3] / 3;      // length in Z dim = 51
                // option 4 - test using roi as a smaller 3D tensor slice - sliced in only X and Z dim (example for 240 x 240 x 155 x 1)
                // roiGenericSrcPtr[i].xyz.x = niftiHeader[i].dim[1] / 5;         // start X dim = 48
                // roiGenericSrcPtr[i].xyz.y = 0;                                 // start Y dim = 0
                // roiGenericSrcPtr[i].xyz.z = niftiHeader[i].dim[3] / 3;         // start Z dim = 51
                // roiGenericSrcPtr[i].roiWidth = niftiHeader[i].dim[1] * 3 / 5;  // length in X dim = 144
                // roiGenericSrcPtr[i].roiHeight = niftiHeader[i].dim[2];         // length in Y dim = 240
                // roiGenericSrcPtr[i].roiDepth = niftiHeader[i].dim[3] / 3;      // length in Z dim = 51
            }

            // Convert default NIFTI_DATATYPE unstrided buffer to RpptDataType::F32 strided buffer
            convert_input_niftitype_to_Rpp32f_generic(niftiDataArray, niftiHeaderTemp, inputF32 , descriptorPtr3D);

            //Copy input buffer to hip
            hipMemcpy(d_inputF32, inputF32, iBufferSizeInBytes, hipMemcpyHostToDevice);

            switch (testCase)
            {
                case 0:
                {
                    testCaseName = "fmadd";
                    Rpp32f *mulTensor = reinterpret_cast<Rpp32f *>(pinnedMemArgs);
                    Rpp32f *addTensor = mulTensor + batchSize;

                    for (int i = 0; i < batchSize; i++)
                    {
                        mulTensor[i] = 80;
                        addTensor[i] = 5;
                    }

                    startWallTime = omp_get_wtime();
                    rppt_fmadd_scalar_gpu(d_inputF32, descriptorPtr3D, d_outputF32, descriptorPtr3D, mulTensor, addTensor, roiGenericSrcPtr, handle);
                    break;
                }
                case 1:
                {
                    testCaseName = "slice";
                    startWallTime = omp_get_wtime();
                    rppt_slice_gpu(d_inputF32, descriptorPtr3D, d_outputF32, descriptorPtr3D, roiGenericSrcPtr, handle);
                    break;
                }
                default:
                {
                    missingFuncFlag = 1;
                    break;
                }
            }

            hipDeviceSynchronize();
            endWallTime = omp_get_wtime();
            wallTime = endWallTime - startWallTime;
            maxWallTime = std::max(maxWallTime, wallTime);
            minWallTime = std::min(minWallTime, wallTime);
            avgWallTime += wallTime;
            wallTime *= 1000;
            if (missingFuncFlag == 1)
            {
                printf("\nThe functionality doesn't yet exist in RPP\n");
                return -1;
            }
            // Copy output buffer to host
            hipMemcpy(outputF32, d_outputF32, oBufferSizeInBytes, hipMemcpyDeviceToHost);
            if(testType == 0)
            {
                cout << "\n\nGPU Backend Wall Time: " << wallTime <<" ms per nifti file"<< endl;
                if(DEBUG_MODE)
                {
                    std::ofstream refFile;
                    std::string refFileName;
                    if(layoutType == 0)
                        refFileName = testCaseName + "_nifti_hip_pkd3.csv";
                    else if(layoutType == 1)
                        refFileName = testCaseName + "_nifti_hip_pln3.csv";
                    else
                        refFileName = testCaseName + "_nifti_hip_pln1.csv";
                    refFile.open(refFileName);
                    for (int i = 0; i < oBufferSize; i++)
                        refFile << *(outputF32 + i) << ",";
                    refFile.close();
                }

                if(qaFlag)
                {
                    compare_output(outputF32, oBufferSize, testCaseName, layoutType, descriptorPtr3D, (RpptRoiXyzwhd *)roiGenericSrcPtr, dst_path);
                }

                for(int batchCount = 0; batchCount < batchSize; batchCount++)
                {
                    int index = iterCount * batchSize + batchCount;
                    Rpp32f *outputTemp = outputF32 + batchCount * descriptorPtr3D->strides[0];
                    for(int i = 0; i < numChannels; i++) // temporary changes to process pln3
                    {
                        int xyFrameSize = niftiHeaderTemp[batchCount].dim[1] * niftiHeaderTemp[batchCount].dim[2];
                        int xyFrameSizeROI = roiGenericSrcPtr[batchCount].roiWidth * roiGenericSrcPtr[batchCount].roiHeight;

                        uint dataSize = niftiHeaderTemp[batchCount].dim[1] * niftiHeaderTemp[batchCount].dim[2] * niftiHeaderTemp[batchCount].dim[3];
                        uchar *niftiDataU8 = (uchar *) malloc(dataSize * sizeof(uchar));
                        uchar *outputBufferOpenCV = (uchar *)calloc(xyFrameSizeROI, sizeof(uchar));

                        // Convert RpptDataType::F32 strided buffer to default NIFTI_DATATYPE unstrided buffer
                        Rpp64u increment;
                        if (descriptorPtr3D->layout == RpptLayout::NCDHW)
                            increment = (Rpp64u)descriptorPtr3D->strides[1];
                        else
                            increment = 1;
                        convert_output_Rpp32f_to_niftitype_generic(outputTemp + i * increment, descriptorPtr3D, niftiDataArray[batchCount], &niftiHeaderTemp[batchCount]);
                        NIFTI_DATATYPE min = niftiDataArray[batchCount][0];
                        NIFTI_DATATYPE max = niftiDataArray[batchCount][0];
                        for (int i = 0; i < dataSize; i++)
                        {
                            min = std::min(min, niftiDataArray[batchCount][i]);
                            max = std::max(max, niftiDataArray[batchCount][i]);
                        }
                        Rpp32f multiplier = 255.0f / (max - min);
                        for (int i = 0; i < dataSize; i++)
                            niftiDataU8[i] = (uchar)((niftiDataArray[batchCount][i] - min) * multiplier);

                        uchar *niftiDataU8Temp = niftiDataU8;
                        for (int zPlane = roiGenericSrcPtr[batchCount].xyz.z; zPlane < roiGenericSrcPtr[batchCount].xyz.z + roiGenericSrcPtr[batchCount].roiDepth; zPlane++)
                        {
                            write_image_from_nifti_opencv(niftiDataU8Temp, niftiHeaderTemp[batchCount].dim[1], (RpptRoiXyzwhd *)roiGenericSrcPtr, outputBufferOpenCV, zPlane, i, batchCount, dst_path, testCaseName, index);
                            niftiDataU8Temp += xyFrameSize;
                        }

                        write_nifti_file(&niftiHeaderTemp[batchCount], niftiDataArray[batchCount], index, i, dst_path, testCaseName);

                        if(i == 0)
                        {
                            std::string command = "convert -delay 10 -loop 0 " + std::string(dst_path) + "/" + testCaseName + "_nifti_" + std::to_string(index) + "_zPlane_chn_0_*.jpg " + std::string(dst_path) + "/" + testCaseName + "_niftiOutput_" + std::to_string(index) + "_chn_" + std::to_string(i) + ".gif";
                            system(command.c_str());
                        }
                        if(i == 1)
                        {
                            std::string command = "convert -delay 10 -loop 0 " + std::string(dst_path) + "/" + testCaseName + "_nifti_" + std::to_string(index) + "_zPlane_chn_1_*.jpg " + std::string(dst_path) + "/" + testCaseName + "_niftiOutput_" + std::to_string(index) + "_chn_" + std::to_string(i) + ".gif";
                            system(command.c_str());
                        }
                        if(i == 2)
                        {
                            std::string command = "convert -delay 10 -loop 0 " + std::string(dst_path) + "/" + testCaseName + "_nifti_" + std::to_string(index) + "_zPlane_chn_2_*.jpg " + std::string(dst_path) + "/" + testCaseName + "_niftiOutput_" + std::to_string(index) + "_chn_" + std::to_string(i) + ".gif";
                            system(command.c_str());
                        }
                        free(niftiDataU8);
                        free(outputBufferOpenCV);
                    }
                }
            }
        }
    }

    if(testType == 1)
    {
        // Display measured times
        maxWallTime *= 1000;
        minWallTime *= 1000;
        avgWallTime *= 1000;
        avgWallTime /= (numRuns * noOfIterations);
        cout << fixed << "\nmax,min,avg wall times in ms/batch = " << maxWallTime << "," << minWallTime << "," << avgWallTime;
    }

    rppDestroyHost(handle);

    // Free memory
    free(niftiDataArray);
    free(inputF32);
    free(outputF32);
    hipHostFree(pinnedMemROI);
    hipHostFree(pinnedMemArgs);
    hipFree(d_inputF32);
    hipFree(d_outputF32);

    return(0);
}