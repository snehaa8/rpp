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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <time.h>
#include <omp.h>
#include <fstream>
#include <unistd.h>
#include <dirent.h>
#include <boost/filesystem.hpp>
#include <cstdlib>
#include <ctime>
#include <algorithm>
#include <iterator>
#include "rpp.h"
// #include "../rpp_test_suite_common.h"

using namespace std;
namespace fs = boost::filesystem;

inline void remove_substring(string &str, string &pattern)
{
    std::string::size_type i = str.find(pattern);
    while (i != std::string::npos)
    {
        str.erase(i, pattern.length());
        i = str.find(pattern, i);
   }
}

void compute_strides(RpptGenericDescPtr descriptorPtr)
{
    if (descriptorPtr->numDims > 0)
    {
        uint64_t v = 1;
        for (int i = descriptorPtr->numDims; i > 0; i--)
        {
            descriptorPtr->strides[i] = v;
            v *= descriptorPtr->dims[i];
        }
        descriptorPtr->strides[0] = v;
    }
}

string get_path(Rpp32u nDim, Rpp32u readType)
{
    string refPath = get_current_dir_name();
    string pattern = "HOST/build";
    string finalPath = "";
    remove_substring(refPath, pattern);
    string dim = std::to_string(nDim) + "d";

    if (readType == 0 || readType == 2 || readType == 3)
        finalPath = refPath + "NORMALIZE/input/" + dim;
    else
        finalPath = refPath + "NORMALIZE/output/" + dim;

    return finalPath;
}

Rpp32u get_buffer_length(Rpp32u nDim)
{
    string dimSpecifier = std::to_string(nDim) + "d";
    string refPath = get_path(nDim, 0);
    string refFile = refPath + "/" + dimSpecifier + "_" + "input" + std::to_string(0) + ".txt";
    ifstream file(refFile);
    Rpp32u bufferLength = std::count(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>(), '\n');
    return bufferLength;
}

void read_data(Rpp32f *data, Rpp32u nDim, Rpp32u readType, Rpp32u bufferLength, Rpp32u batchSize)
{
    Rpp32u sampleLength = bufferLength / batchSize;
    if(nDim != 3)
    {
        std::cerr<<"\nGolden Inputs / Outputs are generated only for 3D data"<<std::endl;
        exit(0);
    }

    string refPath = get_path(nDim, readType);
    string dimSpecifier = std::to_string(nDim) + "d";
    string type = "input";
    if (readType == 1)
        type = "output";
    if(readType == 2)
        type = "mean";
    if(readType == 3)
        type = "stddev";

    for(int i = 0; i < batchSize; i++)
    {
        string refFilePath = refPath + "/" + dimSpecifier + "_" + type + std::to_string(i) + ".txt";
        Rpp32f *curData = data + i * sampleLength;
        fstream refFile;
        refFile.open(refFilePath, ios::in);
        if(!refFile.is_open())
        {
            cerr<<"Unable to open the file specified! Please check the path of the file given as input"<<endl;
            break;
        }
        for(int j = 0; j < sampleLength; j++)
            refFile >> curData[j];
    }
}

void compare_output(Rpp32f *outputF32, Rpp32u nDim, Rpp32u batchSize)
{
    Rpp32u bufferLength = get_buffer_length(nDim);
    Rpp32f *refOutput = (Rpp32f *)calloc(bufferLength * batchSize, sizeof(Rpp32f));
    read_data(refOutput, nDim, 1, bufferLength * batchSize, batchSize);
    int fileMatch = 0;
    for(int i = 0; i < batchSize; i++)
    {
        Rpp32f *ref = refOutput + i * bufferLength;
        Rpp32f *out = outputF32 + i * bufferLength;
        int cnt = 0;
        for(int j = 0; j < bufferLength; j++)
        {
            bool invalid_comparision = ((out[j] == 0.0f) && (ref[j] != 0.0f));
            if(!invalid_comparision && abs(out[j] - ref[j]) < 1e-20)
                cnt += 1;
        }
        if (cnt == bufferLength)
            fileMatch++;
    }
    if (fileMatch == batchSize)
        std::cerr<<"\nPASSED!"<<std::endl;
    else
        std::cerr << "\nFAILED! " << fileMatch << "/" << batchSize << " outputs are matching with reference outputs" << std::endl;

    free(refOutput);
}

int main(int argc, char **argv)
{
    Rpp32u nDim, batchSize, testType;
    bool qaMode;

    nDim = atoi(argv[1]);
    batchSize = atoi(argv[2]);
    testType = atoi(argv[3]);
    qaMode = atoi(argv[4]);

    if (qaMode && batchSize != 1)
    {
        std::cerr<<"QA mode can only run with batchsize 1"<<std::endl;
        return -1;
    }


    // Set the number of threads to be used by OpenMP pragma for RPP batch processing on host.
    // If numThreads value passed is 0, number of OpenMP threads used by RPP will be set to batch size
    Rpp32u numThreads = 0;
    rppHandle_t handle;
    rppCreateWithBatchSize(&handle, batchSize, numThreads);

    double startWallTime, endWallTime;
    double avgWallTime = 0, wallTime = 0;
    Rpp32u numRuns = 1;
    if (testType)
        numRuns = 100;

    // case-wise RPP API and measure time script for Unit and Performance test
    printf("\nRunning normalize %d times (each time with a batch size of %d) and computing mean statistics...", numRuns, batchSize);

    // set src/dst generic tensor descriptors
    RpptGenericDesc srcDescriptor, dstDescriptor;
    RpptGenericDescPtr srcDescriptorPtrND, dstDescriptorPtrND;
    srcDescriptorPtrND  = &srcDescriptor;
    srcDescriptorPtrND->numDims = nDim;
    srcDescriptorPtrND->offsetInBytes = 0;
    srcDescriptorPtrND->dataType = RpptDataType::F32;
    srcDescriptorPtrND->layout = RpptLayout::NDHWC;

    dstDescriptorPtrND  = &dstDescriptor;
    dstDescriptorPtrND->numDims = nDim;
    dstDescriptorPtrND->offsetInBytes = 0;
    dstDescriptorPtrND->dataType = RpptDataType::F32;
    dstDescriptorPtrND->layout = RpptLayout::NDHWC;

    int channels = 16;
    Rpp32f *meanTensor = (Rpp32f *)calloc(channels, sizeof(Rpp32f));
    Rpp32f *stdDevTensor = (Rpp32f *)calloc(channels, sizeof(Rpp32f));
    Rpp32u *roiTensor = (Rpp32u *)calloc(nDim * batchSize, sizeof(Rpp32u));
    Rpp32f *inputF32 = NULL, *outputF32 = NULL;

    if(qaMode)
    {
        if(nDim == 3)
        {
            for(int i = 0; i < batchSize * 3; i += 3)
            {
                roiTensor[i] = 3;
                roiTensor[i + 1] = 4;
                roiTensor[i + 2] = 16;
            }
            read_data(meanTensor, nDim, 2,  16, 1);
            read_data(stdDevTensor, nDim, 3,  16, 1);
        }
    }
    else
    {   // limiting max value in a dimension to 150 for testing purposes
        for(int i = 0; i < batchSize * nDim; i += 3)
        {
            roiTensor[i] = std::rand() % 150;
            roiTensor[i + 1] = std::rand() % 150;
            roiTensor[i + 2] = std::rand() % 150;
            for(int j = 0; j < roiTensor[i + 2]; j++)
            {
                meanTensor[j] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
                stdDevTensor[j] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
            }
        }
    }

    // set dims and compute strides
    srcDescriptorPtrND->dims[0] = batchSize;
    dstDescriptorPtrND->dims[0] = batchSize;
    for(int i = 1; i <= nDim; i++)
        srcDescriptorPtrND->dims[i] = dstDescriptorPtrND->dims[i] = roiTensor[i - 1];

    compute_strides(srcDescriptorPtrND);
    compute_strides(dstDescriptorPtrND);

    Rpp32u numValues = 1;
    for(int i = 0; i <= nDim; i++)
        numValues *= srcDescriptorPtrND->dims[i];

    // allocate memory for input / output
    inputF32 = (Rpp32f *)calloc(numValues, sizeof(Rpp32f));
    outputF32 = (Rpp32f *)calloc(numValues, sizeof(Rpp32f));

    // read input data
    if(qaMode)
        read_data(inputF32, nDim, 0,  numValues, batchSize);
    else
    {
        std::srand(0);
        for(int i = 0; i < numValues; i++)
            inputF32[i] = (float)(std::rand() % 255);
    }

    int axis_mask = 3;
    float scale = 1.0;
    float shift = 0.0;
    for(int i = 0; i < numRuns; i++)
    {
        startWallTime = omp_get_wtime();
        rppt_normalize_generic_host(inputF32, srcDescriptorPtrND, outputF32, dstDescriptorPtrND, axis_mask, meanTensor, stdDevTensor, scale, shift, roiTensor, handle);
        endWallTime = omp_get_wtime();

        wallTime = endWallTime - startWallTime;
        avgWallTime += wallTime;
    }

    // compare outputs
    if(qaMode)
        compare_output(outputF32, nDim, batchSize);
    else
    {
        avgWallTime *= 1000;
        avgWallTime /= numRuns;
        cout << fixed << "\navg wall times in ms/batch = " << avgWallTime << endl;
    }

    free(inputF32);
    free(outputF32);
    free(meanTensor);
    free(stdDevTensor);
    free(roiTensor);
    return 0;
}