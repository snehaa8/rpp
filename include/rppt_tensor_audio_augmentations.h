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

#ifndef RPPT_TENSOR_AUDIO_AUGMENTATIONS_H
#define RPPT_TENSOR_AUDIO_AUGMENTATIONS_H
#include "rpp.h"
#include "rppdefs.h"
#ifdef __cplusplus
extern "C" {
#endif

/******************** non_silent_region_detection ********************/

// Non Silent Region Detection augmentation for 1D audio buffer

// *param[in] srcPtr source tensor memory
// *param[in] srcDescPtr source tensor descriptor
// *param[in] srcSize source audio buffer length
// *param[out] detectedIndex beginning index of non silent region
// *param[out] detectionLength length of non silent region
// *param[in] cutOffDB threshold(dB) below which the signal is considered silent
// *param[in] windowLength size of the sliding window used to calculate of the short-term power of the signal
// *param[in] referencePower reference power that is used to convert the signal to dB.
// *param[in] resetInterval number of samples after which the moving mean average is recalculated to avoid loss of precision
// *param[in] rppHandle HIP-handle for "_gpu" variants and Host-handle for "_host" variants
// *returns a  RppStatus enumeration.
// *retval RPP_SUCCESS : successful completion
// *retval RPP_ERROR : Error

RppStatus rppt_non_silent_region_detection_host(RppPtr_t srcPtr, RpptDescPtr srcDescPtr, Rpp32s *srcSize, Rpp32f *detectedIndexTensor, Rpp32f *detectionLengthTensor, Rpp32f cutOffDB, Rpp32s windowLength, Rpp32f referencePower, Rpp32s resetInterval, rppHandle_t rppHandle);

/******************** to_decibels ********************/

// To Decibels augmentation for magnitude buffer

// *param[in] srcPtr source tensor memory
// *param[in] srcDescPtr source tensor descriptor
// *param[out] dstPtr destination tensor memory
// *param[in] dstDescPtr destination tensor descriptor
// *param[in] srcDims source dimensions
// *param[in] cutOffDB  minimum or cut-off ratio in dB
// *param[in] multiplier factor by which the logarithm is multiplied
// *param[in] referenceMagnitude Reference magnitude if not provided maximum value of input used as reference
// *param[in] rppHandle HIP-handle for "_gpu" variants and Host-handle for "_host" variants
// *returns a  RppStatus enumeration.
// *retval RPP_SUCCESS : successful completion
// *retval RPP_ERROR : Error

RppStatus rppt_to_decibels_host(RppPtr_t srcPtr, RpptDescPtr srcDescPtr, RppPtr_t dstPtr, RpptDescPtr dstDescPtr, RpptImagePatchPtr srcDims, Rpp32f cutOffDB, Rpp32f multiplier, Rpp32f referenceMagnitude, rppHandle_t rppHandle);

/******************** pre_emphasis_filter ********************/

// Pre Emphasis Filter augmentation for 1D audio buffer

// *param[in] srcPtr source tensor memory
// *param[in] srcDescPtr source tensor descriptor
// *param[out] dstPtr destination tensor memory
// *param[in] dstDescPtr destination tensor descriptor
// *param[in] srcSize source audio buffer length
// *param[in] coeffTensor preemphasis coefficient
// *param[in] borderType border value policy
// *param[in] rppHandle HIP-handle for "_gpu" variants and Host-handle for "_host" variants
// *returns a  RppStatus enumeration.
// *retval RPP_SUCCESS : successful completion
// *retval RPP_ERROR : Error

RppStatus rppt_pre_emphasis_filter_host(RppPtr_t srcPtr, RpptDescPtr srcDescPtr, RppPtr_t dstPtr, RpptDescPtr dstDescPtr, Rpp32s *srcSizeTensor, Rpp32f *coeffTensor, RpptAudioBorderType borderType, rppHandle_t rppHandle);

/******************** down_mixing ********************/

// Downmix multi channel audio buffer to single channel audio buffer

// *param[in] srcPtr source tensor memory
// *param[in] srcDescPtr source tensor descriptor
// *param[out] dstPtr destination tensor memory
// *param[in] dstDescPtr destination tensor descriptor
// *param[in] srcLengthTensor number of samples per channel
// *param[in] channelsTensor number of channels in audio buffer
// *param[in] rppHandle HIP-handle for "_gpu" variants and Host-handle for "_host" variants
// *returns a  RppStatus enumeration.
// *retval RPP_SUCCESS : successful completion
// *retval RPP_ERROR : Error

RppStatus rppt_down_mixing_host(RppPtr_t srcPtr, RpptDescPtr srcDescPtr, RppPtr_t dstPtr, RpptDescPtr dstDescPtr, Rpp32s *srcLengthTensor, Rpp32s *channelsTensor, bool normalizeWeights, rppHandle_t rppHandle);

/******************** slice_audio ********************/

// Extracts a subtensor or slice from the audio buffer

// *param[in] srcPtr source tensor memory
// *param[in] srcDescPtr source tensor descriptor
// *param[out] dstPtr destination tensor memory
// *param[in] dstDescPtr destination tensor descriptor
// *param[in] srcLengthTensor number of samples per channel
// *param[in] anchor starting index of the slice
// *param[in] shape length of the slice
// *param[in] axes axes along which slice is needed
// *param[in] fillValues fill values based on out of Bound policy
// *param[in] normalized anchor determines whether the anchor positional input should be interpreted as normalized or as absolute coordinates
// *param[in] normalized shape determines whether the shape positional input should be interpreted as normalized or as absolute coordinates
// *param[in] rppHandle HIP-handle for "_gpu" variants and Host-handle for "_host" variants
// *returns a  RppStatus enumeration.
// *retval RPP_SUCCESS : successful completion
// *retval RPP_ERROR : Error

RppStatus rppt_slice_audio_host(RppPtr_t srcPtr, RpptDescPtr srcDescPtr, RppPtr_t dstPtr, RpptDescPtr dstDescPtr, Rpp32s *srcLengthTensor, Rpp32f *anchorTensor, Rpp32f *shapeTensor, Rpp32f *fillValues, rppHandle_t rppHandle);

/******************** mel_filter_bank ********************/

// Converts a spectrogram to a mel spectrogram

// *param[in] srcPtr source tensor memory
// *param[in] srcDescPtr source tensor descriptor
// *param[out] dstPtr destination tensor memory
// *param[in] dstDescPtr destination tensor descriptor
// *param[in] srcDims source dimensions
// *param[in] maxFreq maximum frequency if not provided maxFreq = sampleRate / 2
// *param[in] minFreq minimum frequency
// *param[in] melFormula formula used to convert frequencies from hertz to mel and from mel to hertz (SLANEY / HTK)
// *param[in] numFilter number of mel filters
// *param[in] sampleRate sampling rate of the audio
// *param[in] normalize boolean variable that determine whether to normalize weights / not
// *param[in] rppHandle HIP-handle for "_gpu" variants and Host-handle for "_host" variants
// *returns a  RppStatus enumeration.
// *retval RPP_SUCCESS : successful completion
// *retval RPP_ERROR : Error

RppStatus rppt_mel_filter_bank_host(RppPtr_t srcPtr, RpptDescPtr srcDescPtr, RppPtr_t dstPtr, RpptDescPtr dstDescPtr, RpptImagePatchPtr srcDims, Rpp32f maxFreq, Rpp32f minFreq, RpptMelScaleFormula melFormula, Rpp32s numFilter, Rpp32f sampleRate, bool normalize, rppHandle_t rppHandle);

/******************** spectrogram ********************/

// Produces a spectrogram from a 1D audio buffer

// *param[in] srcPtr source tensor memory
// *param[in] srcDescPtr source tensor descriptor
// *param[out] dstPtr destination tensor memory
// *param[in] dstDescPtr destination tensor descriptor
// *param[in] srcLengthTensor source audio buffer length (1D tensor of size batchSize)
// *param[in] centerWindows Indicates whether extracted windows should be padded so that the window function is centered at multiples of window_step
// *param[in] reflectPadding Indicates the padding policy when sampling outside the bounds of the signal
// *param[in] windowFunction Samples of the window function that will be multiplied to each extracted window when calculating the STFT
// *param[in] nfft Size of the FFT
// *param[in] power Exponent of the magnitude of the spectrum
// *param[in] windowLength Window size in number of samples
// *param[in] windowStep Step betweeen the STFT windows in number of samples
// *param[in] layout output layout of spectrogram
// *param[in] rppHandle HIP-handle for "_gpu" variants and Host-handle for "_host" variants
// *returns a  RppStatus enumeration.
// *retval RPP_SUCCESS : successful completion
// *retval RPP_ERROR : Error

RppStatus rppt_spectrogram_host(RppPtr_t srcPtr, RpptDescPtr srcDescPtr, RppPtr_t dstPtr, RpptDescPtr dstDescPtr, Rpp32s *srcLengthTensor, bool centerWindows, bool reflectPadding, Rpp32f *windowFunction, Rpp32s nfft, Rpp32s power, Rpp32s windowLength, Rpp32s windowStep, RpptSpectrogramLayout layout, rppHandle_t rppHandle);

/******************** resample ********************/

// Resample audio buffer based on the target sample rate

// *param[in] srcPtr source tensor memory
// *param[in] srcDescPtr source tensor descriptor
// *param[out] dstPtr destination tensor memory
// *param[in] dstDescPtr destination tensor descriptor
// *param[in] inRate Input sampling rate (1D tensor of size batchSize)
// *param[in] outRate Output sampling rate (1D tensor of size batchSize)
// *param[in] srcLengthTensor source audio buffer length (1D tensor of size batchSize)
// *param[in] channelsTensor number of channels in audio buffer (1D tensor of size batchSize)
// *param[in] quality resampling quality, where 0 is the lowest, and 100 is the highest
// *param[in] rppHandle HIP-handle for "_gpu" variants and Host-handle for "_host" variants
// *returns a  RppStatus enumeration.
// *retval RPP_SUCCESS : successful completion
// *retval RPP_ERROR : Error

RppStatus rppt_resample_host(RppPtr_t srcPtr, RpptDescPtr srcDescPtr, RppPtr_t dstPtr, RpptDescPtr dstDescPtr, Rpp32f *inRateTensor, Rpp32f *outRateTensor, Rpp32s *srcLengthTensor, Rpp32s *channelsTensor, Rpp32f quality, rppHandle_t rppHandle);

#ifdef __cplusplus
}
#endif
#endif // RPPT_TENSOR_AUDIO_AUGMENTATIONS_H