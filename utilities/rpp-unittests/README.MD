# RPP UNIT TESTS
The Radeon Performance Primitives library provides a test suite, to run unit tests on its functionalities, executed under 3 backend scenarios - (HOST/OCL/HIP):
- HOST backend - (On a CPU with HOST backend)
- OCL backend - (On a GPU with OpenCL backend)
- HIP backend - (On a GPU with HIP backend)
## Test Suite for HOST backend
### Option 1 - Testing all functionalities with default images (3 224x224 images - batch size = 3)
Executing the following should run and dump output images for all variants of functionalities running with HOST backend.
```
cd utilities/rpp-unittests/HOST
./testAllScript.sh
```
### Option 2 - Testing a sub-range of functionalities with default images (3 224x224 images - batch size = 3)
Executing the following should run and dump output images for all variants of a sub-range of functionalities running with HOST backend.
```
cd utilities/rpp-unittests/HOST
./testAllScript.sh <S> <E> <U>
```
where:
```
S     CASE_START (Starting case# (0-79))
E     CASE_END (Ending case# (0-79))
U     UNIQUE_FUNC (0 = Skip / 1 = Run)
```
### Option 3 - Testing all functionalities with a differnet image folder
Navigate to utilities/rpp-unittests/HOST/testAllScript.sh and set the correct SRC1, SRC2 and DST paths under "MANUAL OVERRIDE".
```
# <<<<<<<<<<<<<< FOR MANUAL OVERRIDE, JUST REPLACE AND POINT TO THE SOURCE AND DESTINATION FOLDERS HERE >>>>>>>>>>>>>>
SRC_FOLDER_1="<your complete folder path that containes images for  srcPtr1>"
SRC_FOLDER_2="<your complete folder path that containes images for  srcPtr2 (can be same as SRC_FOLDER_1 for trials)>"
DST_FOLDER="<your complete folder path where the dstPtr images can be dumped>"
# <<<<<<<<<<<<<< FOR MANUAL OVERRIDE, JUST REPLACE AND POINT TO THE SOURCE AND DESTINATION FOLDERS HERE >>>>>>>>>>>>>>
```
Run the script
```
./testAllScript.sh
```
### Option 4 - Testing individual functionalities and specific sub-cases
First build the HOST backend test suite.
```
cd utilities/rpp-unittests/HOST
mkdir build
cd build
cmake ..
make
```
Run either of the following executibles for packed-3channel / planar-3channel / planar-1channel images. This will show a list of arguments to be passed to the executible.
```
./BatchPD_host_pkd3
./BatchPD_host_pln3
./BatchPD_host_pln1
```

## Test Suite for OCL backend
### Option 1 - Testing all functionalities with default images (3 224x224 images - batch size = 3)
Executing the following should run and dump output images for all variants of functionalities running with OCL backend.
```
cd utilities/rpp-unittests/OCL
./testAllScript.sh
```
### Option 2 - Testing a sub-range of functionalities with default images (3 224x224 images - batch size = 3)
Executing the following should run and dump output images for all variants of a sub-range of functionalities running with OCL backend.
```
cd utilities/rpp-unittests/OCL
./testAllScript.sh <S> <E> <U>
```
where:
```
S     CASE_START (Starting case# (0-79))
E     CASE_END (Ending case# (0-79))
U     UNIQUE_FUNC (0 = Skip / 1 = Run)
```
### Option 3 - Testing all functionalities with a differnet image folder
Navigate to utilities/rpp-unittests/OCL/testAllScript.sh and set the correct SRC1, SRC2 and DST paths under "MANUAL OVERRIDE".
```
# <<<<<<<<<<<<<< FOR MANUAL OVERRIDE, JUST REPLACE AND POINT TO THE SOURCE AND DESTINATION FOLDERS HERE >>>>>>>>>>>>>>
SRC_FOLDER_1="<your complete folder path that containes images for  srcPtr1>"
SRC_FOLDER_2="<your complete folder path that containes images for  srcPtr2 (can be same as SRC_FOLDER_1 for trials)>"
DST_FOLDER="<your complete folder path where the dstPtr images can be dumped>"
# <<<<<<<<<<<<<< FOR MANUAL OVERRIDE, JUST REPLACE AND POINT TO THE SOURCE AND DESTINATION FOLDERS HERE >>>>>>>>>>>>>>
```
Run the script
```
./testAllScript.sh
```
### Option 4 - Testing individual functionalities and specific sub-cases
First build the OCL backend test suite.
```
cd utilities/rpp-unittests/OCL
mkdir build
cd build
cmake ..
make
```
Run either of the following executibles for packed-3channel / planar-3channel / planar-1channel images. This will show a list of arguments to be passed to the executible.
```
./BatchPD_ocl_pkd3
./BatchPD_ocl_pln3
./BatchPD_ocl_pln1
```

## Test Suite for HIP backend
### Option 1 - Testing all functionalities with default images (3 224x224 images - batch size = 3)
Executing the following should run and dump output images for all variants of functionalities running with HIP backend.
```
cd utilities/rpp-unittests/HIP
./testAllScript.sh
```
### Option 2 - Testing a sub-range of functionalities with default images (3 224x224 images - batch size = 3)
Executing the following should run and dump output images for all variants of a sub-range of functionalities running with HIP backend.
```
cd utilities/rpp-unittests/HIP
./testAllScript.sh <S> <E> <U>
```
where:
```
S     CASE_START (Starting case# (0-79))
E     CASE_END (Ending case# (0-79))
U     UNIQUE_FUNC (0 = Skip / 1 = Run)
```
### Option 3 - Testing all functionalities with a differnet image folder
Navigate to utilities/rpp-unittests/HIP/testAllScript.sh and set the correct SRC1, SRC2 and DST paths under "MANUAL OVERRIDE".
```
# <<<<<<<<<<<<<< FOR MANUAL OVERRIDE, JUST REPLACE AND POINT TO THE SOURCE AND DESTINATION FOLDERS HERE >>>>>>>>>>>>>>
SRC_FOLDER_1="<your complete folder path that containes images for  srcPtr1>"
SRC_FOLDER_2="<your complete folder path that containes images for  srcPtr2 (can be same as SRC_FOLDER_1 for trials)>"
DST_FOLDER="<your complete folder path where the dstPtr images can be dumped>"
# <<<<<<<<<<<<<< FOR MANUAL OVERRIDE, JUST REPLACE AND POINT TO THE SOURCE AND DESTINATION FOLDERS HERE >>>>>>>>>>>>>>
```
Run the script
```
./testAllScript.sh
```
### Option 4 - Testing individual functionalities and specific sub-cases
First build the HIP backend test suite.
```
cd utilities/rpp-unittests/HIP
mkdir build
cd build
cmake ..
make
```
Run either of the following executibles for packed-3channel / planar-3channel / planar-1channel images. This will show a list of arguments to be passed to the executible.
```
./BatchPD_hip_pkd3
./BatchPD_hip_pln3
./BatchPD_hip_pln1
```