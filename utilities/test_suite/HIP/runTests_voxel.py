# Copyright (c) 2019 - 2023 Advanced Micro Devices, Inc. All rights reserved.

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import os
import subprocess  # nosec
import argparse
import sys
import datetime
import shutil

# Set the timestamp
timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

cwd = os.getcwd()
headerFilePath = os.path.join(os.path.dirname(cwd), 'TEST_QA_IMAGES_VOXEL')
dataFilePath = os.path.join(os.path.dirname(cwd), 'TEST_QA_IMAGES_VOXEL')
qaInputFile = os.path.join(os.path.dirname(cwd), 'TEST_QA_IMAGES_VOXEL')

# Check if folder path is empty, if it is the root folder, or if it exists, and remove its contents
def validate_and_remove_contents(path):
    if not path:  # check if a string is empty
        print("Folder path is empty.")
        exit()
    if path == "/*":  # check if the root directory is passed to the function
        print("Root folder cannot be deleted.")
        exit()
    if os.path.exists(path):  # check if the folder exists
        os.system("rm -rvf {}/*".format(path))  # Delete the directory if it exists
    else:
        print("Path is invalid or does not exist.")
        exit()

# Check if the folder is the root folder or exists, and remove the specified subfolders
def validate_and_remove_folders(path, folder):
    if path == "/*":  # check if the root directory is passed to the function
        print("Root folder cannot be deleted.")
        exit()
    if path and os.path.isdir(path + "/.."):  # checks if directory string is not empty and it exists
        output_folders = [folder_name for folder_name in os.listdir(path + "/..") if folder_name.startswith(folder)]

        # Loop through each directory and delete it only if it exists
        for folder_name in output_folders:
            folder_path = os.path.join(path, "..", folder_name)
            if os.path.isdir(folder_path):
                os.system("rm -rf {}".format(folder_path))  # Delete the directory if it exists
                print("Deleted directory:", folder_path)
            else:
                print("Directory not found:", folder_path)

# Check if a case file exists and filter its contents based on certain conditions
def case_file_check(CASE_FILE_PATH):
    try:
        case_file = open(CASE_FILE_PATH,'r')
        for line in case_file:
            print(line)
            if not(line.startswith('"Name"')):
                if TYPE in TENSOR_TYPE_LIST:
                    new_file.write(line)
                    d_counter[TYPE] = d_counter[TYPE] + 1
        case_file.close()
        return True
    except IOError:
        print("Unable to open case results")
        return False

 # Generate a directory name based on certain parameters
def directory_name_generator(qaMode, affinity, type, case, path):
    if qaMode == 0:
        functionality_group = func_group_finder(int(case))

        dst_folder_temp = f"{path}/rpp_{affinity}_{type}_{functionality_group}"
    else:
        dst_folder_temp = path

    return dst_folder_temp

# Process the layout based on the given parameters and generate the directory name and log file layout.
def process_layout(layout, qaMode, case, dstPath):
    if layout == 0:
        dstPathTemp = directory_name_generator(qaMode, "hip", "pkd3", case, dstPath)
        log_file_layout = "pkd3"
    elif layout == 1:
        dstPathTemp = directory_name_generator(qaMode, "hip", "pln3", case, dstPath)
        log_file_layout = "pln3"
    elif layout == 2:
        dstPathTemp = directory_name_generator(qaMode, "hip", "pln1", case, dstPath)
        log_file_layout = "pln1"
    
    return dstPathTemp, log_file_layout

# Validate if a path exists and is a directory
def validate_path(input_path):
    if not os.path.exists(input_path):
        raise ValueError("path " + input_path +" does not exist.")
    if not os.path.isdir(input_path):
        raise ValueError("path " + input_path + " is not a directory.")

# Create layout directories within a destination path based on a layout dictionary
def create_layout_directories(dst_path, layout_dict):
    for layout in range(3):
        current_layout = layout_dict[layout]
        try:
            os.makedirs(dst_path + '/' + current_layout)
        except FileExistsError:
            pass
        folder_list = [f for f in os.listdir(dst_path) if current_layout.lower() in f]
        for folder in folder_list:
            os.rename(dst_path + '/' + folder, dst_path + '/' + current_layout +  '/' + folder)

def get_log_file_list():
    return [
        "../OUTPUT_PERFORMANCE_LOGS_HIP_" + timestamp + "/Tensor_voxel_hip_pkd3_raw_performance_log.txt",
        "../OUTPUT_PERFORMANCE_LOGS_HIP_" + timestamp + "/Tensor_voxel_hip_pln3_raw_performance_log.txt",
        "../OUTPUT_PERFORMANCE_LOGS_HIP_" + timestamp + "/Tensor_voxel_hip_pln1_raw_performance_log.txt"
    ]

# Functionality group finder
def func_group_finder(case_number):
    if case_number == 0 or case_number == 3 or case_number == 4:
        return "arithmetic_operations"
    elif case_number == 1 or case_number == 2:
        return "geometric_augmentations"
    else:
        return "miscellaneous"

# Generate performance reports based on counters and a list of types
def generate_performance_reports(d_counter, TYPE_LIST):
    import pandas as pd
    pd.options.display.max_rows = None
    # Generate performance report
    for TYPE in TYPE_LIST:
        print("\n\n\nKernels tested - ", d_counter[TYPE], "\n\n")
        df = pd.read_csv(RESULTS_DIR + "/consolidated_results_" + TYPE + ".stats.csv")
        df["AverageMs"] = df["AverageNs"] / 1000000
        dfPrint = df.drop(['Percentage'], axis = 1)
        dfPrint["HIP Kernel Name"] = dfPrint.iloc[:,0].str.lstrip("Hip_")
        dfPrint_noIndices = dfPrint.astype(str)
        dfPrint_noIndices.replace(['0', '0.0'], '', inplace = True)
        dfPrint_noIndices = dfPrint_noIndices.to_string(index = False)
        print(dfPrint_noIndices)

# Parse and validate command-line arguments for the RPP test suite
def rpp_test_suite_parser_and_validator():
    parser = argparse.ArgumentParser()
    parser.add_argument("--header_path", type = str, default = headerFilePath, help = "Path to the nii header")
    parser.add_argument("--data_path", type = str, default = dataFilePath, help = "Path to the nii data file")
    parser.add_argument("--case_start", type = int, default = 0, help = "Testing range starting case # - (0:4)")
    parser.add_argument("--case_end", type = int, default = 4, help = "Testing range ending case # - (0:4)")
    parser.add_argument('--test_type', type = int, default = 0, help = "Type of Test - (0 = Unit tests / 1 = Performance tests)")
    parser.add_argument('--case_list', nargs = "+", help = "List of case numbers to list", required = False)
    parser.add_argument('--profiling', type = str , default = 'NO', help = 'Run with profiler? - (YES/NO)', required = False)
    parser.add_argument('--qa_mode', type = int, default = 0, help = "Run with qa_mode? Output images from tests will be compared with golden outputs - (0 / 1)", required = False)
    parser.add_argument('--num_runs', type = int, default = 1, help = "Specifies the number of runs for running the performance tests")
    args = parser.parse_args()

    # check if the folder exists
    validate_path(args.header_path)
    validate_path(args.data_path)
    validate_path(qaInputFile)

    # validate the parameters passed by user
    if ((args.case_start < 0 or args.case_start > 4) or (args.case_end < 0 or args.case_end > 4)):
        print("Starting case# and Ending case# must be in the 0:4 range. Aborting!")
        exit(0)
    elif args.case_end < args.case_start:
        print("Ending case# must be greater than starting case#. Aborting!")
        exit(0)
    elif args.test_type < 0 or args.test_type > 1:
        print("Test Type# must be in the 0 / 1. Aborting!")
        exit(0)
    elif args.qa_mode < 0 or args.qa_mode > 1:
        print("QA mode must be in the 0 / 1. Aborting!")
        exit(0)
    elif args.case_list is not None and args.case_start > 0 and args.case_end < 4:
        print("Invalid input! Please provide only 1 option between case_list, case_start and case_end")
        exit(0)
    elif args.num_runs <= 0:
        print("Number of Runs must be greater than 0. Aborting!")
        exit(0)
    elif args.profiling != 'YES' and args.profiling != 'NO':
        print("Profiling option value must be either 'YES' or 'NO'.")
        exit(0)

    if args.case_list is None:
        args.case_list = range(args.case_start, args.case_end + 1)
        args.case_list = [str(x) for x in args.case_list]
    else:
        for case in args.case_list:
            if int(case) < 0 or int(case) > 4:
                 print("The case# must be in the 0:4 range!")
                 exit(0)

    # if QA mode is enabled overwrite the input folders with the folders used for generating golden outputs
    if args.qa_mode:
        args.header_path = headerFilePath
        args.data_path = dataFilePath

    return args

args = rpp_test_suite_parser_and_validator()
headerPath = args.header_path
dataPath = args.data_path
caseStart = args.case_start
caseEnd = args.case_end
testType = args.test_type
caseList = args.case_list
profilingOption = args.profiling
qaMode = args.qa_mode
numRuns = args.num_runs

if qaMode and os.path.abspath(qaInputFile) != os.path.abspath(headerPath):
    print("QA mode should only run with the given Input path: ", qaInputFile)
    exit(0)

if(testType == 0):
    if qaMode:
        outFilePath = os.path.join(os.path.dirname(cwd), 'QA_RESULTS_HIP_VOXEL' + timestamp)
    else:
        outFilePath = os.path.join(os.path.dirname(cwd), 'OUTPUT_IMAGES_HIP_VOXEL' + timestamp)
    numRuns = 1
elif(testType == 1):
    if numRuns == 0:
        numRuns = 100 #default numRuns for running performance tests
    outFilePath = os.path.join(os.path.dirname(cwd), 'OUTPUT_PERFORMANCE_LOGS_HIP_VOXEL' + timestamp)
else:
    print("Invalid TEST_TYPE specified. TEST_TYPE should be 0/1 (0 = Unittests / 1 = Performancetests)")
    exit()
os.mkdir(outFilePath)
loggingFolder = outFilePath
dstPath = outFilePath

# Validate DST_FOLDER
validate_and_remove_contents(dstPath)

# Enable extglob
if os.path.exists("build"):
    shutil.rmtree("build")
os.makedirs("build")
os.chdir("build")

# Run cmake and make commands
subprocess.run(["cmake", ".."])
subprocess.run(["make", "-j16"])

# Create folders based on testType and profilingOption
if testType == 1 and profilingOption == "YES":
    os.makedirs(f"{dstPath}/Tensor_PKD3")
    os.makedirs(f"{dstPath}/Tensor_PLN1")
    os.makedirs(f"{dstPath}/Tensor_PLN3")

print("\n\n\n\n\n")
print("##########################################################################################")
print("Running all layout Inputs...")
print("##########################################################################################")

if(testType == 0):
    for case in caseList:
        if int(case) < 0 or int(case) > 4:
            print(f"Invalid case number {case}. Case number must be in the range of 0 to 4!")
            continue
        for layout in range(3):
            dstPathTemp, log_file_layout = process_layout(layout, qaMode, case, dstPath)

            if qaMode == 0:
                if not os.path.isdir(dstPathTemp):
                    os.mkdir(dstPathTemp)

            print("\n\n\n\n")
            print("--------------------------------")
            print("Running a New Functionality...")
            print("--------------------------------")
            print(f"./Tensor_hip {headerPath} {dataPath} {dstPathTemp} {layout} {case} {numRuns} {testType} {qaMode}")
            subprocess.run(["./Tensor_voxel_hip", headerPath, dataPath, dstPath, str(layout), str(case), str(numRuns), str(testType), str(qaMode)])

            print("------------------------------------------------------------------------------------------")
    layoutDict = {0:"PKD3", 1:"PLN3", 2:"PLN1"}
    if qaMode == 0:
        create_layout_directories(dstPath, layoutDict)
else:
    log_file_list = get_log_file_list()

    functionality_group_list = [
    "color_augmentations",
    "data_exchange_operations",
    "effects_augmentations",
    "filter_augmentations",
    "geometric_augmentations",
    "morphological_operations"
    ]

    if (testType == 1 and profilingOption == "NO"):
        for case in caseList:
            if int(case) < 0 or int(case) > 4:
                print(f"Invalid case number {case}. Case number must be in the range of 0 to 4!")
                continue
            for layout in range(3):
                dstPathTemp, log_file_layout = process_layout(layout, qaMode, case, dstPath)

                print("\n\n\n\n")
                print("--------------------------------")
                print("Running a New Functionality...")
                print("--------------------------------")

                with open(f"{loggingFolder}/Tensor_hip_{log_file_layout}_raw_performance_log.txt", "a") as log_file:
                    print(f"./Tensor_hip {headerPath} {dataPath} {dstPath} {layout} {case}{numRuns} {testType} {qaMode}")
                    process = subprocess.Popen(["./Tensor_voxel_hip", headerPath, dataPath, dstPath, str(layout), str(case), str(numRuns), str(testType), str(qaMode)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                    while True:
                        output = process.stdout.readline()
                        if not output and process.poll() is not None:
                            break
                        print(output.strip())
                        log_file.write(output)
                print("------------------------------------------------------------------------------------------")

        for log_file in log_file_list:
            # Opening log file
            try:
                f = open(log_file,"r")
                print("\n\n\nOpened log file -> " + log_file)
            except IOError:
                print("Skipping file -> " + log_file)
                continue

            stats = []
            maxVals = []
            minVals = []
            avgVals = []
            functions = []
            frames = []
            prevLine = ""
            funcCount = 0

            # Loop over each line
            for line in f:
                for functionality_group in functionality_group_list:
                    if functionality_group in line:
                        functions.extend([" ", functionality_group, " "])
                        frames.extend([" ", " ", " "])
                        maxVals.extend([" ", " ", " "])
                        minVals.extend([" ", " ", " "])
                        avgVals.extend([" ", " ", " "])

                if "max,min,avg wall times in ms/batch" in line:
                    split_word_start = "Running "
                    split_word_end = " "+ str(numRuns)
                    prevLine = prevLine.partition(split_word_start)[2].partition(split_word_end)[0]
                    if prevLine not in functions:
                        functions.append(prevLine)
                        frames.append(str(numRuns))
                        split_word_start = "max,min,avg wall times in ms/batch = "
                        split_word_end = "\n"
                        stats = line.partition(split_word_start)[2].partition(split_word_end)[0].split(",")
                        maxVals.append(stats[0])
                        minVals.append(stats[1])
                        avgVals.append(stats[2])
                        funcCount += 1

                if line != "\n":
                    prevLine = line

            # Print log lengths
            print("Functionalities - " + str(funcCount))

            # Print summary of log
            print("\n\nFunctionality\t\t\t\t\t\tFrames Count\tmax(ms/batch)\t\tmin(ms/batch)\t\tavg(ms/batch)\n")
            if len(functions) != 0:
                maxCharLength = len(max(functions, key = len))
                functions = [x + (' ' * (maxCharLength - len(x))) for x in functions]
                for i, func in enumerate(functions):
                    print(func + "\t" + str(frames[i]) + "\t\t" + str(maxVals[i]) + "\t" + str(minVals[i]) + "\t" + str(avgVals[i]))
            else:
                print("No variants under this category")

            # Closing log file
            f.close()
    elif (testType == 1 and profilingOption == "YES"):
        for case in caseList:
            if int(case) < 0 or int(case) > 4:
                print(f"Invalid case number {case}. Case number must be in the range of 0 to 4!")
                continue
            for layout in range(3):
                dstPathTemp, log_file_layout = process_layout(layout, qaMode, case, dstPath)

                print("\n\n\n\n")
                print("--------------------------------")
                print("Running a New Functionality...")
                print("--------------------------------")

                if not os.path.exists(f"{dstPath}/Tensor_PLN1/case_{case}"):
                    os.mkdir(f"{dstPath}/Tensor_PLN1/case_{case}")
                with open(f"{loggingFolder}/Tensor_hip_{log_file_layout}_raw_performance_log.txt", "a") as log_file:
                    print(f"\nrocprof --basenames on --timestamp on --stats -o {dstPath}/Tensor_PLN3/case_{case}/output_case{case}.csv ./Tensor_hip {headerPath} {dataPath} {dstPath}  {layout} {case}{numRuns} {testType} {qaMode}")
                    process = subprocess.Popen([ 'rocprof', '--basenames', 'on', '--timestamp', 'on', '--stats', '-o', f"{dstPath}/Tensor_PLN3/case_{case}/output_case{case}.csv", './Tensor_hip', headerPath, dataPath, dstPath, str(layout), str(case), str(numRuns), str(testType), str(qaMode)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                    while True:
                        output = process.stdout.readline()
                        if not output and process.poll() is not None:
                            break
                        print(output.strip())
                        log_file.write(output.decode('utf-8'))

                    print("------------------------------------------------------------------------------------------")

        RESULTS_DIR = ""
        RESULTS_DIR = "../OUTPUT_PERFORMANCE_LOGS_HIP_" + timestamp
        print("RESULTS_DIR = " + RESULTS_DIR)
        CONSOLIDATED_FILE_TENSOR_PKD3 = RESULTS_DIR + "/consolidated_results_Tensor_PKD3.stats.csv"
        CONSOLIDATED_FILE_TENSOR_PLN1 = RESULTS_DIR + "/consolidated_results_Tensor_PLN1.stats.csv"
        CONSOLIDATED_FILE_TENSOR_PLN3 = RESULTS_DIR + "/consolidated_results_Tensor_PLN3.stats.csv"

        TYPE_LIST = ["Tensor_PKD3", "Tensor_PLN1", "Tensor_PLN3"]
        TENSOR_TYPE_LIST = ["Tensor_PKD3", "Tensor_PLN1", "Tensor_PLN3"]
        CASE_NUM_LIST = caseList
        OFT_LIST = range(0, 2, 1)
        d_counter = {"Tensor_PKD3":0, "Tensor_PLN1":0, "Tensor_PLN3":0}

        for TYPE in TYPE_LIST:
            # Open csv file
            new_file = open(RESULTS_DIR + "/consolidated_results_" + TYPE + ".stats.csv",'w')
            new_file.write('"HIP Kernel Name","Calls","TotalDurationNs","AverageNs","Percentage"\n')

            prev = ""

            # Loop through cases
            for CASE_NUM in CASE_NUM_LIST:

                # Add functionality group header
                if CASE_NUM in NEW_FUNC_GROUP_LIST:
                    FUNC_GROUP = func_group_finder(CASE_NUM)
                    new_file.write("0,0,0,0,0\n")
                    new_file.write(FUNC_GROUP + ",0,0,0,0\n")
                    new_file.write("0,0,0,0,0\n")

                # Set results directory
                CASE_RESULTS_DIR = RESULTS_DIR + "/" + TYPE + "/case_" + str(CASE_NUM)
                print("CASE_RESULTS_DIR = " + CASE_RESULTS_DIR)

            new_file.close()
            subprocess.call(['chown', '{}:{}'.format(os.getuid(), os.getgid()), RESULTS_DIR + "/consolidated_results_" + TYPE + ".stats.csv"])  # nosec
        try:
            generate_performance_reports(d_counter, TYPE_LIST)

        except ImportError:
            print("\nPandas not available! Results of GPU profiling experiment are available in the following files:\n" + \
                    CONSOLIDATED_FILE_TENSOR_PKD3 + "\n" + \
                        CONSOLIDATED_FILE_TENSOR_PLN1 + "\n" + \
                            CONSOLIDATED_FILE_TENSOR_PLN3 + "\n")

        except IOError:
            print("Unable to open results in " + RESULTS_DIR + "/consolidated_results_" + TYPE + ".stats.csv")