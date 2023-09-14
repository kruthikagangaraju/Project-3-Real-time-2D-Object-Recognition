# Project 3: Real-time 2-D Object Recognition 

## Overview
Project 3 involves object detection which is the task of detecting instances of objects of a certain class within an image. In this project, we created a database of different objects and using live feed from a webcam, determined the given object in the frame by classifying it according to the previously created database. 

## Time Travel days
Used for this project: 0
 
## Development Environment
Operating System: Windows 10 64 bit \
IDE: Microsoft Visual Studio 2022/Visual Studio 2019  \
OpenCV version: 4.5.1

## Project Structure
Ensure the following files are in your directory. 

```
│   csv_util.cpp
│   csv_util.h
│   filters.cpp
│   filters.h
│   labelMeanAndStdDev.csv
│   labelToFeatures.csv
│   matchfunctions.cpp
│   matchfunctions.h
│   objectRecognitionFunctions.cpp
│   objectRecognitionFunctions.h
│   ReadME.md
│   standardDeviations.csv
│   videoObjectDetection.cpp
│   Project 3 Report.pdf
```

## How to run
1. Since IDE used is Visual Studio, there is no makefile.
2. Make sure the files above are present in the directory.
3. Import folder as an existing project into Visual Studio.
4. Configure project to include opencv directory.
5. videoObjectDetection.cpp has main function.
6. The filters.h has all the filter function declarations. The filters.cpp has all the filter function definitions. 
7. The matchfunctions.h has all the feature extraction and matching function declarations. The matchfunctions.cpp has all the feature extraction and matching function definitions. 
8. The objectRecognitionFunctions.h has all the object recognition functions declarations. The objectRecognitionFunctions.cpp has all the object recognition functions definitions. 
9. Alternatively, you can write your own Makefile.
10. If you are using a makefile, you will need to add filters.o and matchfunctions.o to the list of files for the program rule. 
10. Run readfiles for the tasks & extensions.

## How to use Program
Run videoObjectDetection for the tasks & extensions.

### Running videoObjectDetection.cpp
1. On running readfiles.cpp a window will pop-up with a GUI to open a new file and buttons to toggle different matching functions(Extension 1)
2. Click on the `Open Training File` button to open a new training file and assign a label to it (Task 5 and Extension 1).
3. Press `1` or click on `T1` button to run Task 1.
4. Press `2` or click on `T2` button to run Task 2.
5. Press `3` or click on `T3` button to run Task 3.
6. Press `4` or click on `T4` button to run Task 4.
7. Press `n` or click on `T5` or `Open Training File` button to run Task 5 (and Extension 1)
8. You will see a new window with object identified and prompt for a label.
9. Type label and hit 'Enter' to save label. Hit 'Esc' to cancel operation
10. Press `6` or click on `T6` button to run Task 6.
11. Press `7` or click on `T7` button to run Task 7.
12. Press `q` or close the main window to exit the program.
13. Enter the same key to toggle operation on or off.

## Extensions
### 1. Graphical User Interface (GUI)

### Other extensions described in report

### 2. Save video to file
1. Press `v` to start saving video to file and press `v` again to stop recording. (Extension 5)

### 3. Save Image to file
1. Press `s` to save image to file. (Extension 6)
2. A preview of the saved image will popup in a new window.
3. Press any key to close this preview and resume regular program execution.

## Limitations
1. If the video recording is started in color mode it will only capture color filter effects in the video. Similarly, if the video is started in a grayscale effect the whole video will show up as greyscale.
2. GUI File picker is Windows only.
