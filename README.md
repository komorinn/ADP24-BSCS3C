# CC105 App Dev Project

## Needed Prerequisites for this project
```
python 3.12.7 (atleast 3.6 to 3.12 can work with dlib)
cmake
dlib
face_recognition
opencv
imutils
```

## Installation
Install these files in the environment:
- https://www.python.org/downloads/release/python-3127/
- https://cmake.org/download/
- ``` pip install dlib ```
- ``` pip install face_recognition ```
- ``` pip install opencv ```
- ``` pip install imutils ```
- ``` curl -o haarcascade_frontalface_default.xml https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml ```

## Checking for correct requirements

Check each prerequisite before continuing to open the project

Check the Python version (must be lower than 3.12 and not higher than 3.6):
```
python --version
```

Check the cmake is installed:
```
cmake --version
```

Check the dlib is installed:
```
pip show dlib
```

Check the face_recognition is installed
```
pip show face_recognition
```

Check the opencv is installed
```
pip show opencv
```

Check the imutils is installed
```
pip show imutils
```

Get Haar Cascade File 
```
curl -o haarcascade_frontalface_default.xml https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml
```

Create Directory Structure (the photos goes here)
```
mkdir -p Dataset/Face_Capture
```

Run the program using:
```
python face_capture.py
```
If you get webcam errors, try changing the camera index (switching cameras, try different numbers (0, 1, 2) if 0 doesn't work):
```
cap = cv2.VideoCapture(1) 


