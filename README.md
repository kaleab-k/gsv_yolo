## Google Street View (GSV) Object Detection 
YOLOv3 based object detection on Google Street View images extracted via https://github.com/pcarballeira/gsv_database_creator

comment [pcl]: 
- if possible, separate the code in modules/functions. Input (read from GSV database), Process (may be changed for something else), Output (output detections in a JSON format that should be common even if the detector is changed)

# Requirements
- Python distro (*Conda* recommended: https://www.anaconda.com/distribution)
- Tensorflow 
- OpenCV
- Numpy

# Dependencies
This code uses TensorNet library (https://github.com/taehoonlee/tensornets)

# Installation and compilation
## Setting up conda environment
Create your virtual environment

    conda create -n yourenvname
  
Activate your environment 

    source activate yourenvname

comment [pcl]: conda activate yourenvname???

Install packages

    conda install tensorflow (or conda install tensorflow-gpu)
    pip install tensornets 
    conda install opencv
    conda install -c conda-forge numpy 

comments [pcl]: 
- add installation of cython? didn't work otherwise
- Had problems with numpy (ValueError: numpy.ufunc size changed, may indicate binary incompatibility) Solved them installing numpy 1.16.1. Investigate this. It has relation with incompatibilities with cython
- Add installation of pandas
- Use conda installation of possible, for numpy for example

# Compiling the program

Set the directory variable to the path where the images and the associated JSON is stored.

    dataset_dir = "..."
    
and execute the script
    
    python yolov3.py
    
 
comments [pcl]: 

- I get an error when showing images: cv2.error: OpenCV(3.4.2) /tmp/build/80754af9/opencv-suite_1535558553474/work/modules/highgui/src/window.cpp:632: error: (-2:Unspecified error) The function is not implemented. Rebuild the library with Windows, GTK+ 2.x or Carbon support. If you are on Ubuntu or Debian, install libgtk2.0-dev and pkg-config, then re-run cmake or configure script in function 'cvShowImage'

- Visualization errors may have relationship with virtual machine (not sure). Anyway, enable a visualization flag to show/not show detections

- change code to add directory through command line option, so code does not need to be modified

- add folders with a small gsv database (a few images) so code can be tested, even without having used the database extractor

- Enable the visualization flag also through command line option

# Structure of detections:

Detections are stored in a JSON file with the same name of the images folder, where the **_jpegs_** suffix is substituted by **_yolo_**. For instance, if the images folder name is **M:DRIVING_S:608x608-jpegs**, the JSON file name will be **M:DRIVING_S:608x608_yolo.json**. The JSON file contains a root _JSON Array_ with a _JSON Object_ for each detected bounding box, with the following key-value pairs: 

- **_seqNumber:_** &nbsp; an integer value of a number between 00000 and 99999 that identifies the image.
- **_x:_**  &nbsp;         the x-position of the bounding box top left corner. 
- **_y:_**  &nbsp;         the y-position of the bounding box top left corner. 
- **_width:_** &nbsp;      the bounding box width. 
- **_height:_** &nbsp;     the bounding box height. 
- **_class:_**  &nbsp;     the class of the detected objec. 
- **_confidence:_** &nbsp;  the detection confidence. 

