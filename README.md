## Google Street View (GSV) Object Detection 
YOLOv3 based object detection on Google Street View images extracted via https://github.com/pcarballeira/gsv_database_creator
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


    
**Database Structure:**

Executing the script will generate a JSON with detection details. For every jpeg folders, it will create a JSON file with the same name to the **jpegs** folders except the **_jpegs** rather appending **_yolo** under a folder named **yolov3**. For instance, if the jpegs folder name is **M:DRIVING_S:608x608-jpegs**, the JSON file name will be *-**M:DRIVING_S:608x608_yolo.json**. In the JSON file, we will have a root _JSON Array_ containing _JSON Object_ for each bounding box detection store the details as a key-value pair. These details are:

- **_seqNumber:_** &nbsp; an integer value of a number between 00000 and 99999 that will identify the image.
- **_x:_**  &nbsp;         the top left starting x-position of the detected bounding box. 
- **_y:_**  &nbsp;           the top left starting y-position of the detected bounding box. 
- **_width:_** &nbsp;      the width of the detected bounding box. 
- **_height:_** &nbsp;     the height of the detected bounding box. 
- **_class:_**  &nbsp;     the class classification of the detected bounding box. 
- **_confidence:_** &nbsp;  the detection confidence. 

