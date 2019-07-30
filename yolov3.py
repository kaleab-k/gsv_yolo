# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 19:40:49 2019

@author: Kaleab
"""

import tensorflow as tf
import tensornets as nets
import cv2
import numpy as np
import time
import pandas as pd
import os, json
import argparse

# Debugging flag: display results if true
debug_flag = False

# Score Threshold
scoreThresh = 0.5

# size of the input layer 
width = 608
height = 608

inputs = tf.placeholder(tf.float32, [None, width, height, 3])

model = nets.YOLOv3COCO(inputs, nets.Darknet19)
#model = nets.FasterRCNN(inputs)
#model = nets.YOLOv2(inputs, nets.Darknet19)

## JSON Variables
list_json = []

## Dictionary of object classes for detection 
classes={'0':'person', '1': 'bicycle', '2':'car', '3':'motorbike', '4':'aeroplane', '5':'bus', 
        '6':'train', '7':'truck', '9':'traffic light', '10':'fire hydrant', '11':'stop sign',
        '12':'parking meter', '63':'laptop', '67':'cell phone'
        }
## List of object classes for detection
list_of_classes=[0,1,2,3,4,5,6,7,9,10,11,12,63,67]

## Setting the route dataset directory
# Default directory
dataset_dir = 'dataset/40.4589986,-3.6907383'
# Set if a 'dir' arg is provided
parser = argparse.ArgumentParser()

# add long and short argument
parser.add_argument("--dir", "-d", help="set dataset directory")
parser.add_argument("--debug", "-v", help="set debug_flag to visualize results")

# read arguments from the command line
args = parser.parse_args()
# check for --dir and --debug
if args.dir:
    dataset_dir = args.dir
if args.debug:
    debug_flag = True
## Fetch all jpegs directories inside the route directory
jpeg_dirs = [(pos_dir).replace("-jpegs", "") for pos_dir in os.listdir(dataset_dir) if pos_dir.endswith('-jpegs')]

## Function to write list_json to list_file 
def writeJSON(list_json):
    parsed_list = json.loads(pd.Series(list_json).to_json(orient='values'))
    list_file.write(json.dumps(parsed_list, indent=4))
    list_json = []

## Function to process and store in list_json
def processBBoxes(frame, boxes1, width_vid, height_vid):
    
    res_img = cv2.resize(frame,(int(width_vid),int(height_vid)))
            
    for j in list_of_classes:
        count = 0
        if str(j) in classes:
            lab = classes[str(j)]
        if len(boxes1) != 0:                
            for i in range(len(boxes1[j])):
                box = boxes1[j][i] 
                if boxes1[j][i][4] >= scoreThresh:    
                    count += 1 
                    Rx = width_vid/width
                    Ry = height_vid/height   
                    new_box = [0,0,0,0]
                    new_box[0] = int(Rx * box[0])
                    new_box[1] = int(Ry * box[1])
                    new_box[2] = int(Rx * box[2])
                    new_box[3] = int(Ry * box[3])

                    json_object = { 'seqNumber': frameCount,
                                    'class': lab,
                                    'x': new_box[0],
                                    'y': new_box[1],
                                    'width': new_box[2] - new_box[0],
                                    'height': new_box[3]- new_box[1],
                                    'confidence': boxes1[j][i][4]
                                }

                    list_json.append(json_object)
                    if debug_flag:
                        cv2.rectangle(res_img,(new_box[0], new_box[1]),(new_box[2],new_box[3]),(0,255,0),1)
                        cv2.rectangle(img,(box[0],box[1]),(box[2],box[3]),(0,255,0),1)

                        cv2.putText(img, lab, (box[0],box[1]), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 255), lineType=cv2.LINE_AA)
                        cv2.putText(res_img, lab, (new_box[0],new_box[1]), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 255), lineType=cv2.LINE_AA)
    return res_img

with tf.Session() as sess:
    ## Run the pretrained model
    sess.run(model.pretrained())
    # Iterate every jpegs directories and detect objects 
    for i in range(len(jpeg_dirs)):
        ## Set paths 
        vid_dir = jpeg_dirs[i] + '-jpegs'
        path = dataset_dir + '/' + vid_dir + '/%05d.jpg'
        cap = cv2.VideoCapture(path)
        yolo_json_fn = dataset_dir + '/' + 'yolov3/' + jpeg_dirs[i] + '_yolo'
        os.makedirs(os.path.dirname(yolo_json_fn), exist_ok=True)
        list_file = open('%s.json' % yolo_json_fn, 'w+')

        frameCount = -1
        while(cap.isOpened()):
            ret, frame = cap.read()

            if frame is None:
                break
            
            width_vid = cap.get(3)
            height_vid = cap.get(4)
            img = cv2.resize(frame,(int(width),int(height)))
            imge = np.array(img).reshape(-1,int(height),int(height),3)
            start_time = time.time()

            preds = sess.run(model.preds, {inputs: model.preprocess(imge)})
            
            frameCount += 1
            # print("--- %s seconds ---" % (time.time() - start_time)) 
            boxes = model.get_boxes(preds, imge.shape[1:3])

            if debug_flag:
                print("Frame #", frameCount)

            boxes1 = np.array(boxes)

            res_img = processBBoxes(frame, boxes1, width_vid, height_vid)

            if debug_flag:
                cv2.imshow("res_image", res_img)
                #cv2.imshow("image",img)          
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break          
        # Write JSON
        writeJSON(list_json)
        
cap.release()
cv2.destroyAllWindows()    
