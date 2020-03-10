'''
Uses Mask R-CNN to annotate 'people' classes from image frames of a videoclip. 
Outputs image frames, json file, and image masks for verification.
Object classes and class ids based on MSCOCO
'''
import os
import cv2
import sys
import json
import math
import random
import numpy as np 
import skimage.io 
import argparse
import matplotlib 
from mrcnn import utils
from mrcnn import visualize
import mrcnn.model as modellib
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser()
parser.add_argument("--video_file", help="video file directory", required=True)
parser.add_argument("--frame_rate", help="each image captured every _s", default=0.5)
parser.add_argument("--model_path", help="model directory")
parser.add_argument("--weights_path", help="model weights path")
args = parser.parse_args()

# Root directory of the project
ROOT_DIR = os.getcwd()
#import coco config
sys.path.append(os.path.join(ROOT_DIR, "mrcnn"))
from config import Config

#function to get bounding box coordinates
def bbox_coords(class_array, box_array):
    locs = np.asarray(np.where(class_array == category_id)) #checks positions of element '1'
    len = np.size(locs) #no elements in locs
    classes = np.ones(len)
    bbox = box_array[locs,:] #finds corresponding bounding box dims 
    return bbox, classes    

#function to generate frames from the video
def frames_frm_video(sec):
    vidcap = cv2.VideoCapture(args.video_file)
    vidcap.set(cv2.CAP_PROP_POS_MSEC,sec*1000)
    hasFrames, image = vidcap.read()

    return hasFrames, image

#create config class
class InferenceConfig(Config):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 81
    NAME = 'inference'

config = InferenceConfig()

#create model object
model = modellib.MaskRCNN(mode='inference', model_dir=args.model_path, config=config)
model.load_weights(args.weights_path, by_name=True)


if __name__ == "__main__":
    #create a dictionary to store json stuff
    #'images' : filename, height, width, id
    #'type' : 'instances'
    #'annotations': imageid, bbox, category id, id, ignore = 0

    
    image_height = 720
    image_width = 1280
    filename = 2020000001
    category_id = 1 #class for 'people'
    id_val = 1

    #Class Names
    class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
                'bus', 'train', 'truck', 'boat', 'traffic light',
                'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
                'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
                'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
                'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
                'kite', 'baseball bat', 'baseball glove', 'skateboard',
                'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
                'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
                'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
                'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
                'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
                'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
                'teddy bear', 'hair drier', 'toothbrush']

    img_dict = {}
    img_dict['images'] = []
    img_dict['annotations'] = []
    img_dict['categories'] = []
 
    #run detection on frames
    sec = 0
    frame_rate = int(args.frame_rate)
    count = 1
    success, image = frames_frm_video(sec)
    results = model.detect([image], verbose=1)
    r = results[0]
    bbox, classes = bbox_coords(r['class_ids'], r['rois'])
    print(bbox)

    img_dict['images'].append({
            'file_name': str(filename) + '.jpg',
            'height': str(image_height),
            'width': str(image_width),
            'id': str(filename)
        })

    if success:
        #save orig image
        cv2.imwrite('images/' + str(filename) + '.jpg', image)
        #save image with mask
        mask_file_no = 1
        visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], 
                                class_names, r['scores'])
        plt.savefig('annotated_images/annotated{}.jpg'.format(mask_file_no))
        mask_file_no += 1

    for i in bbox:
        for j in i:
        #add to dictionary
            row_list = j.tolist() #converts a numpy array to list with commas
            print(row_list)
            img_dict['annotations'].append({
                'imageid': str(filename),
                'bbox': str(row_list),
                'category_id': str(category_id),
                'id': str(id_val),
            })
            id_val += 1
    filename += 1

    while success:
        count += 1
        sec = sec + frame_rate
        sec = round(sec,2)
        success, image = frames_frm_video(sec)
        if success:
            #save orig image
            cv2.imwrite('images/' + str(filename) + '.jpg', image)
            results = model.detect([image], verbose=1)
            r = results[0]

            #save image with mask
            visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], 
                                class_names, r['scores'])
            plt.savefig('annotated_images/annotated{}.jpg'.format(mask_file_no))
            mask_file_no += 1

            bbox, classes = bbox_coords(r['class_ids'], r['rois'])

            img_dict['images'].append({
                    'file_name': str(filename) + '.jpg',
                    'height': str(image_height),
                    'width': str(image_width),
                    'id': str(filename)
                })
                
            for i in bbox:
                for j in i:
                #add to dictionary
                    row_list = j.tolist() #converts a numpy array to list with commas
                    img_dict['annotations'].append({
                        'imageid': str(filename),
                        'bbox': str(row_list),
                        'category_id': str(category_id),
                        'id': str(id_val),
                    })
                    id_val += 1
            filename += 1
        else:
            break
    
    id = 0
    for item in class_names:
        img_dict['categories'].append({
            'supercategory': 'none',
            'id': id,
            'name': item
        })
        id += 1
    with open('result.json', 'w') as fp:
        json.dump(img_dict, fp)
        



