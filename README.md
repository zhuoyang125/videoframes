## Annotating Bounding Boxes from Video Clips

This repo will allow annotation of bounding boxes of a specific object class, saving the labels into a ```result.json``` file. This is performed with [Mask RCNN](https://github.com/matterport/Mask_RCNN) by matterport. 

### How to Run

Get the arguments needed with:
```
python video_frames.py --help
```
Requires a video file to detect, frame rate, and model path, and weights path. Trained weights can be downloaded here: https://drive.google.com/open?id=1garXSmOFSa6XNzVRNoSRCbHTS4Cl-4B8

Before running the script, create two folders: `images` and `annotated_images`. `images` contains the image frames of the video with the labels in the json file are based upon. `annotated_images` includes the mask and bounding boxes to check the quality of the annotations.

The class ids are based on the MSCOCO dataset. Change the class by editing `category_id`. (Default is '1' for 'people' class)
