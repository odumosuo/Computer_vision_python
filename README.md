# Mask R-CNN for Object Detection and Instance Segmentation
Using Mask R-CNN to create bounding boxes and segmentation masks for each instance of an object in the image.


## Requirements
TensorFlow 2.14, Keras 2.14, and other common packages listed in `requirements.txt`

Install dependencies
   ```bash
   pip3 install -r requirements.txt
   ```
### MS COCO Requirements:
To train or test on MS COCO, you'll also need:
* pycocotools - Install `pycocotools` from one of these repos. They are forks of the original pycocotools with fixes for Python3 and Windows (the official repo doesn't seem to be active anymore).

    * Linux: https://github.com/waleedka/coco
    * Windows: https://github.com/philferriere/cocoapi.
    You must have the Visual C++ 2015 build tools on your path (see the repo for additional details)
* Download 2017 training and validation image subsets along with their annotations [MS COCO Dataset](http://cocodataset.org/#home)

