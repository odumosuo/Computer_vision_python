# Mask R-CNN for Object Detection and Instance Segmentation
[Mask R-CNN](https://arxiv.org/abs/1703.06870), or Mask Region-based Convolutional Neural Network, is a type of deep learning algorithm used for object detection and segmentation in images. It is an extension of the Faster R-CNN object detection algorithm. 

**Object Detection**: Mask R-CNN identifies different objects within an image. It doesn't just tell you what objects are in the image, but also where they are located and generates bounding boxes and labels around objects.

**Instance Segmentation**: After identifying the objects and their locations, Mask R-CNN generates pixel-level masks for each object, which precisely outline the shape of the object in the image **Mask Generation**

   **How It Works**: Mask R-CNN breaks down the image into smaller regions, then analyzes each region to determine if it contains an object and, if so, what type of object it is and where it is located. It uses a combination of convolutional neural networks (CNNs) and a region proposal network (RPN) to accomplish this task.



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

