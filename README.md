# Mask R-CNN for Object Detection and Instance Segmentation
[Mask R-CNN](https://arxiv.org/abs/1703.06870), or Mask Region-based Convolutional Neural Network, is a type of deep learning algorithm used for object detection and segmentation in images. It is an extension of the Faster R-CNN object detection algorithm. 

![street](https://github.com/odumosuo/Computer_vision_python/assets/111093025/b9c9fe3b-ac33-44c4-a9a5-1c9728a7fead)


**Object Detection**: Mask R-CNN identifies different objects within an image. It doesn't just tell you what objects are in the image, but also where they are located and generates bounding boxes and labels around objects.

**Instance Segmentation**: After identifying the objects and their locations, Mask R-CNN generates pixel-level masks for each object, which precisely outline the shape of the object in the image (**Mask Generation**)

**How It Works**: Mask R-CNN breaks down the image into smaller regions, then analyzes each region to determine if it contains an object and, if so, what type of object it is and where it is located. It uses a combination of convolutional neural networks (CNNs) and a region proposal network (RPN) to accomplish this task.



## Requirements
TensorFlow 2.14, Keras 2.14, and other common packages listed in `requirements.txt`

`mrcnn` Folder contains the Mask R-CNN package needed for the dependencies in `requirements.txt`

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
* Pre-trained COCO weights (mask_rcnn_coco.h5) from the [releases page](https://github.com/matterport/Mask_RCNN/releases).

## How to Run
Follow the instructions in the multi-line docstring at the begining of the code in `coco_custom_model.py`

## Testing model
`model_testing.ipynb` This notebook contains the evaluation of the model. 

The model was tested using 'random' images gotten from google. For the purpose of this project, all images had an instance of an animal. The full range of classes/labels that are included in the MS COCO dataset can be found [here](https://cocodataset.org/#explore)

#### Example test

![dog_on_skateboard](https://github.com/odumosuo/Computer_vision_python/assets/111093025/9b093b19-9702-43c8-b413-e4e6a3f2f0d7)

![dog_on_skateboard](https://github.com/odumosuo/Computer_vision_python/assets/111093025/42563f0c-41e0-4cfd-a389-136f33de11e6)

## Example Projects Using this Mask R-CNN 

### [4K Video Demo](https://www.youtube.com/watch?v=OOT3UIXZztE) by Karol Majek.
![4K Video Demo](https://github.com/odumosuo/Computer_vision_python/blob/main/assets/4k_video.gif?raw=true)

### [Splash of Color](https://engineering.matterport.com/splash-of-color-instance-segmentation-with-mask-r-cnn-and-tensorflow-7c761e238b46). A blog post explaining how to train this model from scratch and use it to implement a color splash effect.
![balloon_color_splash](https://github.com/odumosuo/Computer_vision_python/assets/111093025/4798539b-de57-4176-affd-44add2067aeb)


### [Segmenting Nuclei in Microscopy Images](samples/nucleus). Built for the [2018 Data Science Bowl](https://www.kaggle.com/c/data-science-bowl-2018)
Code is in the `samples/nucleus` directory.

![nucleus_segmentation](https://github.com/odumosuo/Computer_vision_python/assets/111093025/26111abc-56dd-4f74-bb01-b9d2b9d1668f)
