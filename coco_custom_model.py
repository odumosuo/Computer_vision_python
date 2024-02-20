"""
MS COCO instance segmentation with bounding box

The code uses Mask R-CNN for detecting images in the MS COCO dataset

Usage:  Follow instruction in read me 
        Change directories on line 27, 35, 38, and 40 to root directory of project, name of pretrained weights file in root directory, name of folder to hold models, and directory of images respectively. 
        run python3 coco_custom_model.py
       
Author: Segun Odumosu
"""

import os
import sys
import numpy as np
import imgaug  

# See README to download and install the Python COCO tools. 
# Linux: https://github.com/waleedka/coco
# Windows: https://github.com/philferriere/cocoapi. 
from pycocotools.coco import COCO
from pycocotools import mask as maskUtils


# Root directory of the project
# Adjust accodringly #
ROOT_DIR = os.path.abspath("C:/Users/segsi/OneDrive/Documents/UOGuelph/coursework/Dan_project/Project") 

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

# Directory to COCO Data set
COCO_DIR = "D:/Coco_Dataset"

############################################################
#  Configurations
############################################################


class CocoConfig(Config):
    """Configuration for training on MS COCO.
    
    Check config.py in mrcnn package folder for all default values 
    
    Derives from the base Config class and overrides values specific
    to the COCO dataset.
    """
    # Give the configuration a recognizable name
    NAME = "coco"

    # Number of images to train with on each GPU. A 12GB GPU can typically
    # handle 2 images of 1024x1024px.
    # Adjust based on your GPU memory and image sizes. Use the highest
    # number that your GPU can handle for best performance.
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 80  # COCO has 80 classes


############################################################
#  Dataset
############################################################

class CocoDataset(utils.Dataset):
    def load_coco(self, dataset_dir, subset):
        """Load a subset of the COCO dataset.
        dataset_dir: The root directory of the COCO dataset.
        subset: What to load (train, val, minival, valminusminival)
        """

        coco = COCO("{}/annotations/instances_{}2017.json".format(dataset_dir, subset))
        image_dir = "{}/{}2017".format(dataset_dir, subset)

        # Get all classIDs
        class_ids = sorted(coco.getCatIds())

        # Get all imagesIDs
        image_ids = list(coco.imgs.keys())
        print("Retrieved class and image IDs")

        # Add classes
        print("Adding classes...")
        for i in class_ids:
            self.add_class("coco", i, coco.loadCats(i)[0]["name"])

        # Add images
        print("Adding images...")
        for i in image_ids:
            IMAGE_PATH = os.path.join(image_dir, coco.imgs[i]['file_name'])
            if os.path.exists(IMAGE_PATH): # Check if the file exists. Seems like some files were corrupt when I downloaded them
                self.add_image(
                    "coco", image_id=i,
                    path= IMAGE_PATH,
                    width=coco.imgs[i]["width"],
                    height=coco.imgs[i]["height"],
                    annotations=coco.loadAnns(coco.getAnnIds(
                        imgIds=[i], catIds=class_ids, iscrowd=None)))
        print("Finished adding classes and images")
        


    def load_mask(self, image_id):
        """Load instance masks for the given image.
        
        This function is called by the data_generator when the train method is called on the model class. Check model.py in mrcnn package folder

        Different datasets use different ways to store masks. This
        function converts the different mask format to one format
        in the form of a bitmap [height, width, instances].

        Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        print("Loading masks...")
        # If not a COCO image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "coco":
            return super(CocoDataset, self).load_mask(image_id)

        instance_masks = []
        class_ids = []
        annotations = self.image_info[image_id]["annotations"]
        # Build mask of shape [height, width, instance_count] and list
        # of class IDs that correspond to each channel of the mask.
        for annotation in annotations:
            class_id = self.map_source_class_id(
                "coco.{}".format(annotation['category_id']))
            if class_id:
                m = self.annToMask(annotation, image_info["height"],
                                   image_info["width"])
                # Some objects are so small that they're less than 1 pixel area
                # and end up rounded out. Skip those objects.
                if m.max() < 1:
                    continue
                # Is it a crowd? If so, use a negative class ID.
                if annotation['iscrowd']:
                    # Use negative class ID for crowds
                    class_id *= -1
                    # For crowd masks, annToMask() sometimes returns a mask
                    # smaller than the given dimensions. If so, resize it.
                    if m.shape[0] != image_info["height"] or m.shape[1] != image_info["width"]:
                        m = np.ones([image_info["height"], image_info["width"]], dtype=bool)
                instance_masks.append(m)
                class_ids.append(class_id)

        # Pack instance masks into an array
        if class_ids:
            mask = np.stack(instance_masks, axis=2).astype(np.bool)
            class_ids = np.array(class_ids, dtype=np.int32)
            return mask, class_ids
        else:
            # Call super class to return an empty mask
            return super(CocoDataset, self).load_mask(image_id)

    def image_reference(self, image_id): 
        """Return a link to the image in the COCO Website.
        Useful when running in inference mode
        """
        info = self.image_info[image_id]
        if info["source"] == "coco":
            return "http://cocodataset.org/#explore?id={}".format(info["id"])
        else:
            super(CocoDataset, self).image_reference(image_id)

    # The following two functions are from pycocotools with a few changes.

    def annToRLE(self, ann, height, width):
        """
        Convert annotation which can be polygons, uncompressed RLE to RLE.
        :return: binary mask (numpy 2D array)
        """
        segm = ann['segmentation']
        if isinstance(segm, list):
            # polygon -- a single object might consist of multiple parts
            # we merge all parts into one mask rle code
            rles = maskUtils.frPyObjects(segm, height, width)
            rle = maskUtils.merge(rles)
        elif isinstance(segm['counts'], list):
            # uncompressed RLE
            rle = maskUtils.frPyObjects(segm, height, width)
        else:
            # rle
            rle = ann['segmentation']
        return rle

    def annToMask(self, ann, height, width):
        """
        Convert annotation which can be polygons, uncompressed RLE, or RLE to binary mask.
        :return: binary mask (numpy 2D array)
        """
        rle = self.annToRLE(ann, height, width)
        m = maskUtils.decode(rle)
        return m




############################################################
#  Training
############################################################

if __name__ == '__main__': # Check to see if the whole script/file is being run as the main program or being imported as a module. Useful to load classes for training/inference in other programs/files/scripts
    # Configurations
    config = CocoConfig()
    # config.display()

    # Create model
    model = modellib.MaskRCNN(mode="training", config=config,
                                    model_dir=DEFAULT_LOGS_DIR)


    # Select weights file to load
    weights_path = COCO_WEIGHTS_PATH

    # Load weights
    print("Loading weights ", weights_path)
    model.load_weights(weights_path, by_name=True)

    # Train model
    dataset_train = CocoDataset()
    print("---------------------------------------Loading training data---------------------------------------")
    dataset_train.load_coco(COCO_DIR, "train")
    print("-------------------------------Finished loading training data... Now preparing training data--------------------------------")
    dataset_train.prepare()
    print("---------------------------------------Finished preparing training data---------------------------------------")

    # Validation dataset
    dataset_val = CocoDataset()
    print("---------------------------------------Loading validation data---------------------------------------")
    dataset_val.load_coco(COCO_DIR, "val")
    print("-------------------------------Finished loading validation data...Now preparing validation data----------------------------")
    dataset_val.prepare()
    print("---------------------------------------Finished preparing validation data---------------------------------------")

    # Image Augmentation
    # Right/Left flip 50% of the time
    augmentation = imgaug.augmenters.Fliplr(0.5)

    # Training model
    ## Because we are training from pretrained COCO weights, we will only train the network heads.
    print("---------------------------------------Training network heads---------------------------------------")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=300,
                layers='heads',
                augmentation=augmentation)

    print("---------------------------------------Done training model, have fun!---------------------------------------")

