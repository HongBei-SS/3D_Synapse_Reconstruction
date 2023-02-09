import os
import numpy as np
import skimage
from config import Config
import utils

####配置
class SynapseConfig(Config):

    # Give the configuration a recognizable name
    NAME = "syn"

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 2

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # background + synapse

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    # IMAGE_MIN_DIM = 256 #height
    # IMAGE_MAX_DIM = 256 #width
    ######liuj for 3D
    IMAGE_HEIGHT = 512
    IMAGE_WIDTH = 512
    IMAGE_CHANNEL = 20

    IMAGE_MIN_DIM = 800
    IMAGE_MAX_DIM = 512

    down_sample_factor = 1
    BACKBONE_STRIDES = [4, 8, 16, 32, 64]

    RPN_ANCHOR_STRIDE = (2, 2, 2)
    RPN_ANCHOR_SCALES = ((32, 6), (64, 8), (128, 12))

    # Use smaller anchors because our image and objects are small
    # RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)  # anchor side in pixels
    # RPN_ANCHOR_SCALES = (16, 32, 64, 128, 256)
    # RPN_ANCHOR_RATIOS = [0.5, 1, 2]
    RPN_ANCHOR_RATIOS_3D = [0.5, 1, 2]
    # RPN_ANCHOR_RATIOS_3D = [1]

    RPN_NMS_THRESHOLD = 0.5
    DETECTION_MIN_CONFIDENCE = 0.80 ## proposal to rcnn

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    # TRAIN_ROIS_PER_IMAGE = 512
    TRAIN_ROIS_PER_IMAGE = 64 #256

    # How many anchors per image to use for RPN training
    RPN_TRAIN_ANCHORS_PER_IMAGE = 128 #512

    # ROIs kept after non-maximum supression (training and inference)
    POST_NMS_ROIS_TRAINING = 1000#512
    POST_NMS_ROIS_INFERENCE = 500#256

    # Maximum number of ground truth instances to use in one image
    MAX_GT_INSTANCES = 32

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 600#600

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 100


class SynapseDataset(utils.Dataset):

    def load_infos(self, imagepath, maskpath):
        """Generate the requested number of synthetic images.
        count: number of images to generate.
        height, width: the size of the generated images.
        """
        # Add classes
        self.add_class("ultrastructure", 1, "syn")
        files = os.listdir(maskpath)
        for (i,file) in enumerate(files):
            self.add_image("ultrastructure", image_id=i, path=os.path.join(imagepath,file),
                           maskpath=os.path.join(maskpath,file))

    def image_reference(self, image_id):
        """Return the shapes data of the image."""
        info = self.image_info[image_id]
        if info["source"] == "ultrastructure":
            return info["ultrastructure"]
        else:
            super(self.__class__).image_reference(self, image_id)

    def load_mask(self, image_id):
        """Generate instance masks for shapes of the given image ID.
        """
        info = self.image_info[image_id]
        mask = skimage.io.imread(info['maskpath'])
        mask = np.transpose(mask, axes=[1, 2, 0])
        # mask = mask[:, :] / 255
        # label = measure.label(mask, connectivity=2)
        labels = np.unique(mask)
        # newmask = np.zeros((mask.shape[0], mask.shape[1], mask.shape[2], labels.shape[0]-1), dtype='int32')
        newmask = []
        count = 0
        for i, label in enumerate(labels):
            if label == 0:
                continue
            temp = np.zeros(shape=(mask.shape[0], mask.shape[1], mask.shape[2]), dtype='uint8')
            temp[mask == label] = 1
            # cv2.imshow('img',temp*255)
            # cv2.waitKey(0)
            # newmask[:, :, :, i] = temp
            # print('sum:', np.sum(temp))
            if np.sum(temp) < 500:
                continue
            newmask.append(temp)
            count = count + 1
        if len(newmask) > 0:
            newmask = np.stack(newmask, axis=3)
        else:
            newmask = np.zeros(shape=(mask.shape[0], mask.shape[1], mask.shape[2], 1), dtype='uint8')
        # rgb = color.label2rgb(label)
        # cv2.imshow('label',np.reshape(rgb,(768,1024,3)))
        # cv2.waitKey(0)

        # class_ids = 1
        class_ids = np.ones(count, dtype='int32')
        return newmask, class_ids