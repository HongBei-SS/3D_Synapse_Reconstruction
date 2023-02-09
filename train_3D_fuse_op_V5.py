import os
import model_fuse_op_V5 as modellib
from SynUtils import SynapseConfig, SynapseDataset
# Root directory of the project
ROOT_DIR = os.getcwd()

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs_SNEMI")

# Path to COCO trained weights
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

config = SynapseConfig()
config.display()

dataset_train = SynapseDataset()
dataset_train.load_infos('../SNEMI/512synapse/trainV2/image/', '../SNEMI/512synapse/trainV2/label/')
dataset_train.prepare()

# Validation dataset
dataset_val = SynapseDataset()
dataset_val.load_infos('../SNEMI/512synapse/valV2/image/', '../SNEMI/512synapse/valV2/label/')
dataset_val.prepare()

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'  #'0, 1, 3'
model = modellib.MaskRCNN(mode="training", config=config,
                          model_dir=MODEL_DIR)
# #
init_with = "last"  # imagenet, coco, or last
#
if init_with == "imagenet":
   model.load_weights(model.get_imagenet_weights(), by_name=True)
elif init_with == "coco":
   # Load weights trained on MS COCO, but skip layers that
   # are different due to the different number of classes
   # See README for instructions to download the COCO weights
   model.load_weights(COCO_MODEL_PATH, by_name=True,
                      exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",
                               "mrcnn_bbox", "mrcnn_mask"])
elif init_with == "last":
   # Load the last model you trained and continue training
   # model.load_weights(model.find_last()[1], by_name=True, exclude=["mrcnn_mask_conv3","mrcnn_mask_bn3", "mrcnn_mask_conv4","mrcnn_mask_bn4"])
   model.load_weights('./logs_SNEMI/syn20220619T2301/mask_rcnn_syn_0088.h5')
   # model.load_weights('./logs_SNEMI/syn20220619T2301/mask_rcnn_syn_0057.h5',
   #                    exclude=["P2_fuse_op","P3_fuse_op","P4_fuse_op","P2_se_add","P3_se_add", "P4_se_add","semantic_featureP3_","semantic_featureP4_","semantic_featureP2_"])
   #                    # exclude=["P2_se_add","P3_se_add", "P4_se_add","semantic_featureP3_","semantic_featureP4_","semantic_featureP2_",
   #                    #          'semantic_featureP2','semantic_featureP3','semantic_featureP4',"fpn_p2","fpn_p3","fpn_p4"])

# Train the head branches
# Passing layers="heads" freezes all layers except the head
# layers. You can also pass a regular expression to select
# which layers to train by name pattern.
model.train(dataset_train, dataset_val,
            learning_rate=config.LEARNING_RATE,
            epochs=35,
            layers="all")
model.train(dataset_train, dataset_val,
            learning_rate=config.LEARNING_RATE / 5,
            epochs=65,
            layers='all')
model.train(dataset_train, dataset_val,
            learning_rate=config.LEARNING_RATE / 5,
            epochs=101,
            layers="heads")

print("Done Training!")
