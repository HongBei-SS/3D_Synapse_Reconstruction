import os
import math
import re
import numpy as np
import cv2

from config import Config
import utils
import model_fuse_op_V5 as modellib
from pairwise_match_lj import pair_match

from SynUtils import SynapseConfig, SynapseDataset

def merge(box1, box2):
    (min_row1, min_col1, max_row1, max_col1) = box1
    (min_row2, min_col2, max_row2, max_col2) = box2
    min_row = min(min_row1, min_row2)
    min_col = min(min_col1, min_col2)
    max_row = max(max_row1, max_row2)
    max_col = max(max_col1, max_col2)
    return (min_row, min_col, max_row, max_col)

# Root directory of the project
ROOT_DIR = os.getcwd()

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs_SEM")

# Path to COCO trained weights
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

config = SynapseConfig()
config.display()


class InferenceConfig(SynapseConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    DETECTION_MIN_CONFIDENCE = 0.95

inference_config = InferenceConfig()

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# Recreate the model in inference mode
model = modellib.MaskRCNN(mode="inference",
                          config=inference_config,
                          model_dir=MODEL_DIR)

# model_path = model.find_last()[1]
model_path = './logs_SEM/syn20220701T1910/mask_rcnn_syn_0060.h5'
# Load trained weights (fill in path to trained weights here)
assert model_path != "", "Provide path to trained weights"
print("Loading weights from ", model_path)
model.load_weights(model_path, by_name=True)

if __name__ == '__main__':
    scale = 1
    thresholds = [0.90, 0.95, 0.99, 0.50,0.55,0.60,0.65,0.70,0.75,0.80, 0.85]
    for threshold in thresholds:
        inference_config.DETECTION_MIN_CONFIDENCE = threshold
        print('inference at detection threshold: ', threshold)
        ImagePath = '../SEM/test/raw_test/'
        MaskPath = 'predict_big/3D_fuse_op_V5_temp/'+str(threshold)
        # cell_path = 'Y:\cochlea\Aligned_data\crop_cell2_fiji\\trakem2.1591008294190.2101327853.3322066\proofreading'

        if not os.path.exists(MaskPath):
            os.mkdir(MaskPath)

        files = os.listdir(ImagePath)
        files.sort(key=lambda i: int(re.match(r'(\d+)', i).group()))
        img = []
        for file in files:
            img_temp = cv2.imread(os.path.join(ImagePath, file), cv2.CV_8UC1)
            img_temp = cv2.resize(img_temp, (int(img_temp.shape[1]*scale),int(img_temp.shape[0]*scale)),cv2.INTER_LINEAR)
            img_temp = img_temp/255
            img.append(img_temp)
        imgs = np.stack(img, axis=0)
        height = imgs.shape[1]
        width = imgs.shape[2]
        channel = imgs.shape[0]

        crop_height = inference_config.IMAGE_HEIGHT
        crop_width = inference_config.IMAGE_WIDTH
        crop_channel = inference_config.IMAGE_CHANNEL
        step_xy = 256
        step_z = 10
        i_count = math.ceil((height - crop_height) / step_xy) - 1
        j_count = math.ceil((width - crop_width) / step_xy) - 1
        t_count = math.ceil((channel - crop_channel) / step_z) - 1
        count = 0
        label_start = 1
        for i in range(i_count + 2):
            for j in range(j_count + 2):
                for t in range(t_count + 2):
                    #x
                    if i < i_count + 1:
                        start_i = i * step_xy
                        halo_size_i = crop_height - step_xy
                    else:
                        start_i = height - crop_height
                        halo_size_i = (i - 1) * step_xy + crop_height - start_i
                    #y
                    if j < j_count + 1:
                        start_j = j * step_xy
                        halo_size_j = crop_width-step_xy
                    else:
                        start_j = width - crop_width
                        halo_size_j = (j - 1) * step_xy + crop_width - start_j
                    #z
                    if t < t_count + 1:
                        start_z = t * step_z
                        crop_img = imgs[t * step_z:t * step_z + crop_channel, start_i:start_i + crop_height, start_j:start_j + crop_width]
                        halo_size_t = crop_channel-step_z
                    else:
                        start_z = channel - crop_channel
                        crop_img = imgs[channel - crop_channel:channel, start_i:start_i + crop_height, start_j:start_j + crop_width]
                        halo_size_t = (t-1) * step_z + crop_channel-start_z
                    count += 1
                    crop_img = np.transpose(crop_img, [1,2,0])[:, :, :, np.newaxis].astype('float32')
                    results = model.detect([crop_img], verbose=1)
                    r = results[0]
                    masks = r['masks']
                    mask = np.zeros(shape=(crop_height, crop_width, crop_channel), dtype='uint16')

                    if masks.shape[0] == crop_height:
                        for ttt in range(masks.shape[3]):
                            mask[masks[:, :, :, ttt] > 0] = label_start
                            label_start += 1
                    mask = np.transpose(mask, [2, 0, 1])
                    if t == 0:
                        out = mask
                    else:
                        out = pair_match(out, mask, direction=1, halo_size=halo_size_t)
                if j == 0:
                    out_z = out
                else:
                    out_z = pair_match(out_z, out, direction=3, halo_size=halo_size_j)
            if i == 0:
                out_x = out_z
            else:
                out_x = pair_match(out_x, out_z, direction=2, halo_size=halo_size_i)
        # stitch_mask = out_x
        ###save mask
        # stitch_mask = cv2.resize(stitch_mask, (width_ori, height_ori), interpolation=cv2.INTER_NEAREST)
        for tt in range(out_x.shape[0]):
            stitch_mask = cv2.resize(out_x[tt], (width*1, height*1), interpolation=cv2.INTER_NEAREST)
            cv2.imwrite(MaskPath + '/' + str(tt+1).zfill(4) + '.png', stitch_mask)