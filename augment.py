import numpy as np
from skimage.transform import resize
from SynUtils import SynapseDataset, SynapseConfig
import cv2
import random
from scipy.ndimage.filters import gaussian_filter

####### grayscale
def Grayscale(data, label, random_state=np.random, contrast_factor=0.3, brightness_factor=0.3, mode='mix'):
    if mode == 'mix':
        mode = '3D' if random_state.rand() > 0.5 else '2D'
    else:
        mode = mode
    # apply augmentations
    if mode == '2D':
        data_trans = _augment2D(data, contrast_factor, brightness_factor, random_state)
    if mode == '3D':
        data_trans = _augment3D(data, contrast_factor, brightness_factor, random_state)
    return data_trans, label

def _augment2D(imgs, contrast_factor, brightness_factor, random_state=np.random):
    """
    Adapted from ELEKTRONN (http://elektronn.org/).
    """
    transformedimgs = np.copy(imgs)
    ran = random_state.rand(transformedimgs.shape[-3] * 3)

    for z in range(transformedimgs.shape[-3]):
        img = transformedimgs[z, :, :]
        img *= 1 + (ran[z * 3] - 0.5) * contrast_factor
        img += (ran[z * 3 + 1] - 0.5) * brightness_factor
        img = np.clip(img, 0, 1)
        img **= 2.0 ** (ran[z * 3 + 2] * 2 - 1)
        transformedimgs[z, :, :] = img

    data_trans = transformedimgs
    return data_trans

def _augment3D(imgs, contrast_factor, brightness_factor, random_state=np.random):
    """
    Adapted from ELEKTRONN (http://elektronn.org/).
    """
    ran = random_state.rand(3)

    transformedimgs = np.copy(imgs)
    transformedimgs *= 1 + (ran[0] - 0.5) * contrast_factor
    transformedimgs += (ran[1] - 0.5) * brightness_factor
    transformedimgs = np.clip(transformedimgs, 0, 1)
    transformedimgs **= 2.0 ** (ran[2] * 2 - 1)

    data_trans = transformedimgs
    return data_trans

####### flip
def Flip(data, label, random_state=np.random, do_ztrans=0):
    rule = random_state.randint(2, size=4 + do_ztrans)
    augmented_image = flip_and_swap(data, rule, do_ztrans)
    augmented_label = flip_and_swap(label, rule, do_ztrans)
    data_trans = augmented_image
    label_trans = augmented_label
    return data_trans, label_trans

def flip_and_swap(data, rule, do_ztrans):
    assert data.ndim == 3 or data.ndim == 4
    if data.ndim == 3:  # 3-channel input in z,y,x
        # z reflection.
        if rule[0]:
            data = data[::-1, :, :]
        # y reflection.
        if rule[1]:
            data = data[:, ::-1, :]
        # x reflection.
        if rule[2]:
            data = data[:, :, ::-1]
        # Transpose in xy.
        if rule[3]:
            data = data.transpose(0, 2, 1)
        # Transpose in xz.
        if do_ztrans == 1 and rule[4]:
            data = data.transpose(2, 1, 0)
    else:  # 4-channel input in c,z,y,x
        # z reflection.
        if rule[0]:
            data = data[:, ::-1, :, :]
        # y reflection.
        if rule[1]:
            data = data[:, :, ::-1, :]
        # x reflection.
        if rule[2]:
            data = data[:, :, :, ::-1]
        # Transpose in xy.
        if rule[3]:
            data = data.transpose(0, 1, 3, 2)
        # Transpose in xz.
        if do_ztrans == 1 and rule[4]:
            data = data.transpose(0, 3, 2, 1)
    data_trans = np.copy(data)
    return data_trans

####### rescale
def Rescale(data, label, low=0.8, high=1.2, fix_aspect=False):
    if fix_aspect:
        sf_x = random_scale(low, high)
        sf_y = sf_x
    else:
        sf_x = random_scale(low, high)
        sf_y = random_scale(low, high)

    data_trans, label_trans = apply_rescale(data, label, sf_x, sf_y)

    return data_trans, label_trans

def random_scale(low, high):
    rand_scale = np.random.rand() * (high - low) + low
    return rand_scale

def apply_rescale(image, label, sf_x, sf_y):
    # apply image and mask at the same time
    transformed_image = image.copy()
    transformed_label = label.copy()

    y_length = int(sf_y * image.shape[1])
    if y_length <= image.shape[1]:
        y0 = np.random.randint(low=0, high=image.shape[1]-y_length+1)
        y1 = y0 + y_length
        transformed_image = transformed_image[:, y0:y1, :]
        transformed_label = transformed_label[:, :, y0:y1, :]
    else:
        y0 = int(np.floor((y_length - image.shape[1]) / 2))
        y1 = int(np.ceil((y_length - image.shape[1]) / 2))
        transformed_image = np.pad(transformed_image, ((0, 0),(y0, y1),(0, 0)), mode='constant')
        transformed_label = np.pad(transformed_label, ((0, 0),(0, 0),(y0, y1),(0, 0)), mode='constant')

    x_length = int(sf_x * image.shape[2])
    if x_length <= image.shape[2]:
        x0 = np.random.randint(low=0, high=image.shape[2]-x_length+1)
        x1 = x0 + x_length
        transformed_image = transformed_image[:, :, x0:x1]
        transformed_label = transformed_label[:, :, :, x0:x1]
    else:
        x0 = int(np.floor((x_length - image.shape[2]) / 2))
        x1 = int(np.ceil((x_length - image.shape[2]) / 2))
        transformed_image = np.pad(transformed_image, ((0, 0),(0, 0),(x0, x1)), mode='constant')
        transformed_label = np.pad(transformed_label, ((0, 0),(0, 0),(0, 0),(x0, x1)), mode='constant')

    output_image = resize(transformed_image, image.shape, order=1, mode='constant', cval=0,
                          clip=True, preserve_range=True)
    # output_label = resize(transformed_label, label.shape, order=0, mode='constant', cval=0,
    #                       clip=True, preserve_range=True)
    temp_ = [resize(transformed_label[channel], image.shape, order=0, mode='constant', cval=0,
                          clip=True, preserve_range=True) for channel in range(label.shape[0])]
    output_label = np.stack(temp_, 0)
    return output_image, output_label

####### rotate
def Rotate(data, label, random_state=np.random):
    height, width = data.shape[-2:]
    M = cv2.getRotationMatrix2D((height / 2, width / 2), random_state.rand() * 360.0, 1)

    data_trans = _rotate(data, M, cv2.INTER_LINEAR)
    label_trans = _rotate(label, M, cv2.INTER_NEAREST)
    return data_trans, label_trans

def _rotate(imgs, M, interpolation):
    height, width = imgs.shape[-2:]
    transformedimgs = []
    for z in range(imgs.shape[-3]):
        if imgs.ndim == 3:
            dst = cv2.warpAffine(imgs[z], M, (height,width), 1.0, flags=interpolation, borderMode=cv2.BORDER_CONSTANT)
            transformedimgs.append(dst)
        else:
            dst = [cv2.warpAffine(imgs[channel, z], M, (height,width), 1.0,
                                  flags=interpolation, borderMode=cv2.BORDER_CONSTANT) for channel in range(imgs.shape[0])]
            transformedimgs.append(np.stack(dst, 0))
    if imgs.ndim == 3:
        transformedimgs = np.stack(transformedimgs, 0)
    else:
        transformedimgs = np.stack(transformedimgs, 1)
    return transformedimgs

####### motionBlur
def MotionBlur(data, label, sections=2, kernel_size=11):
    # generating the kernel
    kernel_motion_blur = np.zeros((kernel_size, kernel_size))
    if random.random() > 0.5:  # horizontal kernel
        kernel_motion_blur[int((kernel_size - 1) / 2), :] = np.ones(kernel_size)
    else:  # vertical kernel
        kernel_motion_blur[:, int((kernel_size - 1) / 2)] = np.ones(kernel_size)
    kernel_motion_blur = kernel_motion_blur / kernel_size

    k = min(sections, data.shape[0])
    selected_idx = np.random.choice(data.shape[0], k, replace=True)
    data_trans= np.copy(data)
    for idx in selected_idx:
        # applying the kernel to the input image
        data_trans[idx] = cv2.filter2D(data[idx], -1, kernel_motion_blur)
    return data_trans, label

####### elastic
def Elastic(data, label, alpha=10.0, sigma=4.0, random_state=np.random):
    height, width = data.shape[-2:]  # (c, z, y, x)

    dx = np.float32(gaussian_filter((random_state.rand(height, width) * 2 - 1), sigma) * alpha)
    dy = np.float32(gaussian_filter((random_state.rand(height, width) * 2 - 1), sigma) * alpha)

    x, y = np.meshgrid(np.arange(width), np.arange(height))
    mapx, mapy = np.float32(x + dx), np.float32(y + dy)

    transformed_image = []
    transformed_label = []

    for i in range(data.shape[-3]):
        if data.ndim == 3:
            transformed_image.append(cv2.remap(data[i], mapx, mapy,
                                               cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT))
        else:
            temp = [cv2.remap(data[channel, i], mapx, mapy, cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT) for channel in range(data.shape[0])]
            transformed_image.append(np.stack(temp, 0))
        if label.ndim == 3:
            transformed_label.append(
                cv2.remap(label[i], mapx, mapy, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT))
        else:
            temp_ = [cv2.remap(label[channel, i], mapx, mapy, cv2.INTER_NEAREST,
                              borderMode=cv2.BORDER_CONSTANT) for channel in range(label.shape[0])]
            transformed_label.append(np.stack(temp_, 0))

    if data.ndim == 3:  # (z,y,x)
        transformed_image = np.stack(transformed_image, 0)
    else:  # (c,z,y,x)
        transformed_image = np.stack(transformed_image, 1)
    if label.ndim == 3:  # (z,y,x)
        transformed_label = np.stack(transformed_label, 0)
    else:  # (c,z,y,x)
        transformed_label = np.stack(transformed_label, 1)

    data_trans = np.copy(transformed_image)
    label_trans = np.copy(transformed_label)

    return data_trans, label_trans

####### noise
def CutNoise(data, label, length_ratio=0.25, scale=0.2, random_state=np.random):
    zl, zh = random_region(data.shape[0], length_ratio, random_state)
    yl, yh = random_region(data.shape[1], length_ratio, random_state)
    xl, xh = random_region(data.shape[2], length_ratio, random_state)

    temp = data[zl:zh, yl:yh, xl:xh].copy()
    noise = random_state.uniform(-scale, scale, temp.shape)
    temp = temp + noise
    temp = np.clip(temp, 0, 1)
    data_trans = np.copy(data)
    data_trans[zl:zh, yl:yh, xl:xh] = temp

    return data_trans, label

def random_region(vol_len, length_ratio, random_state):
    cuboid_len = int(length_ratio * vol_len)
    low = random_state.randint(0, vol_len-cuboid_len)
    high = low + cuboid_len
    return low, high

####### misAlignment
def MisAlignment(data, label, displacement=16, random_state=np.random):
    out_shape = (data.shape[0],
                 data.shape[1] - displacement,
                 data.shape[2] - displacement)
    out_label_shape = (label.shape[0], data.shape[0],
                 data.shape[1] - displacement,
                 data.shape[2] - displacement)
    new_images = np.zeros(out_shape, data.dtype)
    new_labels = np.zeros(out_label_shape, label.dtype)

    x0 = random_state.randint(displacement)
    y0 = random_state.randint(displacement)
    x1 = random_state.randint(displacement)
    y1 = random_state.randint(displacement)
    idx = random_state.choice(np.array(range(1, out_shape[0] - 1)), 1)[0]

    if random_state.rand() < 0.5:
        # slip misalignment
        new_images = data[:, y0:y0 + out_shape[1], x0:x0 + out_shape[2]]
        new_images[idx] = data[idx, y1:y1 + out_shape[1], x1:x1 + out_shape[2]]

        new_labels = label[:, :, y0:y0 + out_shape[1], x0:x0 + out_shape[2]]
        new_labels[:, idx] = label[:, idx, y1:y1 + out_shape[1], x1:x1 + out_shape[2]]
    else:
        # translation misalignment
        new_images[:idx] = data[:idx, y0:y0 + out_shape[1], x0:x0 + out_shape[2]]
        new_images[idx:] = data[idx:, y1:y1 + out_shape[1], x1:x1 + out_shape[2]]

        new_labels[:, :idx] = label[:, :idx, y0:y0 + out_shape[1], x0:x0 + out_shape[2]]
        new_labels[:, idx:] = label[:, idx:, y1:y1 + out_shape[1], x1:x1 + out_shape[2]]
    return new_images, new_labels


from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from matplotlib.colors import Normalize
norm = Normalize(vmin=0, vmax=1, clip=False)
def show(image, cmap='gray', title='Test Title', use_norm=False):
    num_imgs = image.shape[0]
    fig = plt.figure(figsize=(20., 3.))
    fig.suptitle(title)
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                     nrows_ncols=(1, 6),  # creates 2x2 grid of axes
                     axes_pad=0.1,  # pad between axes in inch.
                     )
    image_list = np.split(image, num_imgs, 0)
    for ax, im in zip(grid, [np.squeeze(x) for x in image_list]):
        # Iterating over the grid returns the Axes.
        ax.imshow(im, cmap=cmap, norm=norm if use_norm else None,
                  interpolation='none')
        ax.axis('off')
    plt.show()

def merge_mask(data):
    data_trans = np.zeros(data.shape[-3:])
    for i in range(data.shape[0]):
        data_trans += data[i]
    return data_trans

if __name__ == '__main__':
    dataset_train = SynapseDataset()
    dataset_train.load_infos('.\\SNEMI\\train\images\\', '.\\SNEMI\\train\masks\\')
    dataset_train.prepare()
    for image_id in dataset_train.image_ids:
        image = dataset_train.load_image(image_id)  # h,w,c
        mask, class_ids = dataset_train.load_mask(image_id)
        image = np.transpose(image, axes=[2, 0, 1])
        mask = np.transpose(mask, axes=[3, 2, 0, 1])
        image = image.astype(np.float32)/255
        mask = mask.astype(np.float32)
        image_Grayscale = Grayscale(image, mask)
        image_Flip = Flip(image, mask)
        image_Rescale = Rescale(image, mask)
        image_Rotate = Rotate(image, mask)
        image_MotionBlur = MotionBlur(image, mask, sections=32)
        image_Elastic = Elastic(image, mask)
        image_CutNoise = CutNoise(image, mask, length_ratio=0.8)
        image_MisAlignment = MisAlignment(image, mask)

        show(image_Grayscale[0])
        show(image_Flip[0])
        show(image_Rescale[0])
        show(image_Rotate[0])
        show(image_MotionBlur[0])
        show(image_Elastic[0])
        show(image_CutNoise[0])
        show(image_MisAlignment[0])

        show(merge_mask(image_Grayscale[1]))
        show(merge_mask(image_Flip[1]))
        show(merge_mask(image_Rescale[1]))
        show(merge_mask(image_Rotate[1]))
        show(merge_mask(image_MotionBlur[1]))
        show(merge_mask(image_Elastic[1]))
        show(merge_mask(image_CutNoise[1]))
        show(merge_mask(image_MisAlignment[1]))
    print('pass through')