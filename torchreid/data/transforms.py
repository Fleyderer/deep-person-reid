from __future__ import division, print_function, absolute_import
import math
import random
from collections import deque
import torch
from PIL import Image
from torchvision.transforms import (
    Resize, Compose, ToTensor, Normalize, ColorJitter, RandomHorizontalFlip, RandomCrop
)


class AlignResizePad(torch.nn.Module):
    """
    Torch transform module for resizing and padding an image tensor to a target shape,
    aligned to the top center. Assumes input tensor is in HWC format (height, width, channels).

    Returns the transformed image along with scaling ratio and padding details.
    """

    def __init__(self, target_shape: tuple[int, int]):
        """
        Args:
            target_shape (tuple[int, int]): Target image size as (height, width).
        """
        super().__init__()
        self.target_shape = target_shape

    def forward(self, img: torch.Tensor):
        """
        Args:
            img (torch.Tensor): Image tensor in HWC format with dtype float32 or uint8.

        Returns:
            img_out (torch.Tensor): Resized and padded image in CHW format.
            ratio (tuple[float, float]): (r_w, r_h) scaling ratio.
            dwdh (tuple[int, int, int]): (dw_left, dw_right, dh) padding applied.
        """
        _, h0, w0, = img.shape
        h_target, w_target = self.target_shape

        # Compute resize ratio
        r = min(w_target / w0, h_target / h0)
        new_w, new_h = int(w0 * r), int(h0 * r)

        # Resize
        img_nchw = img.unsqueeze(0)  # (1, C, H, W)
        img_resized = torch.nn.functional.interpolate(
            img_nchw, size=(new_h, new_w), mode='bilinear', align_corners=False)
        img_resized = img_resized.squeeze(0)  # (C, H, W)

        # Compute padding
        dw_total = w_target - new_w
        dh = h_target - new_h
        dw_left = dw_total // 2
        dw_right = dw_total - dw_left

        # Pad (left, right, top, bottom)
        img_padded = torch.nn.functional.pad(img_resized, (dw_left, dw_right, 0, dh), value=0.5)

        return img_padded


class RandomCropByScaleAndRatio(torch.nn.Module):
    def __init__(self, scale=(0.08, 1.0), ratio=(3/4, 4/3), min_upper_boundary=0.0, p=0.5):
        """
        Args:
            scale (tuple of float): The minimum and maximum fraction of the original image area to be used for the crop.
            ratio (tuple of float): The minimum and maximum aspect ratio (width/height) for the crop.
            min_upper_boundary (float): A value between 0.0 and 1.0 that determines how much of the lower part 
                                      of the image should be excluded from the starting point of crops.
                                      For example:
                                      - 0.0 means crops can start from anywhere in the image (default)
                                      - 0.5 means crops can only start from the upper half of the image
                                      - 0.9 means crops can only start from the upper 90% of the image
            p (float): probability of transform
        """
        super().__init__()
        self.scale = scale
        self.ratio = ratio
        self.min_upper_boundary = min(max(0.0, min_upper_boundary), 1.0)  # Clamp between 0 and 1
        self.p = p

    def get_params(self, img):
        """
        Compute crop parameters based on the original image size, and the provided
        scale and ratio ranges.

        Args:
            img (PIL.Image): Image to be cropped.

        Returns:
            tuple: (i, j, h, w) where (i, j) are the top-left coordinates of the crop,
                   and h and w are the height and width of the crop.
        """
        width, height = img.size
        area = width * height

        # Randomly choose an aspect ratio from the given range.
        aspect_ratio = random.uniform(self.ratio[0], self.ratio[1])
        
        # Compute the maximum scale factor allowed for the chosen aspect ratio so that
        # the computed width and height do not exceed the image dimensions.
        # For width:  sqrt(s*area*aspect_ratio) <= width  => s <= width^2 / (area * aspect_ratio)
        # For height: sqrt(s*area/aspect_ratio) <= height => s <= height^2 * aspect_ratio / area
        s_max_allowed = min(self.scale[1],
                            (width ** 2) / (area * aspect_ratio),
                            (height ** 2 * aspect_ratio) / area)
        
        # If the maximum allowed scale is less than the minimum requested scale,
        # we use the maximum allowed value.
        if s_max_allowed < self.scale[0]:
            s = s_max_allowed
        else:
            s = random.uniform(self.scale[0], s_max_allowed)
        
        # With the chosen scale factor, compute the target crop area.
        target_area = s * area
        
        # Derive the width and height of the crop from the target area and the aspect ratio.
        crop_width = int(round(math.sqrt(target_area * aspect_ratio)))
        crop_height = int(round(math.sqrt(target_area / aspect_ratio)))
        
        # Ensure the computed dimensions do not exceed the original image dimensions.
        crop_width = min(crop_width, width)
        crop_height = min(crop_height, height)
        
        # Randomly select the top-left coordinate for the crop, considering min_upper_boundary
        if width - crop_width > 0:
            j = random.randint(0, width - crop_width)
        else:
            j = 0
            
        # Calculate the maximum allowed i value based on min_upper_boundary
        # When min_upper_boundary = 0.9, we want i to be in the top 90% of the image
        # When min_upper_boundary = 0, we can use the full height as before
        max_i_value = int((1.0 - self.min_upper_boundary) * height)
        
        # Ensure that max_i_value + crop_height doesn't exceed the image height
        max_i_allowed = min(max_i_value, height - crop_height)
        
        if max_i_allowed > 0:
            i = random.randint(0, max_i_allowed)
        else:
            i = 0
            
        return i, j, crop_height, crop_width

    def forward(self, img):
        """
        Crop the image using the randomly computed parameters.
        
        Args:
            img (PIL.Image): Image to be cropped.
            
        Returns:
            PIL.Image: The cropped region of the image.
        """
        if random.random() > self.p:
            return img
        i, j, h, w = self.get_params(img)
        # PIL's crop takes (left, upper, right, lower)
        return img.crop((j, i, j + w, i + h))


class Random2DTranslation(object):
    """Randomly translates the input image with a probability.

    Specifically, given a predefined shape (height, width), the input is first
    resized with a factor of 1.125, leading to (height*1.125, width*1.125), then
    a random crop is performed. Such operation is done with a probability.

    Args:
        height (int): target image height.
        width (int): target image width.
        p (float, optional): probability that this operation takes place.
            Default is 0.5.
        interpolation (int, optional): desired interpolation. Default is
            ``PIL.Image.BILINEAR``
    """

    def __init__(self, height, width, p=0.5, interpolation=Image.BILINEAR):
        self.height = height
        self.width = width
        self.p = p
        self.interpolation = interpolation

    def __call__(self, img):
        if random.uniform(0, 1) > self.p:
            return img.resize((self.width, self.height), self.interpolation)

        new_width, new_height = int(round(self.width * 1.125)
                                    ), int(round(self.height * 1.125))
        resized_img = img.resize((new_width, new_height), self.interpolation)
        x_maxrange = new_width - self.width
        y_maxrange = new_height - self.height
        x1 = int(round(random.uniform(0, x_maxrange)))
        y1 = int(round(random.uniform(0, y_maxrange)))
        croped_img = resized_img.crop(
            (x1, y1, x1 + self.width, y1 + self.height)
        )
        return croped_img


class RandomErasing(object):
    """Randomly erases an image patch.

    Origin: `<https://github.com/zhunzhong07/Random-Erasing>`_

    Reference:
        Zhong et al. Random Erasing Data Augmentation.

    Args:
        probability (float, optional): probability that this operation takes place.
            Default is 0.5.
        sl (float, optional): min erasing area.
        sh (float, optional): max erasing area.
        r1 (float, optional): min aspect ratio.
        mean (list, optional): erasing value.
    """

    def __init__(
        self,
        probability=0.5,
        sl=0.02,
        sh=0.4,
        r1=0.3,
        mean=[0.4914, 0.4822, 0.4465]
    ):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1

    def __call__(self, img):
        if random.uniform(0, 1) > self.probability:
            return img

        for attempt in range(100):
            area = img.size()[1] * img.size()[2]

            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1 / self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < img.size()[2] and h < img.size()[1]:
                x1 = random.randint(0, img.size()[1] - h)
                y1 = random.randint(0, img.size()[2] - w)
                if img.size()[0] == 3:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                    img[1, x1:x1 + h, y1:y1 + w] = self.mean[1]
                    img[2, x1:x1 + h, y1:y1 + w] = self.mean[2]
                else:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                return img

        return img


class ColorAugmentation(object):
    """Randomly alters the intensities of RGB channels.

    Reference:
        Krizhevsky et al. ImageNet Classification with Deep ConvolutionalNeural
        Networks. NIPS 2012.

    Args:
        p (float, optional): probability that this operation takes place.
            Default is 0.5.
    """

    def __init__(self, p=0.5):
        self.p = p
        self.eig_vec = torch.Tensor(
            [
                [0.4009, 0.7192, -0.5675],
                [-0.8140, -0.0045, -0.5808],
                [0.4203, -0.6948, -0.5836],
            ]
        )
        self.eig_val = torch.Tensor([[0.2175, 0.0188, 0.0045]])

    def _check_input(self, tensor):
        assert tensor.dim() == 3 and tensor.size(0) == 3

    def __call__(self, tensor):
        if random.uniform(0, 1) > self.p:
            return tensor
        alpha = torch.normal(mean=torch.zeros_like(self.eig_val)) * 0.1
        quatity = torch.mm(self.eig_val * alpha, self.eig_vec)
        tensor = tensor + quatity.view(3, 1, 1)
        return tensor


class RandomPatch(object):
    """Random patch data augmentation.

    There is a patch pool that stores randomly extracted pathces from person images.
    
    For each input image, RandomPatch
        1) extracts a random patch and stores the patch in the patch pool;
        2) randomly selects a patch from the patch pool and pastes it on the
           input (at random position) to simulate occlusion.

    Reference:
        - Zhou et al. Omni-Scale Feature Learning for Person Re-Identification. ICCV, 2019.
        - Zhou et al. Learning Generalisable Omni-Scale Representations
          for Person Re-Identification. TPAMI, 2021.
    """

    def __init__(
        self,
        prob_happen=0.5,
        pool_capacity=50000,
        min_sample_size=100,
        patch_min_area=0.01,
        patch_max_area=0.5,
        patch_min_ratio=0.1,
        prob_rotate=0.5,
        prob_flip_leftright=0.5,
    ):
        self.prob_happen = prob_happen

        self.patch_min_area = patch_min_area
        self.patch_max_area = patch_max_area
        self.patch_min_ratio = patch_min_ratio

        self.prob_rotate = prob_rotate
        self.prob_flip_leftright = prob_flip_leftright

        self.patchpool = deque(maxlen=pool_capacity)
        self.min_sample_size = min_sample_size

    def generate_wh(self, W, H):
        area = W * H
        for attempt in range(100):
            target_area = random.uniform(
                self.patch_min_area, self.patch_max_area
            ) * area
            aspect_ratio = random.uniform(
                self.patch_min_ratio, 1. / self.patch_min_ratio
            )
            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))
            if w < W and h < H:
                return w, h
        return None, None

    def transform_patch(self, patch):
        if random.uniform(0, 1) > self.prob_flip_leftright:
            patch = patch.transpose(Image.FLIP_LEFT_RIGHT)
        if random.uniform(0, 1) > self.prob_rotate:
            patch = patch.rotate(random.randint(-10, 10))
        return patch

    def __call__(self, img):
        W, H = img.size # original image size

        # collect new patch
        w, h = self.generate_wh(W, H)
        if w is not None and h is not None:
            x1 = random.randint(0, W - w)
            y1 = random.randint(0, H - h)
            new_patch = img.crop((x1, y1, x1 + w, y1 + h))
            self.patchpool.append(new_patch)

        if len(self.patchpool) < self.min_sample_size:
            return img

        if random.uniform(0, 1) > self.prob_happen:
            return img

        # paste a randomly selected patch on a random position
        patch = random.sample(self.patchpool, 1)[0]
        patchW, patchH = patch.size
        x1 = random.randint(0, W - patchW)
        y1 = random.randint(0, H - patchH)
        patch = self.transform_patch(patch)
        img.paste(patch, (x1, y1))

        return img


def build_transforms_old(
    height,
    width,
    transforms='random_flip',
    norm_mean=[0.485, 0.456, 0.406],
    norm_std=[0.229, 0.224, 0.225],
    **kwargs
):
    """Builds train and test transform functions.

    Args:
        height (int): target image height.
        width (int): target image width.
        transforms (str or list of str, optional): transformations applied to model training.
            Default is 'random_flip'.
        norm_mean (list or None, optional): normalization mean values. Default is ImageNet means.
        norm_std (list or None, optional): normalization standard deviation values. Default is
            ImageNet standard deviation values.
    """
    if transforms is None:
        transforms = []

    if isinstance(transforms, str):
        transforms = [transforms]

    if not isinstance(transforms, list):
        raise ValueError(
            'transforms must be a list of strings, but found to be {}'.format(
                type(transforms)
            )
        )

    if len(transforms) > 0:
        transforms = [t.lower() for t in transforms]

    if norm_mean is None or norm_std is None:
        norm_mean = [0.485, 0.456, 0.406] # imagenet mean
        norm_std = [0.229, 0.224, 0.225] # imagenet std
    normalize = Normalize(mean=norm_mean, std=norm_std)

    print('Building train transforms ...')
    transform_tr = []

    print('+ resize to {}x{}'.format(height, width))
    transform_tr += [Resize((height, width))]

    if 'random_flip' in transforms:
        print('+ random flip')
        transform_tr += [RandomHorizontalFlip()]

    if 'random_crop' in transforms:
        print(
            '+ random crop (enlarge to {}x{} and '
            'crop {}x{})'.format(
                int(round(height * 1.125)), int(round(width * 1.125)), height,
                width
            )
        )
        transform_tr += [Random2DTranslation(height, width)]

    if 'random_patch' in transforms:
        print('+ random patch')
        transform_tr += [RandomPatch()]

    if 'color_jitter' in transforms:
        print('+ color jitter')
        transform_tr += [
            ColorJitter(brightness=0.2, contrast=0.15, saturation=0, hue=0)
        ]

    print('+ to torch tensor of range [0, 1]')
    transform_tr += [ToTensor()]

    print('+ normalization (mean={}, std={})'.format(norm_mean, norm_std))
    transform_tr += [normalize]

    if 'random_erase' in transforms:
        print('+ random erase')
        transform_tr += [RandomErasing(mean=norm_mean)]

    transform_tr = Compose(transform_tr)

    print('Building test transforms ...')
    print('+ resize to {}x{}'.format(height, width))
    print('+ to torch tensor of range [0, 1]')
    print('+ normalization (mean={}, std={})'.format(norm_mean, norm_std))

    transform_te = Compose([
        Resize((height, width)),
        ToTensor(),
        normalize,
    ])

    return transform_tr, transform_te


def build_transforms(
    height,
    width,
    transforms='random_flip',
    norm_mean=[0.485, 0.456, 0.406],
    norm_std=[0.229, 0.224, 0.225],
    **kwargs
):
    """Builds train and test transform functions.

    Args:
        height (int): target image height.
        width (int): target image width.
        transforms (str or list of str, optional): transformations applied to model training.
            Default is 'random_flip'.
        norm_mean (list or None, optional): normalization mean values. Default is ImageNet means.
        norm_std (list or None, optional): normalization standard deviation values. Default is
            ImageNet standard deviation values.
    """
    if transforms is None:
        transforms = []

    if isinstance(transforms, str):
        transforms = [transforms]

    if not isinstance(transforms, list):
        raise ValueError(
            'transforms must be a list of strings, but found to be {}'.format(
                type(transforms)
            )
        )

    if len(transforms) > 0:
        transforms = [t.lower() for t in transforms]

    if norm_mean is None or norm_std is None:
        norm_mean = [0.485, 0.456, 0.406] # imagenet mean
        norm_std = [0.229, 0.224, 0.225] # imagenet std
    normalize = Normalize(mean=norm_mean, std=norm_std)

    print('Building train transforms ...')
    transform_tr = []

    # print('+ resize to {}x{}'.format(height, width))
    # transform_tr += [Resize((height, width))]

    if 'random_flip' in transforms:
        print('+ random flip')
        transform_tr += [RandomHorizontalFlip()]

    if 'random_crop' in transforms:
        print(
            '+ random crop (crop from upper part of the image)'.format(
                int(round(height * 1.125)), int(round(width * 1.125)), height,
                width
            )
        )
        transform_tr += [RandomCropByScaleAndRatio(scale=(0.8, 1.0), ratio=(1 / 2.5, 1 / 1.), min_upper_boundary=0.85, p=0.5)]

    if 'random_patch' in transforms:
        print('+ random patch')
        transform_tr += [RandomPatch()]

    if 'color_jitter' in transforms:
        print('+ color jitter')
        transform_tr += [
            ColorJitter(brightness=0.2, contrast=0.15, saturation=0, hue=0)
        ]

    print('+ to torch tensor of range [0, 1]')
    transform_tr += [ToTensor()]

    print('+ resize and pad')
    transform_tr += [AlignResizePad((height, width))]

    print('+ normalization (mean={}, std={})'.format(norm_mean, norm_std))
    transform_tr += [normalize]

    if 'random_erase' in transforms:
        print('+ random erase')
        transform_tr += [RandomErasing(mean=norm_mean)]

    transform_tr = Compose(transform_tr)

    print('Building test transforms ...')
    print('+ to torch tensor of range [0, 1]')
    print('+ resize and pad')
    print('+ normalization (mean={}, std={})'.format(norm_mean, norm_std))

    transform_te = Compose([
        ToTensor(),
        AlignResizePad((height, width)),
        normalize,
    ])

    return transform_tr, transform_te
