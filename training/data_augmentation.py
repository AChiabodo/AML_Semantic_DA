"""
    # data_augmentation.py #

    This module provides a set of data augmentation transformations specifically 
    designed for image segmentation tasks. The aim of data augmentation is to 
    increase the amount of training data using reasonable transformations for our
    specific case, i.e. street view images.

    - ExtTransforms is a base callabale class that defines the interface for
    all the transformations.

    These transformations can be combined to create a robust data augmentation 
    pipeline, enhancing the generalization of semantic segmentation models 
    trained on one dataset (like GTA5) and tested on another (like Cityscapes).

    - ExtCompose: Composes several transforms together.

    Usage:
    >>> from data_augmentation import ExtCompose, ExtResize, ExtToTensor
    >>> transform = ExtCompose([
    >>>     ExtScale(scale=0.5),
    >>>     ExtRandomHorizontalFlip(),
    >>>     ExtToTensor(),
    >>> ])
    >>> augmented_image, augmented_label = transform(image, label)
"""


import torch
import torchvision.transforms.functional as F
from PIL import Image
import numpy as np
import random
import numbers
from typing import Optional, List


##############
# BASE CLASS #
##############
class ExtTransforms(object):
	def	__init__(self) -> None:
		pass

	def __call__(self, img, lbl):
		pass


####################
# BUILDING CLASSES #
####################
class ExtCompose(ExtTransforms):
    """
    Composes several transforms together.

    Args:
    - transforms: list of ``Transform`` objects to compose.
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img : Image, lbl : Image) -> (Image, Image):
        for t in self.transforms:
            img, lbl = t(img, lbl)
        return img, lbl
    
class ExtRandomCompose(ExtTransforms):
    """
    Composes several transforms together.

    Args:
    - transforms: list of ``Transform`` objects to compose.
    """

    def __init__(self, transforms_1, transforms_2):
        self.transforms_1 = transforms_1
        self.transforms_2 = transforms_2

    def __call__(self, img : Image, lbl : Image) -> (Image, Image):
        if random.uniform(0, 1) > 0.5:
            for t in self.transforms_2:
                img, lbl = t(img, lbl)
        else:
            for t in self.transforms_1:
                img, lbl = t(img, lbl)
        return img, lbl

class ExtToTensor(ExtTransforms):
    """
    Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor.
    Converts a PIL Image or numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].

    Args:
    - target_type: type of the target tensor.
    """
    def __init__(self, target_type='uint8'):
        self.target_type = target_type

    def __call__(self, pic : Image, lbl : Image) -> (torch.Tensor, torch.Tensor):
        return torch.from_numpy( np.array( pic, dtype=np.float32).transpose(2, 0, 1) ), torch.from_numpy( np.array( lbl, dtype=self.target_type) )


######################
# TRANSFORMS CLASSES #
######################

# SIZE ADJUSTMENTS
# Resize
class ExtResize(ExtTransforms):
    """
    Resize the input PIL Image and its label to the given size.

    Args:
    - size (height, width): Desired output size
    - interpolation: Desired interpolation for the image
    
    For the label, we use nearest neighbor interpolation.
    """

    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img : Image, lbl : Image) -> (Image, Image):
        return F.resize(img, self.size, self.interpolation), F.resize(lbl, self.size, Image.NEAREST)
    
# Rescale
class ExtScale(ExtTransforms):
    """
    Rescale the input PIL Image and its label by a factor.

    Args:
    - scale: Desired scale factor
    - interpolation: Desired interpolation for the image

    For the label, we use nearest neighbor interpolation.
    """

    def __init__(self, scale : float = 0.5, interpolation=Image.BILINEAR):
        self.scale = scale
        self.interpolation = interpolation

    def __call__(self, img : Image, lbl : Image) -> (Image, Image):
        assert img.size == lbl.size
        target_size = ( int(img.size[1]*self.scale), int(img.size[0]*self.scale) ) # (H, W)
        return F.resize(img, target_size, self.interpolation), F.resize(lbl, target_size, Image.NEAREST)

# Crop
class ExtRandomCrop(ExtTransforms):
    """
    Crop the given PIL Image at a random location.

    Args:
    - size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
    - padding (int or sequence, optional): Optional padding on each border
            of the image. Default is 0, i.e no padding. If a sequence of length
            4 is provided, it is used to pad left, top, right, bottom borders
            respectively.
    - pad_if_needed (boolean): It will pad the image if smaller than the
            desired size to avoid raising an exception.
    """

    def __init__(self, size, padding=0, pad_if_needed=False):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding
        self.pad_if_needed = pad_if_needed

    @staticmethod
    def get_params(img, output_size):
        """Get parameters for ``crop`` for a random crop.
        Args:
            img (PIL Image): Image to be cropped.
            output_size (tuple): Expected output size of the crop.
        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for random crop.
        """
        w, h = img.size
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw

    def __call__(self, img : Image, lbl : Image) -> (Image , Image):
        """
        Args:
            img (PIL Image): Image to be cropped.
            lbl (PIL Image): Label to be cropped.
        Returns:
            PIL Image: Cropped image.
            PIL Image: Cropped label.
        """
        assert img.size == lbl.size, 'size of img and lbl should be the same. %s, %s'%(img.size, lbl.size)
        if self.padding > 0:
            img = F.pad(img, self.padding)
            lbl = F.pad(lbl, self.padding)

        # pad the width if needed
        if self.pad_if_needed and img.size[0] < self.size[1]:
            img = F.pad(img, padding=int((1 + self.size[1] - img.size[0]) / 2))
            lbl = F.pad(lbl, padding=int((1 + self.size[1] - lbl.size[0]) / 2))

        # pad the height if needed
        if self.pad_if_needed and img.size[1] < self.size[0]:
            img = F.pad(img, padding=int((1 + self.size[0] - img.size[1]) / 2))
            lbl = F.pad(lbl, padding=int((1 + self.size[0] - lbl.size[1]) / 2))

        i, j, h, w = self.get_params(img, self.size)

        return F.crop(img, i, j, h, w), F.crop(lbl, i, j, h, w)

    def __repr__(self):
        return self.__class__.__name__ + '(size={0}, padding={1})'.format(self.size, self.padding)
    

# FLIPS AND ROTATIONS
# Horizontally flip
class ExtRandomHorizontalFlip(ExtTransforms):
    """
    Horizontally flip the given PIL Image and its label randomly with a given probability.

    Args:
    - p: probability of the image being flipped.
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img : Image, lbl : Image) -> (Image, Image):
        if random.random() < self.p:
            return F.hflip(img), F.hflip(lbl)
        return img, lbl


# NOISE AND BLUR
# Gaussian BLur
class ExtGaussianBlur(ExtTransforms):
    """
    Apply Gaussian Blur to the given PIL Image and its label with a given probability.

    Args:
    - p: probability of the image being blurred.
    - radius: radius of the Gaussian blur kernel (must be odd and positive)
    - sigma: Optional standard deviation of the Gaussian blur kernel
    """

    def __init__( self, p=0.5, radius:List[int]=1, sigma:Optional[List[float]]=None ):
        self.p = p
        self.radius = radius
        self.sigma = sigma

    def __call__(self, img : Image, lbl : Image) -> (Image, Image):
        if random.random() < self.p:
            return F.gaussian_blur(img, self.radius, self.sigma), lbl
        return img, lbl  


# COLOR ADJUSTMENTS
# Color
class ExtColorJitter(ExtTransforms):
    """
    Randomly change the brightness, contrast and saturation of an image.

    Args:
    - p: probability of the image being color jittered.
    - brightness (float or tuple of float (min, max)): How much to jitter brightness.
            brightness_factor is chosen uniformly from [max(0, 1 - brightness), 1 + brightness]
            or the given [min, max]. Should be non negative numbers.
    - contrast (float or tuple of float (min, max)): How much to jitter contrast.
            contrast_factor is chosen uniformly from [max(0, 1 - contrast), 1 + contrast]
            or the given [min, max]. Should be non negative numbers.
    - saturation (float or tuple of float (min, max)): How much to jitter saturation.
            saturation_factor is chosen uniformly from [max(0, 1 - saturation), 1 + saturation]
            or the given [min, max]. Should be non negative numbers.
    - hue (float or tuple of float (min, max)): How much to jitter hue.
            hue_factor is chosen uniformly from [-hue, hue] or the given [min, max].
            Should have 0<= hue <= 0.5 or -0.5 <= min <= max <= 0.5.
    """
    def __init__(self, p=0.5, brightness=0, contrast=0, saturation=0, hue=0):
        self.p = p
        self.brightness = self._check_input(brightness, 'brightness')
        self.contrast = self._check_input(contrast, 'contrast')
        self.saturation = self._check_input(saturation, 'saturation')
        self.hue = self._check_input(hue, 'hue', center=0, bound=(-0.5, 0.5),
                                     clip_first_on_zero=False)

    def _check_input(self, value, name, center=1, bound=(0, float('inf')), clip_first_on_zero=True):
        if isinstance(value, numbers.Number):
            if value < 0:
                raise ValueError("If {} is a single number, it must be non negative.".format(name))
            value = [center - value, center + value]
            if clip_first_on_zero:
                value[0] = max(value[0], 0)
        elif isinstance(value, (tuple, list)) and len(value) == 2:
            if not bound[0] <= value[0] <= value[1] <= bound[1]:
                raise ValueError("{} values should be between {}".format(name, bound))
        else:
            raise TypeError("{} should be a single number or a list/tuple with lenght 2.".format(name))

        # if value is 0 or (1., 1.) for brightness/contrast/saturation
        # or (0., 0.) for hue, do nothing
        if value[0] == value[1] == center:
            value = None
        return value

    @staticmethod
    def get_params(brightness, contrast, saturation, hue):
        """Get a randomized transform to be applied on image.
        Arguments are same as that of __init__.
        Returns:
            Transform which randomly adjusts brightness, contrast and
            saturation in a random order.
        """
        transforms = []

        if brightness is not None:
            brightness_factor = random.uniform(brightness[0], brightness[1])
            transforms.append(Lambda(lambda img: F.adjust_brightness(img, brightness_factor)))

        if contrast is not None:
            contrast_factor = random.uniform(contrast[0], contrast[1])
            transforms.append(Lambda(lambda img: F.adjust_contrast(img, contrast_factor)))

        if saturation is not None:
            saturation_factor = random.uniform(saturation[0], saturation[1])
            transforms.append(Lambda(lambda img: F.adjust_saturation(img, saturation_factor)))

        if hue is not None:
            hue_factor = random.uniform(hue[0], hue[1])
            transforms.append(Lambda(lambda img: F.adjust_hue(img, hue_factor)))

        random.shuffle(transforms)
        transform = Compose(transforms)

        return transform

    def __call__(self, img, lbl):
        """
        Args:
            img (PIL Image): Input image.
        Returns:
            PIL Image: Color jittered image.
        """
        if random.random() < self.p:
          transform = self.get_params(self.brightness, self.contrast,
                                      self.saturation, self.hue)
          return transform(img), lbl
        return img, lbl

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'brightness={0}'.format(self.brightness)
        format_string += ', contrast={0}'.format(self.contrast)
        format_string += ', saturation={0}'.format(self.saturation)
        format_string += ', hue={0})'.format(self.hue)
        return format_string
	
# Lambda 
class Lambda(object):
    """Apply a user-defined lambda as a transform.
    Args:
        lambd (function): Lambda/function to be used for transform.
    """

    def __init__(self, lambd):
        assert callable(lambd), repr(type(lambd).__name__) + " object is not callable"
        self.lambd = lambd

    def __call__(self, img):
        return self.lambd(img)

    def __repr__(self):
        return self.__class__.__name__ + '()'
    

# COMPOSE different transformations together
class Compose(object):
    """Composes several transforms together.
    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.
    Example:
        >>> transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string
