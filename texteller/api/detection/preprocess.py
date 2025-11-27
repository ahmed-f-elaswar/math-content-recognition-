"""Image preprocessing for text detection in TexTeller.

This module provides preprocessing operations for preparing images for text
detection using PaddleOCR-based detection models. It includes image decoding,
resizing, normalization, and composition of preprocessing pipelines.

The preprocessing pipeline typically consists of:
1. decode_image: Load and decode image from file or array
2. Resize: Resize image to target size with optional aspect ratio preservation
3. NormalizeImage: Normalize pixel values for model input
4. Permute: Convert image from HWC to CHW format

Classes:
    Resize: Resize images to target dimensions.
    NormalizeImage: Normalize image pixel values.
    Permute: Permute image dimensions from HWC to CHW.
    Compose: Compose multiple preprocessing operations.

Functions:
    decode_image: Decode image from file path or numpy array.

Examples:
    Basic preprocessing pipeline::
    
        from texteller.api.detection.preprocess import Compose
        
        preprocess = Compose([
            {"type": "Resize", "target_size": [640, 640]},
            {"type": "NormalizeImage", "mean": [0.485, 0.456, 0.406],
             "std": [0.229, 0.224, 0.225], "is_scale": True},
            {"type": "Permute"}
        ])
        
        # Process image
        result = preprocess("image.jpg")
        img = result["image"]  # Preprocessed image
    
    Custom preprocessing::
    
        from texteller.api.detection.preprocess import (
            decode_image, Resize, NormalizeImage
        )
        
        # Load image
        img, img_info = decode_image("image.jpg")
        
        # Resize
        resize_op = Resize(target_size=[800, 800], keep_ratio=True)
        img, img_info = resize_op(img, img_info)
        
        # Normalize
        norm_op = NormalizeImage(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        img, img_info = norm_op(img, img_info)
"""

import copy

import cv2
import numpy as np


def decode_image(img_path):
	"""Decode image from file path or numpy array.
	
	Loads an image from a file path or accepts a numpy array directly.
	Converts the image from BGR to RGB color space.
	
	Args:
		img_path (str or np.ndarray): Either a file path to an image or
			a numpy array containing image data.
	
	Returns:
		tuple: A tuple containing:
			- im (np.ndarray): Decoded image in RGB format with shape (H, W, C).
			- img_info (dict): Dictionary with image metadata:
				- im_shape: Original image shape as (height, width).
				- scale_factor: Scaling factors as [1.0, 1.0].
	
	Examples:
		Load from file::
		
			img, info = decode_image("photo.jpg")
			print(img.shape)  # (480, 640, 3)
			print(info["im_shape"])  # array([480., 640.])
		
		Use existing array::
		
			import numpy as np
			arr = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
			img, info = decode_image(arr)
	"""
    if isinstance(img_path, str):
        with open(img_path, "rb") as f:
            im_read = f.read()
        data = np.frombuffer(im_read, dtype="uint8")
    else:
        assert isinstance(img_path, np.ndarray)
        data = img_path

    im = cv2.imdecode(data, 1)  # BGR mode, but need RGB mode
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    img_info = {
        "im_shape": np.array(im.shape[:2], dtype=np.float32),
        "scale_factor": np.array([1.0, 1.0], dtype=np.float32),
    }
    return im, img_info


class Resize(object):
    """Resize image to target dimensions.
    
    Resizes images to specified dimensions with optional aspect ratio preservation.
    This is commonly used as the first preprocessing step to standardize input sizes.
    
    Args:
        target_size (int or list): Target size for resizing. If int, resizes to
            [target_size, target_size]. If list, should be [height, width].
        keep_ratio (bool, optional): Whether to preserve aspect ratio during resize.
            If True, scales the image such that the smaller dimension matches the
            smaller target dimension without exceeding the larger target dimension.
            Defaults to True.
        interp (int, optional): OpenCV interpolation method. Common values:
            - cv2.INTER_LINEAR (default): Bilinear interpolation
            - cv2.INTER_NEAREST: Nearest neighbor
            - cv2.INTER_CUBIC: Bicubic interpolation
            Defaults to cv2.INTER_LINEAR.
    
    Examples:
        Resize with aspect ratio preservation::
        
            resize_op = Resize(target_size=640, keep_ratio=True)
            img, img_info = resize_op(img, img_info)
        
        Resize to exact dimensions::
        
            resize_op = Resize(target_size=[480, 640], keep_ratio=False)
            img, img_info = resize_op(img, img_info)
    """

    def __init__(self, target_size, keep_ratio=True, interp=cv2.INTER_LINEAR):
        if isinstance(target_size, int):
            target_size = [target_size, target_size]
        self.target_size = target_size
        self.keep_ratio = keep_ratio
        self.interp = interp

    def __call__(self, im, im_info):
        """
        Args:
            im (np.ndarray): image (np.ndarray)
            im_info (dict): info of image
        Returns:
            im (np.ndarray):  processed image (np.ndarray)
            im_info (dict): info of processed image
        """
        assert len(self.target_size) == 2
        assert self.target_size[0] > 0 and self.target_size[1] > 0
        im_channel = im.shape[2]
        im_scale_y, im_scale_x = self.generate_scale(im)
        im = cv2.resize(im, None, None, fx=im_scale_x, fy=im_scale_y, interpolation=self.interp)
        im_info["im_shape"] = np.array(im.shape[:2]).astype("float32")
        im_info["scale_factor"] = np.array([im_scale_y, im_scale_x]).astype("float32")
        return im, im_info

    def generate_scale(self, im):
        """
        Args:
            im (np.ndarray): image (np.ndarray)
        Returns:
            im_scale_x: the resize ratio of X
            im_scale_y: the resize ratio of Y
        """
        origin_shape = im.shape[:2]
        im_c = im.shape[2]
        if self.keep_ratio:
            im_size_min = np.min(origin_shape)
            im_size_max = np.max(origin_shape)
            target_size_min = np.min(self.target_size)
            target_size_max = np.max(self.target_size)
            im_scale = float(target_size_min) / float(im_size_min)
            if np.round(im_scale * im_size_max) > target_size_max:
                im_scale = float(target_size_max) / float(im_size_max)
            im_scale_x = im_scale
            im_scale_y = im_scale
        else:
            resize_h, resize_w = self.target_size
            im_scale_y = resize_h / float(origin_shape[0])
            im_scale_x = resize_w / float(origin_shape[1])
        return im_scale_y, im_scale_x


class NormalizeImage(object):
    """Normalize image pixel values.
    
    Normalizes image pixels using mean subtraction and standard deviation division.
    This is a critical preprocessing step for neural network models.
    
    Args:
        mean (list): Mean values for each channel (R, G, B). Subtracted from pixel values.
        std (list): Standard deviation values for each channel (R, G, B). Pixel values
            are divided by these after mean subtraction.
        is_scale (bool, optional): Whether to scale pixel values by 1/255 before
            normalization. Defaults to True.
        norm_type (str, optional): Normalization type. Options:
            - 'mean_std': Apply mean subtraction and std division (default)
            - 'none': No normalization (only scaling if is_scale=True)
            Defaults to 'mean_std'.
    
    Examples:
        Standard ImageNet normalization::
        
            norm_op = NormalizeImage(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                is_scale=True
            )
            img, img_info = norm_op(img, img_info)
        
        Only scale without normalization::
        
            norm_op = NormalizeImage(
                mean=[0, 0, 0],
                std=[1, 1, 1],
                is_scale=True,
                norm_type='none'
            )
    """

    def __init__(self, mean, std, is_scale=True, norm_type="mean_std"):
        self.mean = mean
        self.std = std
        self.is_scale = is_scale
        self.norm_type = norm_type

    def __call__(self, im, im_info):
        """
        Args:
            im (np.ndarray): image (np.ndarray)
            im_info (dict): info of image
        Returns:
            im (np.ndarray):  processed image (np.ndarray)
            im_info (dict): info of processed image
        """
        im = im.astype(np.float32, copy=False)
        if self.is_scale:
            scale = 1.0 / 255.0
            im *= scale

        if self.norm_type == "mean_std":
            mean = np.array(self.mean)[np.newaxis, np.newaxis, :]
            std = np.array(self.std)[np.newaxis, np.newaxis, :]
            im -= mean
            im /= std
        return im, im_info


class Permute(object):
    """Permute image dimensions from HWC to CHW format.
    
    Converts image from Height-Width-Channel (H, W, C) format to
    Channel-Height-Width (C, H, W) format, which is required by most
    PyTorch and ONNX models.
    
    Examples:
        Convert HWC to CHW::
        
            permute_op = Permute()
            img, img_info = permute_op(img, img_info)
            # img shape changes from (H, W, 3) to (3, H, W)
    """

    def __init__(
        self,
    ):
        """Initialize the Permute operation."""
        super(Permute, self).__init__()

    def __call__(self, im, im_info):
        """
        Args:
            im (np.ndarray): image (np.ndarray)
            im_info (dict): info of image
        Returns:
            im (np.ndarray):  processed image (np.ndarray)
            im_info (dict): info of processed image
        """
        im = im.transpose((2, 0, 1)).copy()
        return im, im_info


class Compose:
    """Compose multiple preprocessing operations into a pipeline.
    
    Chains together multiple preprocessing operations that are applied sequentially
    to images. Each operation receives the output of the previous one.
    
    Args:
        transforms (list): List of dictionaries, each specifying a preprocessing
            operation. Each dict must have a 'type' key with the operation class
            name, and other keys for operation parameters.
    
    Examples:
        Create a complete preprocessing pipeline::
        
            compose = Compose([
                {"type": "Resize", "target_size": [640, 640], "keep_ratio": True},
                {"type": "NormalizeImage", 
                 "mean": [0.485, 0.456, 0.406],
                 "std": [0.229, 0.224, 0.225],
                 "is_scale": True},
                {"type": "Permute"}
            ])
            
            result = compose("image.jpg")
            preprocessed_img = result["image"]
    """
    
    def __init__(self, transforms):
        """Initialize the Compose pipeline with a list of transforms.
        
        Args:
            transforms (list): List of transform specifications as dictionaries.
        """
        self.transforms = []
        for op_info in transforms:
            new_op_info = op_info.copy()
            op_type = new_op_info.pop("type")
            self.transforms.append(eval(op_type)(**new_op_info))

    def __call__(self, img_path):
        """Apply all preprocessing operations to an image.
        
        Args:
            img_path (str or np.ndarray): Path to image file or numpy array.
        
        Returns:
            dict: Dictionary containing:
                - 'image': Preprocessed image as numpy array
                - 'im_shape': Original image shape
                - 'scale_factor': Scaling factors applied
        
        Examples:
            >>> compose = Compose([{"type": "Resize", "target_size": 640}])
            >>> result = compose("image.jpg")
            >>> print(result.keys())
            dict_keys(['image', 'im_shape', 'scale_factor'])
        """
        img, im_info = decode_image(img_path)
        for t in self.transforms:
            img, im_info = t(img, im_info)
        inputs = copy.deepcopy(im_info)
        inputs["image"] = img
        return inputs
