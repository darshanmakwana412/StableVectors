import torch
import numpy as np
import PIL
from PIL import Image
from typing import List, Optional, Union

def denormalize(images):
    """
    Denormalize an image array to [0,1]
    """
    return (images / 2 + 0.5).clamp(0, 1)

def pt_to_numpy(images: torch.FloatTensor) -> np.ndarray:
    """
    Convert a pytorch tensor to a numpy image
    """
    images = images.cpu().permute(0, 2, 3, 1).float().numpy()
    return images

def numpy_to_pil(images: np.ndarray) -> PIL.Image.Image:
    """
    Convert a numpy image or a batch of images to a PIL image.
    """
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    if images.shape[-1] == 1:
        # special case for grayscale (single channel) images
        pil_images = [Image.fromarray(image.squeeze(), mode="L") for image in images]
    else:
        pil_images = [Image.fromarray(image) for image in images]

    return pil_images

def pil_to_numpy(images: Union[List[PIL.Image.Image], PIL.Image.Image]) -> np.ndarray:
    """
    Convert a PIL image or a list of PIL images to numpy arrays.
    """
    if not isinstance(images, list):
        images = [images]
    images = [np.array(image).astype(np.float32) / 255.0 for image in images]
    images = np.stack(images, axis=0)

    return images

def numpy_to_pt(images: np.ndarray) -> torch.FloatTensor:
    """
    Convert a numpy image to a pytorch tensor
    """
    if images.ndim == 3:
        images = images[..., None]

    images = torch.from_numpy(images.transpose(0, 3, 1, 2))
    return images