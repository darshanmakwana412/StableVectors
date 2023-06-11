import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import clip

USE_CUDA = torch.cuda.is_available()
DEVICE = 'cuda' if USE_CUDA else 'cpu'
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
CLIP_NORMALIZE = torchvision.transforms.Normalize(CLIP_MEAN, CLIP_STD)  # normalize an image that is already scaled to [0, 1]
clip_model_vit, _ = clip.load("ViT-L/14@336px", device=DEVICE, jit=False)
clip_model_vit.eval()

@torch.no_grad()
def embed_text(text: str):
    assert isinstance(text, str)
    text = clip.tokenize(text).to(DEVICE)
    text_features_vit = clip_model_vit.encode_text(text)
    return torch.cat([text_features_vit], dim=-1)

@torch.no_grad()
def rgba_to_rgb(rgba_image):
    return rgba_image[:, :, 3:4] * rgba_image[:, :, :3] + torch.ones(rgba_image.shape[0], rgba_image.shape[1], 3, device=DEVICE) * (1 - rgba_image[:, :, 3:4])

@torch.no_grad()
def embed_image(image):
    image = CLIP_NORMALIZE(image.to(DEVICE))
    image_features_vit = clip_model_vit.encode_image(image)
    return torch.cat([image_features_vit], dim=-1)