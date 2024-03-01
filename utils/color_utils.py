import cv2
from einops import rearrange
import imageio
import numpy as np

def read_image(img_path, img_wh, blend_a=True):
    img = imageio.imread(img_path).astype(np.float32)/255.0
    if img.shape[2] == 4: # blend A to RGB
        if blend_a:
            img = img[..., :3]*img[..., -1:]+(1-img[..., -1:])
        else:
            img = img[..., :3]*img[..., -1:]

    img = cv2.resize(img, img_wh)
    img = rearrange(img, 'h w c -> (h w) c')
    return img