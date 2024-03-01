import torch
import numpy as np
import os
import glob
from tqdm import tqdm

from utils.ray_utils import *
from utils.color_utils import read_image
from utils.colmap_utils import \
    read_cameras_binary, read_images_binary, read_points3d_binary
from utils.dark_channel import Atomospheric_light_k_means

from .base import BaseDataset


class LLFFDataset(BaseDataset):
    def __init__(self, root_dir, split='train', downsample=1.0, **kwargs):
        super().__init__(root_dir, split, downsample)

        self.read_intrinsics()
        self.haz_dir_name = kwargs.get("haz_dir_name")

        if kwargs.get('read_meta', True):
            self.read_meta(split, **kwargs)

    def read_intrinsics(self):
        # Step 1: read and scale intrinsics (same for all images)
        camdata = read_cameras_binary(os.path.join(self.root_dir, 'sparse/0/cameras.bin'))
        h = int(camdata[1].height / self.downsample)
        w = int(camdata[1].width / self.downsample)
        self.img_wh = (w, h)

        if camdata[1].model == 'SIMPLE_RADIAL':
            fx = fy = camdata[1].params[0] / self.downsample
            cx = camdata[1].params[1] / self.downsample
            cy = camdata[1].params[2] / self.downsample
        elif camdata[1].model in ['PINHOLE', 'OPENCV']:
            fx = camdata[1].params[0] / self.downsample
            fy = camdata[1].params[1] / self.downsample
            cx = camdata[1].params[2] / self.downsample
            cy = camdata[1].params[3] / self.downsample
        else:
            raise ValueError(f"Please parse the intrinsics for camera model {camdata[1].model}!")
        self.K = torch.FloatTensor([[fx, 0, cx],
                                    [0, fy, cy],
                                    [0, 0, 1]])
        self.directions = get_ray_directions(h, w, self.K)

    def read_meta(self, split, **kwargs):
        imdata = read_images_binary(os.path.join(self.root_dir, 'sparse/0/images.bin'))
        img_names = [imdata[k].name for k in imdata]
        perm = np.argsort(img_names)
        folder = f'images_{int(self.downsample)}'
        img_paths = [os.path.join(self.root_dir, folder, name)
                     for name in sorted(img_names)]
        w2c_mats = []
        bottom = np.array([[0, 0, 0, 1.]])
        for k in imdata:
            im = imdata[k]
            R = im.qvec2rotmat()
            t = im.tvec.reshape(3, 1)
            w2c_mats += [np.concatenate([np.concatenate([R, t], 1), bottom], 0)]
        w2c_mats = np.stack(w2c_mats, 0)
        poses = np.linalg.inv(w2c_mats)[perm, :3]

        pts3d = read_points3d_binary(os.path.join(self.root_dir, 'sparse/0/points3D.bin'))
        pts3d = np.array([pts3d[k].xyz for k in pts3d])  # (N, 3)

        self.poses, self.pts3d = center_poses(poses, pts3d)

        scale = np.linalg.norm(self.poses[..., 3], axis=-1).min()
        self.poses[..., 3] /= scale
        self.pts3d /= scale

        self.haz_images = []
        self.clear_images = []
        self.atmospheric_lights = []

        # use every 8th image as test set
        if split == 'train':
            img_paths = [x for i, x in enumerate(img_paths) if i % 8 != 0]
            self.poses = np.array([x for i, x in enumerate(self.poses) if i % 8 != 0])
        elif split == 'test':
            img_paths = [x for i, x in enumerate(img_paths) if i % 8 == 0]
            self.poses = np.array([x for i, x in enumerate(self.poses) if i % 8 == 0])

        print(f'Loading {len(img_paths)} {split} images ...')
        for img_path in tqdm(img_paths):
            clear_buf = []
            haz_buf = []

            if self.downsample is not 1.0:
                img_path = img_path.replace("JPG", "png")
                img_path = img_path.replace("jpg", "png")

            clear_img = read_image(img_path, self.img_wh, blend_a=False)
            clear_img = torch.FloatTensor(clear_img)
            clear_buf += [clear_img]
            self.clear_images += [torch.cat(clear_buf, 1)]

            haz_path = img_path.replace(folder, folder + "_" + self.haz_dir_name)
            haz_img = read_image(haz_path, self.img_wh, blend_a=False)
            if split == "train":
                self.atmospheric_lights += [Atomospheric_light_k_means(
                    rearrange(haz_img, "(h w) c->h w c", h=self.img_wh[0]))]
            haz_img = torch.FloatTensor(haz_img)
            haz_buf += [haz_img]
            self.haz_images += [torch.cat(haz_buf, 1)]


        self.clear_images = torch.stack(self.clear_images)
        self.haz_images = torch.stack(self.haz_images)
        if split == "train":
            self.atmospheric_lights = torch.FloatTensor(np.stack(self.atmospheric_lights))
        self.poses = torch.FloatTensor(self.poses)
