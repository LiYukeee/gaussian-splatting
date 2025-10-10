#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from torch import nn
import numpy as np
from utils.graphics_utils import getWorld2View2, getProjectionMatrix
import math
from utils.general_utils import PILtoTorch
import cv2

class Camera(nn.Module):
    def __init__(self, resolution, colmap_id, R, T, FoVx, FoVy, depth_params, image, invdepthmap_path,
                 image_name, uid,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda",
                 train_test_exp = False, is_test_dataset = False, is_test_view = False
                 ):
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name
        self.resolution = resolution
        self.depth_params = depth_params
        self.depth_path = invdepthmap_path

        if data_device != "disk":  # the origin method
            try:
                self.data_device = torch.device(data_device)
            except Exception as e:
                print(e)
                print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
                self.data_device = torch.device("cuda")
        else:
            self.data_device = "disk"
        
        if data_device != "disk":  # the origin method
            resized_image_rgb = PILtoTorch(image, resolution)
            gt_image = resized_image_rgb[:3, ...]
            self.alpha_mask = None
            if resized_image_rgb.shape[0] == 4:
                self.alpha_mask = resized_image_rgb[3:4, ...].to(self.data_device)
            else: 
                self.alpha_mask = torch.ones_like(resized_image_rgb[0:1, ...].to(self.data_device))

            if train_test_exp and is_test_view:
                if is_test_dataset:
                    self.alpha_mask[..., :self.alpha_mask.shape[-1] // 2] = 0
                else:
                    self.alpha_mask[..., self.alpha_mask.shape[-1] // 2:] = 0

            self.gt_image = gt_image.clamp(0.0, 1.0).to(self.data_device)
            self.image_width = self.gt_image.shape[2]
            self.image_height = self.gt_image.shape[1]
        else:
            self.image = image
            self.image_width = resolution[0]
            self.image_height = resolution[1]
            self.alpha_mask = None

        self.invdepthmap_path = invdepthmap_path
        if self.invdepthmap_path == "":
            self.depth_reliable = False
        else:
            self.depth_reliable = True

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

    @property
    def invdepthmap(self):
        depth_path = self.depth_path
        depth_params = self.depth_params
        invdepthmap = cv2.imread(depth_path, -1).astype(np.float32) / float(2**16)
        invdepthmap = cv2.resize(invdepthmap, self.resolution)
        invdepthmap[invdepthmap < 0] = 0
        if depth_params is not None:
            if depth_params["scale"] < 0.2 * depth_params["med_scale"] or depth_params["scale"] > 5 * depth_params["med_scale"]:
                self.depth_reliable = False
            if depth_params["scale"] > 0:
                invdepthmap = invdepthmap * depth_params["scale"] + depth_params["offset"]
        if invdepthmap.ndim != 2:
            invdepthmap = invdepthmap[..., 0]
        return torch.tensor(invdepthmap[None]).to('cuda')
    
    @property
    def original_image(self):
        if self.data_device == "disk":
            # The image is not read entirely from the disk; 
            # instead, the source file of the image is stored in memory, 
            # and the image is decoded only when needed. This approach is very memory-efficient.
            image = self.image
            return PILtoTorch(image, image.size)[:3, ...]
        else:
            return self.gt_image

class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
        self.image_width = width
        self.image_height = height    
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]

