import cv2
import numpy as np
import torch
from pytorch_grad_cam_modified.base_cam_original import BaseCAM
from pytorch_grad_cam_modified.utils.svd_on_activations import get_2d_projection

# https://arxiv.org/abs/2008.00299
class EigenCAM_original(BaseCAM):
    def __init__(self, model, target_layer, gpu_id=0, 
        reshape_transform=None):
        super(EigenCAM_original, self).__init__(model, target_layer, gpu_id, 
            reshape_transform)

    def get_cam_image(self,
                      input_tensor,
                      target_category,
                      activations,
                      grads,
                      eigen_smooth):
        return get_2d_projection(activations)
