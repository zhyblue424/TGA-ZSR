import cv2
import numpy as np
import torch
from pytorch_grad_cam_modified.base_cam import BaseCAM


class GradCAM(BaseCAM):
    def __init__(self, model, target_layer, gpu_id=0, 
        reshape_transform=None):
        super(GradCAM, self).__init__(model, target_layer, gpu_id, reshape_transform)

    def get_cam_weights(self,
                        input_tensor,
                        target_category,
                        activations,
                        grads):
        return np.mean(grads, axis=(2, 3))
