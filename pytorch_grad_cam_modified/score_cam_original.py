import cv2
import numpy as np
import torch
import tqdm
from pytorch_grad_cam_modified.base_cam_original import BaseCAM

class ScoreCAM_original(BaseCAM):
    def __init__(self, model, target_layer, gpu_id=0, reshape_transform=None):
        super(ScoreCAM_original, self).__init__(model, target_layer, gpu_id, 
            reshape_transform=reshape_transform)

    def get_cam_weights(self,
                        input_tensor,
                        target_category,
                        activations,
                        grads):
        with torch.no_grad():
            upsample = torch.nn.UpsamplingBilinear2d(size=input_tensor.shape[-2 : ])
            activation_tensor = torch.from_numpy(activations)
            activation_tensor = activation_tensor.to(self.gpu_id)

            upsampled = upsample(activation_tensor)

            maxs = upsampled.view(upsampled.size(0),
                upsampled.size(1), -1).max(dim=-1)[0]
            mins = upsampled.view(upsampled.size(0),
                upsampled.size(1), -1).min(dim=-1)[0]
            maxs, mins = maxs[:, :, None, None], mins[:, :, None, None]
            upsampled = (upsampled - mins) / (maxs - mins)

            input_tensors = input_tensor[:, None, :, :]*upsampled[:, :, None, :, :]

            if hasattr(self, "batch_size"):
                BATCH_SIZE = self.batch_size
            else: 
                BATCH_SIZE = 64

            scores = []
            for batch_index, img_tensor in enumerate(input_tensors):
                category = target_category[batch_index]
                for i in tqdm.tqdm(range(0, img_tensor.size(0), BATCH_SIZE)):
                    batch_img = img_tensor[i : i + BATCH_SIZE, :]
                    outputs = self.model.forward(batch_img)
                    outputs = outputs.cpu().numpy()[:, category]
                    scores.extend(outputs)
            scores = torch.Tensor(scores)
            scores = scores.view(activations.shape[0], activations.shape[1])

            weights = torch.nn.Softmax(dim=-1)(scores).numpy()
            return weights
        