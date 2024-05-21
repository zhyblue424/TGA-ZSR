import cv2
import numpy as np
import torch
import ttach as tta
from pytorch_grad_cam_modified.activations_and_gradients import ActivationsAndGradients
from pytorch_grad_cam_modified.utils.svd_on_activations import get_2d_projection


class BaseCAM:
    
    def __init__(self, 
                 model,  
                 target_layer,  
                 gpu_id=0,  
                 reshape_transform=None):  
                 
        
        self.model = model.eval().to(gpu_id)  
        self.target_layer = target_layer  
        self.gpu_id = gpu_id  
        self.reshape_transform = reshape_transform  
        self.optimizer =  torch.optim.Adam(target_layer.parameters(),lr=0.01)

        self.activations_and_grads = ActivationsAndGradients(self.model, 
            target_layer, reshape_transform)
            
        self.text_tensor = 0 
        self.input_tensor = 0  




    # def forward(self, input_img):
    #     return self.model(input_img)

    def get_cam_weights(self,
                        input_tensor,
                        target_category,
                        activations,
                        grads):
        raise Exception("Not Implemented")

    def get_loss(self, output, target_category):
        loss = 0
        for i in range(len(target_category)):
            loss = loss + output[i, target_category[i]]
        return loss


    def get_cam_image(self,
                  input_tensor,
                  target_category,
                  activations,
                  grads,
                  eigen_smooth=False):


        weights = self.get_cam_weights(input_tensor, target_category, activations, grads)
        # print(weights.shape)

        weighted_activations = weights[:, :, None, None] * activations
        # print(weighted_activations.shape) #(1,768,7,7)
        # print(activations.shape)

        
        # eigen_smooth 
        if eigen_smooth:
            cam = get_2d_projection(weighted_activations)
        else:
            cam = weighted_activations.sum(axis=1)
            # print(cam.shape) #(1,7,7)

        return cam


    # def get_cam_image(self,
    #                   input_tensor,
    #                   target_category,
    #                   activations,
    #                   grads,
    #                   eigen_smooth=False):
    #     weights = self.get_cam_weights(input_tensor, target_category, activations, grads)
    #     weighted_activations = weights[:, :, None, None] * activations
    #     if eigen_smooth:
    #         cam = get_2d_projection(weighted_activations)
    #     else:
    #         cam = weighted_activations.sum(axis=1)
    #     return cam

    def forward(self, input_tensor, text_tensor, target_category=None, eigen_smooth=False, compute_text=False):
        self.text_tensor = text_tensor
        self.input_tensor = input_tensor
        output, _ = self.activations_and_grads(input_tensor, text_tensor)
        if type(target_category) is int:
            target_category = [target_category] * input_tensor.size(0)
        if target_category is None:
            target_category = np.argmax(output.cpu().data.numpy(), axis=-1)
        self.model.zero_grad()
        loss = self.get_loss(output, target_category)
        # loss.backward(retain_graph=True)
        loss.backward()
        activations = self.activations_and_grads.activations[-1].cpu().data.numpy()
        grads = self.activations_and_grads.gradients[-1].cpu().data.numpy()
        cam = self.get_cam_image(text_tensor, target_category, activations, grads, eigen_smooth)
        cam = np.maximum(cam, 0)
        
        
        result = [] 
        for img in cam: 
            img = np.float32(img)

            if not compute_text:
                img = cv2.resize(img, input_tensor.shape[-2:][::-1])


            img = img - np.min(img)
            img = img / np.max(img)

            result.append(img)

        result = np.float32(result)
        torch.cuda.empty_cache()
        return result



    # def forward(self, input_tensor, text_tensor, target_category=None, eigen_smooth=False, compute_text=False):

    #     self.text_tensor = text_tensor
    #     self.input_tensor = input_tensor

    #     # if self.cuda:
    #     #     input_tensor = input_tensor.cuda()
    #     #     text_tensor = text_tensor.cuda()

    #     #logit per image반환
    #     output, _ = self.activations_and_grads(input_tensor, text_tensor)

    #     if type(target_category) is int:
    #         target_category = [target_category] * input_tensor.size(0)

    #     if target_category is None:
    #         target_category = np.argmax(output.cpu().data.numpy(), axis=-1)
    #         # print(target_category)
    #     else:
    #         assert(len(target_category) == input_tensor.size(0))

    #     self.model.zero_grad()
    #     loss = self.get_loss(output, target_category)
    #     loss.backward(retain_graph=True)

    #     activations = self.activations_and_grads.activations[-1].cpu().data.numpy()
    #     grads = self.activations_and_grads.gradients[-1].cpu().data.numpy()
    #     # print(len(activations[0]))
    #     # print(len(grads[0]))

    #     cam = self.get_cam_image(text_tensor, target_category, activations, grads, eigen_smooth)

    #     cam = np.maximum(cam, 0)
    #     #print(cam)
    #     result = []
    #     for img in cam:
    #         img = np.float32(img)
    #         if not compute_text:
    #             img = cv2.resize(img, input_tensor.shape[-2:][::-1])
    #         img = img - np.min(img)
    #         img = img / np.max(img)
    #         result.append(img)
    #     result = np.float32(result)
    #     return result

    def forward_augmentation_smoothing(self,
                                       input_tensor,
                                       target_category=None,
                                       eigen_smooth=False):
        transforms = tta.Compose(
            [
                tta.HorizontalFlip(),
                tta.Multiply(factors=[0.9, 1, 1.1]),
            ]
        )
        cams = []
        for transform in transforms:
            augmented_tensor = transform.augment_image(input_tensor)
            cam = self.forward(augmented_tensor,
                target_category, eigen_smooth)

            # The ttach library expects a tensor of size BxCxHxW
            cam = cam[:, None, :, :]
            cam = torch.from_numpy(cam)
            cam = transform.deaugment_mask(cam)

            # Back to numpy float32, HxW
            cam = cam.numpy()
            cam = cam[:, 0, :, :]
            cams.append(cam)

        cam = np.mean(np.float32(cams), axis=0)
        return cam

    def __call__(self,
                 input_tensor,
                 text_tensor,
                 target_category=None,
                 aug_smooth=False,
                 eigen_smooth=False,
                 compute_text=False):
        if aug_smooth is True:
            return self.forward_augmentation_smoothing(input_tensor, text_tensor,
                target_category, eigen_smooth, compute_text)

        return self.forward(input_tensor, text_tensor,
            target_category, eigen_smooth, compute_text)
