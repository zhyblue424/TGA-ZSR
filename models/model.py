import torch
import json
from torch import nn
import numpy as np
import os
IMAGENET_MEAN = (0.48145466, 0.4578275, 0.40821073)
IMAGENET_STD = (0.26862954, 0.26130258, 0.27577711)

mu = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
std = torch.tensor(IMAGENET_STD).view(3, 1, 1)

def normalize(X,device):
    return (X - mu.cuda(device)) / std.cuda(device)

def clip_img_preprocessing(X,device):
    img_size = 224
    X = torch.nn.functional.upsample(X, size=(img_size, img_size), mode='bicubic').cuda(device)
    X = normalize(X,device)
    return X

def create_logits(x1, x2, logit_scale):
    x1 = x1 / x1.norm(dim=-1, keepdim=True)
    x2 = x2 / x2.norm(dim=-1, keepdim=True)
    # cosine similarity as logits
    logits_per_x1 = logit_scale * x1 @ x2.t()
    logits_per_x2 = logit_scale * x2 @ x1.t()
    return logits_per_x1, logits_per_x2

def multiGPU_CLIP_image_logits(images, model, text_tokens,target, device, prompter=None, add_prompter=None):
    image_tokens = clip_img_preprocessing(images, device)
    prompt_token = None if add_prompter is None else add_prompter()
    if prompter is not None:
        image_tokens = prompter(image_tokens)
    return multiGPU_CLIP(model, image_tokens, text_tokens, target, device, prompt_token=prompt_token)[0]

def multiGPU_CLIP(clip_model, images, text_tokens, target, device, prompt_token=None):
    if prompt_token is not None:
        bs = images.size(0)
        prompt_token = prompt_token.repeat(bs, 1, 1)
    if text_tokens.size()[0] == 1000:
        "Processing ImageNet"
        img_embed = clip_model.encode_image(images, prompt_token)[:,0,:]
        img_embed = img_embed / img_embed.norm(dim=-1, keepdim=True)

        text_features = imagenet_text_features(clip_model, text_tokens).to(device)
        scale_text_embed = clip_model.logit_scale.exp() * text_features
    else:
        img_embed, scale_text_embed = clip_model(images, text_tokens, prompt_token)#torch.Size([50, 64, 512]),torch.Size([10, 512])
        # text_features = clip_model.encode_text(text_tokens)#torch.Size([100, 77])
    logits_per_image = img_embed @ scale_text_embed.t()
    logits_per_text = scale_text_embed @ img_embed.t()
    return logits_per_image, logits_per_text, scale_text_embed# text_features[target,:]

def imagenet_text_features(clip_model, text_tokens):
    """Insufficient memory, save memory"""
    dir="./save/imagenet"
    if os.path.exists(dir) is False:
        """save"""
        os.makedirs(dir)
        for i in range(1,11):
            text_features = clip_model.encode_text(text_tokens[100*i-100:100*i,:])
            text_features = text_features.cpu().detach().numpy().tolist()
            file_path = f"text_features_{i}.json"
            with open(dir + str("/"+file_path), "w") as file:
                json.dump(text_features, file)
    """read"""
    text_features=[]
    for i in range(1, 11):
        file_path = f"text_features_{i}.json"
        with open(dir + str("/"+file_path), "r") as file:
            for line in file:
                loaded_data = json.loads(line)
                text_features.extend(loaded_data)
    text_features = torch.tensor(text_features, dtype=torch.float16)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    return text_features