from replace import clip
from pytorch_grad_cam_modified import GradCAM, ScoreCAM, GradCAMPlusPlus, XGradCAM, EigenCAM, EigenGradCAM, GuidedBackpropReLUModel, LayerCAM 
from pytorch_grad_cam_modified import GradCAM_original, ScoreCAM_original, GradCAMPlusPlus_original, XGradCAM_original, EigenCAM_original, EigenGradCAM_original, GuidedBackpropReLUModel_original, LayerCAM_original
from torchvision.models import resnet50, resnet101
import torch
import timm

def getCLIP(model,model_name, gpu_id):
    reshape_transform = None
    if model_name == "RN50":
        target_layer = model.visual.layer4[-1]
    elif model_name == "RN101":
        target_layer = model.visual.layer4[-1]
    elif model_name == "RN50x4":
        target_layer = model.visual.layer4[-1] 
    elif model_name == "RN50x16":
        target_layer = model.visual.layer4[-1]
    elif model_name == "ViT-B/32":
        target_layer = model.visual.transformer.resblocks[-1].ln_1
        reshape_transform = reshapeTransform7 
    elif model_name == "ViT-B/16":
        target_layer = model.visual.transformer.resblocks[-1].ln_1
        reshape_transform = reshapeTransform14 
    elif model_name == "RN50-pretrained":
        target_layer = model.layer4[-1]
    elif model_name == "DeiT-pretrained":
        model = torch.hub.load('facebookresearch/deit:main', 'deit_base_patch16_224', pretrained=True).to(gpu_id)
        target_layer = model.blocks[-1].norm1
        reshape_transform = reshapeTransform_original
    elif model_name == "ViT-B/16-pretrained":
        model = timm.create_model("vit_base_patch16_224", pretrained=True).to(gpu_id)
        model.eval()
        target_layer = model.blocks[-1].norm1
        reshape_transform = reshape_transform_vitb16
    elif model_name == "ViT-B/32-pretrained":
        model = timm.create_model("vit_base_patch32_224", pretrained=True).to(gpu_id)
        model.eval()
        target_layer = model.blocks[-1].norm1
        reshape_transform = reshape_transform_vitb32
    elif model_name == "RN101-pretrained":
        model = resnet101(pretrained=True).to(gpu_id)
        target_layer = model.layer4[-1]
    return target_layer, reshape_transform

def reshapeTransform7(x):
    x = x[1:, : , :]
    x = x.permute(1,2,0) #N * D * L-1 (width * height)
    x = x.reshape(x.shape[0], x.shape[1], 7, 7)
    return x

def reshapeTransform14(x):
    x = x[1:, : , :]
    x = x.permute(1,2,0) #N * D * L-1 (width * height)
    x = x.reshape(x.shape[0], x.shape[1], 14, 14)
    return x

def reshape_transform_vitb16(tensor, height=14, width=14):
    result = tensor[:, 1 :  , :].reshape(tensor.size(0),
        height, width, tensor.size(2))

    result = result.transpose(2, 3).transpose(1, 2)
    return result

def reshape_transform_vitb32(tensor, height=7, width=7):
    result = tensor[:, 1 :  , :].reshape(tensor.size(0),
        height, width, tensor.size(2))

    result = result.transpose(2, 3).transpose(1, 2)
    return result

def getCAM(model_name, model, target_layer, gpu_id, reshape_transform):
    if model_name == "GradCAM":
        cam = GradCAM(model=model, target_layer=target_layer, gpu_id=gpu_id, reshape_transform = reshape_transform)
    elif model_name == "GradCAMPlusPlus":
        cam = GradCAMPlusPlus(model=model, target_layer=target_layer, gpu_id=gpu_id, reshape_transform = reshape_transform)
    elif model_name == "XGradCAM":
        cam = XGradCAM(model=model, target_layer=target_layer, gpu_id=gpu_id, reshape_transform = reshape_transform)
    elif model_name == "ScoreCAM":
        cam = ScoreCAM(model=model, target_layer=target_layer, gpu_id=gpu_id, reshape_transform = reshape_transform)
    elif model_name == "EigenCAM":
        cam = EigenCAM(model=model, target_layer=target_layer, gpu_id=gpu_id, reshape_transform = reshape_transform)
    elif model_name == "EigenGradCAM":
        cam = EigenGradCAM(model=model, target_layer=target_layer, gpu_id=gpu_id, reshape_transform = reshape_transform)
    elif model_name == "GuidedBackpropReLUModel":
        cam = GuidedBackpropReLUModel(model=model, gpu_id=gpu_id)
    elif model_name == "LayerCAM":
        cam = LayerCAM(model=model, target_layer=target_layer, gpu_id=gpu_id, reshape_transform = reshape_transform)
    elif model_name == "GradCAM_original":
        cam = GradCAM_original(model=model, target_layer=target_layer, gpu_id=gpu_id, reshape_transform = reshape_transform)
    elif model_name == "GradCAMPlusPlus_original":
        cam = GradCAMPlusPlus_original(model=model, target_layer=target_layer, gpu_id=gpu_id, reshape_transform = reshape_transform)
    elif model_name == "XGradCAM_original":
        cam = XGradCAM_original(model=model, target_layer=target_layer, gpu_id=gpu_id, reshape_transform = reshape_transform)
    elif model_name == "ScoreCAM_original":
        cam = ScoreCAM_original(model=model, target_layer=target_layer, gpu_id=gpu_id, reshape_transform = reshape_transform)
    elif model_name == "EigenGradCAM_original":
        cam = EigenGradCAM_original(model=model, target_layer=target_layer, gpu_id=gpu_id, reshape_transform = reshape_transform)
    elif model_name == "EigenCAM_original":
        cam = EigenCAM_original(model=model, target_layer=target_layer, gpu_id=gpu_id, reshape_transform = reshape_transform)
    elif model_name == "GuidedBackpropReLUModel_original":
        cam = GuidedBackpropReLUModel_original(model=model, gpu_id=gpu_id)
    elif model_name == "LayerCAM_original":
        cam = LayerCAM_original(model=model, target_layer=target_layer, gpu_id=gpu_id, reshape_transform = reshape_transform)
    return cam

def reshapeTransform_original(tensor, height=14, width=14):
    result = tensor[:, 1 :  , :].reshape(tensor.size(0), 
        height, width, tensor.size(2))

    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result

def getFineTune(model_name, model, out_feature):
    if model_name == "RN50-pretrained":
        model.fc = torch.nn.Linear(in_features=2048, out_features = out_feature)
    elif model_name == "ViT-B/16-pretrained":
        model.head = torch.nn.Linear(in_features=768, out_features = out_feature)
    elif model_name == "ViT-B/32-pretrained":
        model.head = torch.nn.Linear(in_features=768, out_features = out_feature)
    elif model_name == "RN101-pretrained":
        model.fc = torch.nn.Linear(in_features=2048, out_features = out_feature)
    # for para in model.parameters():
    #     para.require_grad = False
    # print(model)

    return model
