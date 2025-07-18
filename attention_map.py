from util.model import getCLIP, getCAM
import torch

def attention_map(text_features, clip_model, images, prompt_token, args):
    if args.atten_methods=='text':
        """feature extract extraction"""
        image_features = clip_model.encode_image(images,prompt_token)  
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        img_spatial_feat = image_features[:,1:,:]
        
        """Text guided attention map"""
        am = img_spatial_feat @ text_features.unsqueeze(-1)
        am = (am-am.min(1, keepdim=True)[0]) / (am.max(1, keepdim=True)[0] - am.min(1, keepdim=True)[0] )
        """reshape"""
        side = int(am.shape[1] ** 0.5) 
        am = am.reshape(am.shape[0], side, side, -1).permute(0, 3, 1, 2)

        """interpolate"""
        am = torch.nn.functional.interpolate(am, args.image_size, mode='bilinear')
    else:
        """visual based attention map"""
        am=attention_map_visual(clip_model, images, text_features, args)
    return am

def attention_map_visual(clip_model, images, text_features, args):
    target_layer, reshape_transform = getCLIP(clip_model,model_name='ViT-B/32', gpu_id=args.gpu) 
    cam = getCAM(model_name='GradCAM', model=clip_model, target_layer=target_layer,gpu_id=args.gpu, reshape_transform=reshape_transform)
    grayscale_cam = cam(input_tensor=images, text_tensor=text_features)
    am = torch.tensor(grayscale_cam)
    return am

def attention_map_comp(text_features, no_text_features, clip_model, images, prompt_token, args):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    """feature extract extraction"""
    image_features = clip_model.encode_image(images,prompt_token)  
    image_features = image_features / image_features.norm(dim=1, keepdim=True)
    img_spatial_feat = image_features[:,1:,:]
    
    """Text guided attention map"""
    am = img_spatial_feat @ text_features.unsqueeze(-1)
    am = (am-am.min(1, keepdim=True)[0]) / (am.max(1, keepdim=True)[0] - am.min(1, keepdim=True)[0] )
    """reshape"""
    side = int(am.shape[1] ** 0.5) 
    am = am.reshape(am.shape[0], side, side, -1).permute(0, 3, 1, 2)

    """interpolate"""
    resize_transform = transforms.Resize((args.image_size, args.image_size), antialias=True)
    am = resize_transform(am)
    """no Text guided attention map"""
    no_am = img_spatial_feat @ no_text_features.unsqueeze(-1)
    no_am = (no_am-no_am.min(1, keepdim=True)[0]) / (no_am.max(1, keepdim=True)[0] - no_am.min(1, keepdim=True)[0] )
    """reshape"""
    side = int(no_am.shape[1] ** 0.5) 
    no_am = no_am.reshape(no_am.shape[0], side, side, -1).permute(0, 3, 1, 2)

    """interpolate"""
    no_am = resize_transform(no_am)
    
    atten = am * (1 - no_am)
    return atten
    return am.view(am.shape[0], -1), no_am.view(no_am.shape[0], -1)
