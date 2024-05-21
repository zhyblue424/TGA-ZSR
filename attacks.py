import torch
from utils import one_hot_embedding
from models.model import *
import torch.nn.functional as F
from autoattack import AutoAttack
import functools
import gc
lower_limit, upper_limit = 0, 1
def clamp(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)


def attack_pgd(prompter, model, add_prompter,  X, target, text_tokens, alpha,
               attack_iters, norm,device, args, restarts=1, early_stop=True, epsilon=0):
    delta = torch.zeros_like(X).cuda(device)
    if norm == "l_inf":
        delta.uniform_(-epsilon, epsilon)
    elif norm == "l_2":
        delta.normal_()
        d_flat = delta.view(delta.size(0), -1)
        n = d_flat.norm(p=2, dim=1).view(delta.size(0), 1, 1, 1)
        r = torch.zeros_like(n).uniform_(0, 1)
        delta *= r / n * epsilon
    else:
        raise ValueError
    delta = clamp(delta, lower_limit - X, upper_limit - X)
    delta.requires_grad = True
    for i in range(attack_iters):
        _images = clip_img_preprocessing(X + delta,device)
        if prompter is not None:
            prompted_images = prompter(_images)
        else:
            prompted_images = _images
        prompt_token = add_prompter() if add_prompter is not None else None

        output, _, _= multiGPU_CLIP(model, prompted_images, text_tokens,target,device, prompt_token)
        CrossEntropyLoss = torch.nn.CrossEntropyLoss().to(device)
        loss = CrossEntropyLoss(output, target)
        loss.backward()
        grad = delta.grad.detach()
        d = delta[:, :, :, :]
        g = grad[:, :, :, :]
        x = X[:, :, :, :]
        if norm == "l_inf":
            d = torch.clamp(d + alpha * torch.sign(g), min=-epsilon, max=epsilon)
        elif norm == "l_2":
            g_norm = torch.norm(g.view(g.shape[0], -1), dim=1).view(-1, 1, 1, 1)
            scaled_g = g / (g_norm + 1e-10)
            d = (d + scaled_g * alpha).view(d.size(0), -1).renorm(p=2, dim=0, maxnorm=epsilon).view_as(d)
        d = clamp(d, lower_limit - x, upper_limit - x)
        delta.data[:, :, :, :] = d
        delta.grad.zero_()
    return delta

def attack_auto(model, images, target, text_tokens, prompter, add_prompter,device,
                         attacks_to_run=['apgd-ce', 'apgd-dlr'], epsilon=0):

    forward_pass = functools.partial(
        multiGPU_CLIP_image_logits,
        model=model, text_tokens=text_tokens, target=target, device=device,
        prompter=None, add_prompter=None
    )

    adversary = AutoAttack(forward_pass, norm='Linf', eps=epsilon, version='standard', verbose=False, device=device)
    adversary.attacks_to_run = attacks_to_run
    x_adv = adversary.run_standard_evaluation(images, target, bs=images.shape[0])
    return x_adv