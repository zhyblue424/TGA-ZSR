import shutil
import os
import pickle
import torch
import numpy as np
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10, CIFAR100, STL10, ImageFolder
from replace.datasets import caltech, country211,dtd,eurosat,fgvc_aircraft,food101,flowers102,oxford_iiit_pet,pcam,stanford_cars,sun397
from typing import Any, Callable, Optional, Tuple
from PIL import Image
# from utils import load_imagenet_folder2name
def convert_models_to_fp32(model):
    for p in model.parameters():
        p.data = p.data.float()
        if p.grad:
            p.grad.data = p.grad.data.float()


def refine_classname(class_names):
    for i, class_name in enumerate(class_names):
        class_names[i] = class_name.lower().replace('_', ' ').replace('-', ' ').replace('/', ' ')
    return class_names

def save_checkpoint(state, args, is_best=False, filename='checkpoint.pth.tar'):
    savefile = os.path.join(args.model_folder, filename)
    bestfile = os.path.join(args.model_folder, 'model_best.pth.tar')
    torch.save(state, savefile)
    if is_best:
        shutil.copyfile(savefile, bestfile)
        print ('saved best file')

def assign_learning_rate(optimizer, new_lr):
    for param_group in optimizer.param_groups:
        param_group["lr"] = new_lr

def _warmup_lr(base_lr, warmup_length, step):
    return base_lr * (step + 1) / warmup_length


def cosine_lr(optimizer, base_lr, warmup_length, steps):
    def _lr_adjuster(step):
        if step < warmup_length:
            lr = _warmup_lr(base_lr, warmup_length, step)
        else:
            e = step - warmup_length
            es = steps - warmup_length
            lr = 0.5 * (1 + np.cos(np.pi * e / es)) * base_lr
        assign_learning_rate(optimizer, lr)
        return lr
    return _lr_adjuster


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))
        return entries
    
    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def load_imagenet_folder2name(path):
    dict_imagenet_folder2name = {}
    with open(path) as f:
        line = f.readline()
        while line:
            split_name = line.strip().split()
            cat_name = split_name[2]
            id = split_name[0]
            dict_imagenet_folder2name[id] = cat_name
            line = f.readline()
    # print(dict_imagenet_folder2name)
    return dict_imagenet_folder2name



def one_hot_embedding(labels, num_classes,device):
    """Embedding labels to one-hot form.
    Args:
      labels: (LongTensor) class labels, sized [N,].
      num_classes: (int) number of classes.
    Returns:
      (tensor) encoded labels, sized [N, #classes].
    """
    y = torch.eye(num_classes).to(device)
    return y[labels]


preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])
preprocess224 = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor()
])

def load_train_dataset(args):
    if args.dataset == 'cifar100':
        return CIFAR100('./datasets/CIFAR100', transform=preprocess, download=True, train=True)
    elif args.dataset == 'cifar10':
        return CIFAR10('./datasets/CIFAR10', transform=preprocess, download=True, train=True)
    elif args.dataset == 'ImageNet':
        print(f"Loading ImageNet from {'./datasets/ImageNet'}")
        return ImageFolder(os.path.join('./datasets/ImageNet', 'train'), transform=preprocess224)
    elif args.dataset == 'tinyImageNet':
        print(f"Loading tinyImageNet from {'./datasets/tiny-imagenet-200'}")
        return ImageFolder('./datasets/tiny-imagenet-200/train/', transform=preprocess224)
    else:
        print(f"Train dataset {args.dataset} not implemented")
        raise NotImplementedError

def load_val_datasets(args, val_dataset_names):
    val_dataset_list = []
    for each in val_dataset_names:
        if each == 'cifar10':
            val_dataset_list.append(CIFAR10('./datasets/CIFAR10', transform=preprocess,
                               download=True, train=False))
        elif each == 'cifar100':
            val_dataset_list.append(CIFAR100('./datasets/CIFAR100', transform=preprocess,
                                            download=True, train=False))                                                         
        elif each == 'Caltech101':
            val_dataset_list.append(caltech.Caltech101('./datasets/', target_type='category', transform=preprocess224,
                                             download=True))
        elif each == 'PCAM':
            val_dataset_list.append(pcam.PCAM('./datasets/', split='test', transform=preprocess224,
                                             download=True))
        elif each == 'STL10':
            val_dataset_list.append(STL10('./datasets/STL10', split='test',
                                               transform=preprocess, download=True))
        elif each == 'SUN397':
            val_dataset_list.append(sun397.SUN397('./datasets/',
                                               transform=preprocess224, download=False))
        elif each == 'StanfordCars':
            val_dataset_list.append(stanford_cars.StanfordCars('./datasets/', split='test',
                                           transform=preprocess224, download=False))
        elif each == 'Food101':
            val_dataset_list.append(food101.Food101('./datasets/', split='test',
                                           transform=preprocess224, download=True))
        elif each == 'oxfordpet':
            val_dataset_list.append(oxford_iiit_pet.OxfordIIITPet('./datasets/', split='test',
                                           transform=preprocess224, download=True))
        elif each == 'EuroSAT':
            val_dataset_list.append(eurosat.EuroSAT('./datasets/',
                                           transform=preprocess224, download=True))
        elif each == 'Caltech256':
            val_dataset_list.append(caltech.Caltech256('./datasets/', transform=preprocess224,
                                             download=True))
        elif each == 'flowers102':
            val_dataset_list.append(flowers102.Flowers102('./datasets/', split='test',
                                           transform=preprocess224, download=True))
        elif each == 'Country211':
            val_dataset_list.append(country211.Country211('./datasets/', split='test',
                                           transform=preprocess224, download=True))
        elif each == 'dtd':
            val_dataset_list.append(dtd.DTD('./datasets/', split='test',
                                           transform=preprocess224, download=True))
        elif each == 'fgvc_aircraft':
            val_dataset_list.append(fgvc_aircraft.FGVCAircraft('./datasets/', split='test',
                                           transform=preprocess224, download=True))
        # elif each == 'hateful_memes':
        #     val_dataset_list.append(HatefulMemes(args.root, splits=['test_seen', 'test_unseen'],
        #                                    transform=preprocess))
        elif each == 'ImageNet':
            val_dataset_list.append(ImageFolder(os.path.join('./datasets/ImageNet', 'val'), transform=preprocess224))
        elif each == 'tinyImageNet':
            val_dataset_list.append(ImageFolder('./datasets/tiny-imagenet-200/val/', transform=preprocess224))
        elif each == 'ImageNet-A':
            val_dataset_list.append(ImageFolder('./datasets/imagenet-a/', transform=preprocess224))
        elif each == 'ImageNet-R':
            val_dataset_list.append(ImageFolder('./datasets/imagenet-r/', transform=preprocess224))
        elif each == 'ImageNet-O':
            val_dataset_list.append(ImageFolder('./datasets/imagenet-o/', transform=preprocess224))
        else:
            print(f"Val dataset {each} not implemented")
            raise NotImplementedError
    return val_dataset_list

def get_text_prompts_train(args, train_dataset, template='This is a photo of a {}'):
    class_names = train_dataset.classes
    if args.dataset == 'ImageNet':
        folder2name = load_imagenet_folder2name('./imagenet_classes_names.txt')
        new_class_names = []
        for each in class_names:
            new_class_names.append(folder2name[each])
        class_names = new_class_names
    elif args.dataset == 'tinyImageNet':
        folder2name = load_imagenet_folder2name('./tinyimagenet_classes_name.txt')
        new_class_names = []
        for each in class_names:
            new_class_names.append(folder2name[each])
        class_names = new_class_names
        
    class_names = refine_classname(class_names)
    texts_train = [template.format(label) for label in class_names]
    return texts_train

def get_text_prompts_val(val_dataset_list, val_dataset_name, template='This is a photo of a {}'):
    texts_list = []
    for cnt, each in enumerate(val_dataset_list):
        if hasattr(each, 'clip_prompts'):
            texts_tmp = each.clip_prompts
        else:
            try:
                class_names = each.classes
            except AttributeError:
                class_names = each.categories
            # class_names = each.classes
            if val_dataset_name[cnt] == 'ImageNet' or val_dataset_name[cnt] == 'ImageNet-A' or val_dataset_name[cnt] == 'ImageNet-R' or val_dataset_name[cnt] == 'ImageNet-O':
                folder2name = load_imagenet_folder2name('./imagenet_classes_names.txt')
                new_class_names = []
                for class_name in class_names:
                    new_class_names.append(folder2name[class_name])
                class_names = new_class_names
            elif val_dataset_name[cnt] == 'tinyImageNet':
                folder2name = load_imagenet_folder2name('./tinyimagenet_classes_name.txt')
                new_class_names = []
                for each in class_names:
                    new_class_names.append(folder2name[each])
                class_names = new_class_names
            class_names = refine_classname(class_names)
            texts_tmp = [template.format(label) for label in class_names]
        texts_list.append(texts_tmp)
    assert len(texts_list) == len(val_dataset_list)
    return texts_list
