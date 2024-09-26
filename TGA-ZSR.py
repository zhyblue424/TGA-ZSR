from __future__ import print_function
import cv2
import numpy as np
import argparse, os, time, random
from tqdm import tqdm
import logging
import torch, torchvision
import torch.backends.cudnn as cudnn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torchvision.datasets import *
from replace import clip
from models import prompters
from models.prompters import TokenPrompter,NullPrompter
from models.model import *
from attacks import *
import copy
from utils import accuracy, AverageMeter, ProgressMeter, save_checkpoint
from utils import cosine_lr, convert_models_to_fp32, refine_classname
from utils import load_train_dataset, load_val_datasets, get_text_prompts_train, \
    get_text_prompts_val

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from attention_map import *

def parse_option():
    parser = argparse.ArgumentParser('Adapting CLIP for zero-shot adv robustness')
    parser.add_argument('--print_freq', type=int, default=50, help='print frequency')
    parser.add_argument('--save_freq', type=int, default=50, help='save frequency')
    parser.add_argument('--validate_freq', type=int, default=2, help='validate frequency')
    parser.add_argument('--batch_size', type=int, default=128, help='batch_size')
    parser.add_argument('--epochs', type=int, default=10, help='number of training epochs')

    # optimization
    parser.add_argument('--Method', type=str, default='TGA-ZSR')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='learning rate')
    parser.add_argument("--weight_decay", type=float, default=0, help="weight decay")
    parser.add_argument("--warmup", type=int, default=1000, help="number of steps to warmup for")
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')

    # adversarial attack
    parser.add_argument('--train_eps', type=float, default=1, help='momentum')
    parser.add_argument('--train_numsteps', type=int, default=2)
    parser.add_argument('--train_stepsize', type=int, default=1)
    parser.add_argument('--test_eps', type=float, default=1, help='momentum')
    parser.add_argument('--test_numsteps', type=int, default=100)
    parser.add_argument('--test_stepsize', type=int, default=1)
    
    # model
    parser.add_argument('--patience', type=int, default=1000)
    parser.add_argument('--model', type=str, default='clip')
    parser.add_argument('--arch', type=str, default='vit_b32')
    parser.add_argument('--method', type=str, default='null_patch',
                        choices=['padding', 'random_patch', 'fixed_patch', 'null_patch'],
                        help='choose visual prompting method')
    # parser.add_argument('--prompt_size', type=int, default=30, help='size for visual prompts')
    # parser.add_argument('--add_prompt_size', type=int, default=10, help='size for additional visual prompts')

    # dataset
    parser.add_argument('--root', type=str, default='./data', 
                        help='dataset')
    parser.add_argument('--dataset', type=str, default='tinyImageNet',
                        choices=['cifar100', 'ImageNet', 'cifar10', 'tinyImageNet'], help='Data set for training')
    parser.add_argument('--image_size', type=int, default=224, help='image size')

    # other
    parser.add_argument('--seed', type=int, default=0, help='seed for initializing training')
    parser.add_argument('--model_dir', type=str, default='./save/models', 
                        help='path to save models')
    parser.add_argument('--filename', type=str, default=None, 
                        help='filename to save')
    parser.add_argument('--trial', type=int, default=1, help='number of trials')
    parser.add_argument('--resume', type=str, default=None,
                        help='path to resume from checkpoint')

    parser.add_argument('--gpu', type=int, default=0, help='gpu to use')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--VPbaseline', action='store_true')
    parser.add_argument('--attack', choices=['pgd', 'autoattack'], default='pgd')
    parser.add_argument('--noimginprop', action='store_true')
    
    #FT
    parser.add_argument('--last_num_ft', type=int, default=0)
    parser.add_argument('--adaptation_method', type=str, default='FT', choices=['VPT','FT'],
                        help='choose visual adaptation method')
    parser.add_argument('--Distance_metric', type=str, default='l2', choices=['cos', 'l2', 'l1'],
                        help='Select the distance measure in the loss function')
    parser.add_argument('--atten_methods',type=str,default='text',choices=['text','visual'])
    parser.add_argument('--Alpha', type=float, default=0.08, help='L_AR in Equ.6')
    parser.add_argument('--Beta', type=float, default=0.05, help='L_AMC in Equ.7')
    parser.add_argument('--testdata', type=str, nargs='+')
    args = parser.parse_args()

    args.filename = '{}_{}_{}_{}_lr-{}_decay-{}_bsz-{}_warmup-{}_trial-{}_Alpha-{}_Beta-{}_distance-{}_atten_methods-{}'. \
        format(args.Method, args.dataset, args.model, args.arch, args.learning_rate, 
               args.weight_decay, args.batch_size, args.warmup, args.trial, args.Alpha, 
               args.Beta, args.Distance_metric, args.atten_methods)
    return args

def main():
    global best_acc1, device, logger
    args = parse_option()
    device = torch.device("cuda:{}".format(args.gpu))
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    log_dir = './save/loggers/'
    os.makedirs(log_dir, exist_ok=True)
    file_handler = logging.FileHandler(os.path.join(log_dir,f'{args.filename}.log'))
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s [%(filename)s] => %(message)s")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    args.train_eps = args.train_eps / 255.
    args.test_eps = args.test_eps / 255.
    args.train_stepsize = args.train_stepsize / 255.
    args.test_stepsize = args.test_stepsize / 255.

    args_dict = vars(args)
    for key, value in args_dict.items():
        print(f'{key}: {value}')
        logger.info(f'{key}: {value}')

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
    """create model"""
    if args.adaptation_method == 'VPT':
        add_prompt_len = args.add_prompt_size
    else:
        add_prompt_len = 0
    print(" create model")
    model, preprocess = clip.load('ViT-B/32', device, jit=False, prompt_len=add_prompt_len)

    convert_models_to_fp32(model)
    model = model.to(device)
    frozen_model = copy.deepcopy(model).to(device)
    
    model.eval()
    frozen_model.eval() 
    
    """define criterion and optimizer"""
    if args.adaptation_method == 'VPT':
        prompter = prompters.__dict__[args.method](args).to(device)
        add_prompter = TokenPrompter(args.add_prompt_size).to(device)
        optimizer = torch.optim.SGD(list(prompter.parameters()) + list(add_prompter.parameters()),
                                lr=args.learning_rate,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    else:
        prompter = NullPrompter().to(device)
        add_prompter = TokenPrompter(0).to(device)
        if args.last_num_ft == 0:
            optimizer = torch.optim.SGD(model.visual.parameters(),
                                        lr=args.learning_rate,
                                        momentum=args.momentum,
                                        weight_decay=args.weight_decay)
        else:
            optimizer = torch.optim.SGD(list(model.visual.parameters())[-args.last_num_ft:],
                                        lr=args.learning_rate,
                                        momentum=args.momentum,
                                        weight_decay=args.weight_decay)
    
    """Load the pre-trained model"""
    args.start_epoch = 0
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']

            if 'vision_encoder_state_dict' in checkpoint.keys():
                model.visual.load_state_dict(checkpoint['vision_encoder_state_dict'], strict=False)
            else:
                prompter.load_state_dict(checkpoint['state_dict'])
                add_prompter.load_state_dict(checkpoint['add_prompter'])
            print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
            logger.info("loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    template = 'This is a photo of a {}'
    print(f'template: {template}')


    """load training dataset"""
    train_dataset = load_train_dataset(args)
    
    """load val dataset(s)"""
    if args.testdata is None:
        val_dataset_name = ['tinyImageNet','cifar10', 'cifar100','STL10','Food101','oxfordpet','flowers102','dtd','EuroSAT',\
                            'fgvc_aircraft','Caltech101','Caltech256','StanfordCars','PCAM','ImageNet','SUN397']
    else:
        val_dataset_name = args.testdata
    val_dataset_list = load_val_datasets(args, val_dataset_name)


    """create dataloaders"""
    train_sampler = None
    val_sampler = None

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, pin_memory=True,
                               shuffle=True, sampler=train_sampler)

    val_loader_list = [DataLoader(each, batch_size=args.batch_size*2, pin_memory=True,
                                   shuffle=False, sampler=val_sampler) for each in val_dataset_list]

    """get text prompts for training/val"""
    texts_train = get_text_prompts_train(args, train_dataset, template=template)
    texts_list = get_text_prompts_val(val_dataset_list, val_dataset_name, template=template)
    
    scaler = GradScaler()
    total_steps = len(train_loader) * args.epochs
    scheduler = cosine_lr(optimizer, args.learning_rate, args.warmup, total_steps)

    cudnn.benchmark = True
    args.model_folder = os.path.join(args.model_dir, args.filename)
    if not os.path.isdir(args.model_folder):
        os.makedirs(args.model_folder)

    epochs_since_improvement = 0
    

    """training"""
    for epoch in range(args.epochs):
        # train for one epoch
        train(train_loader, texts_train, model,frozen_model, prompter, add_prompter, optimizer, scheduler,
              scaler, epoch,  args)
        
        # evaluate on validation set
        if epoch % args.validate_freq == 0:
            acc1_mean = validate(val_loader_list, val_dataset_name, texts_list, model,frozen_model,optimizer, device,
                                 prompter, add_prompter, args)
            
        # remember best acc@1 and save checkpoint
        is_best = acc1_mean > best_acc1
        best_acc1 = max(acc1_mean, best_acc1)

        save_checkpoint({
            'epoch': args.start_epoch + epoch + 1,
            'state_dict': prompter.state_dict(),
            'add_prompter': add_prompter.state_dict(),
            'best_acc1': best_acc1,
            'optimizer': optimizer.state_dict(),
            'vision_encoder_state_dict':model.visual.state_dict(),
        }, args, is_best=is_best)

        if is_best:
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1
            print(f"There's no improvement for {epochs_since_improvement} epochs.")
            logger.info(f"There's no improvement for {epochs_since_improvement} epochs.")
            if epochs_since_improvement >= args.patience:
                print("The training halted by early stopping criterion.")
                logger.info("The training halted by early stopping criterion.")
                break

"""train function"""
def train(train_loader, texts, model,frozen_model, prompter, add_prompter,
          optimizer, scheduler, scaler, epoch,  args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1],
        prefix="Epoch: [{}]".format(args.start_epoch + epoch))

    """switch to train mode"""
    prompter.train()
    add_prompter.train()
    model.visual.train()
    num_batches_per_epoch = len(train_loader)

    alpha = args.train_stepsize
    attack_iters = args.train_numsteps

    end = time.time()
    for i, (images, target) in enumerate(tqdm(train_loader)):

        # measure data loading time
        data_time.update(time.time() - end)
        BATCH_SIZE = images.size(0)

        # adjust learning rate
        step = num_batches_per_epoch * epoch + i
        scheduler(step)

        optimizer.zero_grad()

        images = images.to(device)
        target = target.to(device)
        text_tokens = clip.tokenize(texts).to(device)
        

        # with automatic mixed precision
        with autocast():
            """Build adversarial example"""
            if not args.VPbaseline:
                delta = attack_pgd(prompter, model,add_prompter,images,
                                target, text_tokens, alpha, attack_iters, 'l_inf',
                                device=device, args=args, epsilon=args.train_eps)
                tmp = clip_img_preprocessing(images + delta,device)
            else:
                tmp = clip_img_preprocessing(images,device)

            prompted_images = prompter(tmp)
            clean_images = prompter(clip_img_preprocessing(images,device))
            prompt_token = add_prompter()

            output, _ , text_features= multiGPU_CLIP(model, prompted_images, text_tokens, target, device, prompt_token)
            text_features = text_features[target,:]

            """Calculated to gain attention"""
            attack_tar = attention_map(text_features, model, prompted_images, prompt_token, args).view(prompted_images.size()[0], -1)
            clean_ori = attention_map(text_features, frozen_model, clean_images, prompt_token, args).view(prompted_images.size()[0], -1)
            clean_tar = attention_map(text_features, model, clean_images, prompt_token, args).view(prompted_images.size()[0], -1)
            
            loss_TeCoA ,loss_SoftCe, loss_AM1 ,loss_AM2=criterion(model, output, target, attack_tar, clean_ori, clean_tar, args)
            loss = loss_TeCoA +loss_SoftCe +loss_AM1 + loss_AM2
            scaler.scale(loss).backward()
            scaler.step(optimizer)
        scaler.update()

        model.logit_scale.data = torch.clamp(model.logit_scale.data, 0, 4.6052)   
        # measure accuracy
        acc1 = accuracy(output, target, topk=(1,))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0].item(), images.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            entries = progress.display(i)
            logger.info(entries)
            logger.info("TeCoA Loss: %f, AM1 Loss: %f, AM2 Loss: %f", loss_TeCoA, loss_AM1, loss_AM2)
            if args.debug:
                break
    save_checkpoint({
        'epoch': args.start_epoch + epoch + 1,
        'state_dict': prompter.state_dict(),
        'add_prompter': add_prompter.state_dict(),
        'best_acc1': best_acc1,
        'optimizer': optimizer.state_dict(),
        'vision_encoder_state_dict':model.visual.state_dict(),
        }, args)
    return losses.avg, top1.avg


def validate(val_loader_list, val_dataset_name, texts_list, model,frozen_model,optimizer, device,
                prompter, add_prompter, args):
    dataset_num = len(val_loader_list)
    acc_all = []

    test_stepsize = args.test_stepsize

    for cnt in range(dataset_num):

        val_loader = val_loader_list[cnt]
        texts = texts_list[cnt]
        dataset_name = val_dataset_name[cnt]

        binary = ['PCAM', 'hateful_memes']
        attacks_to_run=['apgd-ce', 'apgd-dlr']
        if dataset_name in binary:
            attacks_to_run=['apgd-ce']
            
        batch_time = AverageMeter('Time', ':6.3f')
        losses = AverageMeter('Loss', ':.4e')
        top1_org = AverageMeter('Original Acc@1', ':6.2f')
        top1_adv_org = AverageMeter('Adv Original Acc@1', ':6.2f')

        progress = ProgressMeter(
            len(val_loader),
            [batch_time, losses, top1_org, top1_adv_org],
            prefix=dataset_name + '_Validate: ')

        # switch to evaluation mode
        prompter.eval()
        add_prompter.eval()
        model.eval()
        model.zero_grad()
        frozen_model.eval()

        end = time.time()
        for i, (images, target) in enumerate(tqdm(val_loader)):
            images = images.to(device)
            target = target.to(device)
            text_tokens = clip.tokenize(texts).to(device)

            with autocast():

                # compute output
                with torch.no_grad():
                    """clean images"""
                    prompt_token = None
                    output_org = multiGPU_CLIP(model, clip_img_preprocessing(images,device),text_tokens,target, device, None)[0]

                    acc1 = accuracy(output_org, target, topk=(1,))
                    top1_org.update(acc1[0].item(), images.size(0))

                """adv images"""
                if args.attack == 'pgd':
                    delta_noprompt = attack_pgd(None, model, None, images, target, text_tokens,
                                        test_stepsize, args.test_numsteps,'l_inf',device, args, epsilon=args.test_eps)
                    attacked_images = images + delta_noprompt
                else:
                    attacked_images  = attack_auto(model, images, target, text_tokens, None, None, device,
                                            attacks_to_run=attacks_to_run, epsilon=args.test_eps)

                # torch.cuda.empty_cache()
                with torch.no_grad():
                    output_org_adv, _, text_features= multiGPU_CLIP(model, clip_img_preprocessing(attacked_images,device),
                                                        text_tokens, target, device, None)
                    
                    text_features = text_features[target,:]
                    attack_atten =attention_map(text_features, model, clip_img_preprocessing(attacked_images,device), prompt_token, args).view(images.size()[0], -1)
                    clean_atten = attention_map(text_features, frozen_model, clip_img_preprocessing(images,device), prompt_token, args).view(images.size()[0], -1)
                    clean_atten_model = attention_map(text_features, model, clip_img_preprocessing(images,device), prompt_token, args).view(images.size()[0], -1)
                    # torch.cuda.empty_cache()
                    loss_TeCoA ,loss_SoftCe, loss_AM1 ,loss_AM2=criterion(model, output_org_adv, target, attack_atten, clean_atten,clean_atten_model, args)
                    loss = loss_TeCoA +loss_SoftCe +loss_AM1 + loss_AM2
                    losses.update(loss.item(), images.size(0))
                    
                    acc1 = accuracy(output_org_adv, target, topk=(1,))
                    top1_adv_org.update(acc1[0].item(), images.size(0))
                # torch.cuda.empty_cache()
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                entries = progress.display(i)
                logger.info(entries)
                if args.debug:
                    break
        torch.cuda.empty_cache()
        print(dataset_name + ' * Adv Original Acc@1 {top1_adv_org.avg:.3f}' '* Original Acc@1 {top1_org.avg:.3f}'
              .format(top1_adv_org=top1_adv_org, top1_org=top1_org))
        logger.info(dataset_name + ' * Adv Original Acc@1 {top1_adv_org.avg:.3f} ' '* Original Acc@1 {top1_org.avg:.3f}'
              .format(top1_adv_org=top1_adv_org, top1_org=top1_org))
        
        acc_all.append(top1_adv_org.avg)
    return np.mean(acc_all)

def criterion(model, output, target, adv_atten, clean_atten, clean_atten_model, args):
    """Cross entropy loss"""
    CrossEntropyLoss = torch.nn.CrossEntropyLoss().to(device)
    loss_TeCoA = CrossEntropyLoss(output, target)
    
    """attention map loss"""
    if args.Distance_metric == 'cos':
        loss_AM1 = torch.mean(1-torch.nn.functional.cosine_similarity(adv_atten, clean_atten, dim=1, eps=1e-8))
        loss_AM2 = torch.mean(1-torch.nn.functional.cosine_similarity(clean_atten_model, clean_atten, dim=1, eps=1e-8))
    elif args.Distance_metric == 'l2':
        loss_AM1 = torch.mean(torch.norm(adv_atten - clean_atten,dim=1, p=2))
        loss_AM2 = torch.mean(torch.norm(clean_atten_model - clean_atten,dim=1, p=2))
    elif args.Distance_metric == 'l1':
        l1_loss = torch.nn.L1Loss(reduction='mean')
        loss_AM1 = l1_loss(adv_atten, clean_atten)
        loss_AM2 = l1_loss(clean_atten_model, clean_atten)
    return loss_TeCoA ,args.Alpha*loss_AM1 ,args.Beta*loss_AM2


if __name__ == '__main__':
    main()
