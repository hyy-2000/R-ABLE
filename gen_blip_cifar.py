import torch
import torchvision
import torchvision.transforms as transforms
import os
import numpy as np
import argparse

from PIL import Image

from transformers import Blip2Processor, Blip2ForConditionalGeneration

import time 

from tqdm import tqdm

from huggingface_hub import snapshot_download
from huggingface_hub import hf_hub_download

from imbalance_cifar import IMBALANCECIFAR10


'''
#load blipv2 caption generation model

def get_arguments():
    parser = argparse.ArgumentParser(description='One-shot training')
    # Training model hyperparameter settings
    parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
    parser.add_argument('--imb_type', choices=['exp', 'step',None],default='exp', type=str, help='imbalance type')
    parser.add_argument('--imb_factor', default=0.1, type=float, help='imbalance factor')
    parser.add_argument('--if_transform', default=False, type=bool, help='transform image or not')
    parser.add_argument('--epochs', type=int, default=120, help="number of training epochs")
    parser.add_argument('--momentum', type=float, default=0.9, help="SGD momentum")
    parser.add_argument('--batch_size', type=int, default=128, help="batch size for training")
    parser.add_argument('--weight_decay', '--wd', default=2e-4,
                        type=float, metavar='W')
    parser.add_argument('--num_workers', type=int, default=4, help="number of workers for data loading")
    # model setting
    parser.add_argument('--net', type=str, 
                        choices=['holocron_resnet18', 'holocron_resnet34', 'holocron_resnet50', "resnet50"],
                        default="holocron_resnet18",
                        help='model name to train')
    parser.add_argument('--net_path', type=str, 
                        default=None,
                        help='load model weight path')
    # dataset setting 
    parser.add_argument('--data_type', type=str, 
                        choices=["imagenet1000", "domainnet", "imagenet100", "imagenette", "imagefruit", "imageyellow", "imagesquawk"],
                        default="imagenette",
                        help='data set type')
    parser.add_argument('--data_path_train', default=None, type=str, help='data path for train')
    parser.add_argument('--data_path_test', default=None, type=str, help='data path for test')
    parser.add_argument('--sample_data_nums', default=None, type=int, help='sample number of syn images if None samples all data')
    parser.add_argument('--syn', type=int, choices=[0, 1], default=0, help='if syn dataset')
    parser.add_argument('--if_blip', type=int, choices=[0, 1], default=0, help='if use instance-level syn data')
    # domainnet dataset setting
    parser.add_argument('--labels', nargs='+', type=int, 
                        default=[1, 73, 11, 19, 29, 31, 290, 121, 225, 39], #['airplane', 'clock', 'axe', 'basketball', 'bicycle', 'bird', 'strawberry', 'flower', 'pizza', 'bracelet'],
                        help='domainnet subdataset labels')
    parser.add_argument('--domains', nargs='+', type=str, 
                        default=['clipart', 'infograph', 'painting', 'quickdraw', 'real', 'sketch'],
                        help='domainent domain')
    # others setting
    parser.add_argument('--seed', type=int, default=0, help="random seed for reproducibility")
    parser.add_argument('--exp_name', type=str, default="exp_1",
                        help="the name of this run")
    parser.add_argument('--wandb', type=int, default=1,
                        help="set 1 for wandb logging")
    args = parser.parse_args()
    # post processing
    args.syn = (args.syn==1)
    args.if_blip = (args.if_blip==1)
    return args
args=get_arguments()
'''
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.cuda.empty_cache()
trainset = IMBALANCECIFAR10(root='./data', train=True, download=True, transform=transforms.ToTensor())
#snapshot_download(repo_id='Salesforce/blip2-opt-2.7b',local_dir='/root/autodl-tmp/blipv2_pretrained',cache_dir='/root/autodl-tmp/blipv2_pretrained',force_download=True, resume_download=False)

#hf_hub_download(repo_id='Salesforce/blip2-opt-2.7b',filename='pytorch_model-00002-of-00002.bin',local_dir='/root/autodl-tmp/blipv2_pretrained',cache_dir='/root/autodl-tmp/blipv2_pretrained',force_download=True, resume_download=False)

processor = Blip2Processor.from_pretrained("/root/autodl-tmp/blipv2_pretrained")
model = Blip2ForConditionalGeneration.from_pretrained("/root/autodl-tmp/blipv2_pretrained", torch_dtype=torch.float16)
model.to(device)



with open('cifar_0.01.txt','w') as f:
    with tqdm(total=len(trainset),desc='generating captions:',leave=True, ncols=100, unit='B', unit_scale=True) as pbar:
        for i in range(0,len(trainset),200):
            l, r = i, min(i + 200, len(trainset))
            images=list()
            for j in range(l,r):
                tensor = trainset[j][0]
                img_nor = (tensor - tensor.min()) / (tensor.max() - tensor.min())
                rescaled_img = (img_nor * 255).to(torch.uint8)
                to_pil = transforms.ToPILImage()
                pil_image = to_pil(rescaled_img)
                images.append(pil_image)
            inputs=processor(images=images,return_tensors='pt').to(device,torch.float16)

            generated_ids=model.generate(**inputs)
            generated_texts=processor.batch_decode(generated_ids, skip_special_tokens=True)
            #for generated_text in generated_texts:
            for j in range(len(generated_texts)):
                generated_text = generated_texts[j]
                generated_text = generated_text.strip()
                with open('cifar_0.01.txt','a+') as f:
                    f.write(str(trainset[l+j][1]).encode().decode()+'\t'+generated_text+'\n')
                pbar.update(1)
