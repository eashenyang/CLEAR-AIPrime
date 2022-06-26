# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


from inspect import istraceback
import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import ConcatDataset, random_split
from timm.data.constants import \
    IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from timm.data import create_transform




def build_dataset(is_train, args, trial):
    transform = build_transform(is_train, args)

    print("Transform = ")
    if isinstance(transform, tuple):
        for trans in transform:
            print(" - - - - - - - - - - ")
            for t in trans.transforms:
                print(t)
    else:
        for t in transform.transforms:
            print(t)
    print("---------------------------")

 
    print("reading from datapath", args.data_path)
    print("eval in %s mode" % args.eval_domain)
    
    # indomain train and eval
    if args.eval_domain == 'in_domain':
        last_dataset = datasets.ImageFolder(os.path.join(args.data_path, str(trial + args.start_folder)), transform=transform)
        eval_size = int(len(last_dataset) * 0.1)
        
        if is_train:
            dataset_list=[]
            for i in range(trial):
                data_root = os.path.join(args.data_path, str(i + args.start_folder))
                dataset_list.append(datasets.ImageFolder(data_root, transform=transform))
            
            train_last_dataset, _ = random_split(last_dataset, [len(last_dataset) - eval_size, eval_size], generator=torch.Generator().manual_seed(42))
            dataset = ConcatDataset(dataset_list + [train_last_dataset])
        else:
            _, dataset = random_split(last_dataset, [len(last_dataset) - eval_size, eval_size], generator=torch.Generator().manual_seed(42))
            
    # next-domain, forward  train and eval
    else:
        if is_train:
            dataset_list=[]
            for i in range(trial+1):
                data_root = os.path.join(args.data_path, str(i + args.start_folder))
                dataset_list.append(datasets.ImageFolder(data_root, transform=transform))      
            dataset = ConcatDataset(dataset_list)
            
        else:
            eval_folder = []
            if args.eval_domain == 'next':
                if trial==args.n_trials-1:
                    eval_folder.append(os.path.join(args.data_path, str(args.start_folder)))
                else:
                    eval_folder.append(os.path.join(args.data_path, str(trial + 1 + args.start_folder)))
                    
            elif args.eval_domain == 'forward':
                if trial==args.n_trials-1:
                    eval_folder = [os.path.join(args.data_path, str(args.start_folder))]
                else:
                    for i in range(trial+1, args.n_trials):
                        eval_folder.append(os.path.join(args.data_path, str(i + args.start_folder)))
                    
            else:
                raise NotImplementedError

            if len(eval_folder)==1:
                dataset = datasets.ImageFolder(eval_folder[0], transform=transform)
            else:
                dataset = ConcatDataset([datasets.ImageFolder(x, transform=transform) for x in eval_folder])
                
            # print(trial, eval_folder)
            
    n_classes = args.n_classes
        

    print("Number of the class = %d" % n_classes)

    return dataset, n_classes


def build_transform(is_train, args):
    resize_im = args.input_size > 32
    imagenet_default_mean_and_std = args.imagenet_default_mean_and_std
    mean = IMAGENET_INCEPTION_MEAN if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_MEAN
    std = IMAGENET_INCEPTION_STD if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_STD

    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=mean,
            std=std,
        )
        if not resize_im:
            transform.transforms[0] = transforms.RandomCrop(
                args.input_size, padding=4)
        return transform

    t = []
    if resize_im:
        # warping (no cropping) when evaluated at 384 or larger
        if args.input_size >= 384:  
            t.append(
            transforms.Resize((args.input_size, args.input_size), 
                            interpolation=transforms.InterpolationMode.BICUBIC), 
        )
            print(f"Warping {args.input_size} size input images...")
        else:
            if args.crop_pct is None:
                args.crop_pct = 224 / 256
            size = int(args.input_size / args.crop_pct)
            t.append(
                # to maintain same ratio w.r.t. 224 images
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC),  
            )
            t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)
