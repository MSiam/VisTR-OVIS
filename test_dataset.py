"""
Training script of VisTR
Modified from DETR (https://github.com/facebookresearch/detr)
"""
import argparse
import datetime
import json
import random
import time
from pathlib import Path
from tqdm import tqdm

import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler

import datasets
import util.misc as utils
from datasets import build_dataset, get_coco_api_from_dataset
from util.vis_utils import denormalize_img, denormalize_box, create_overlay
import os
import cv2

def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--num_frames', default=36, type=int,
                        help="Number of frames")
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")
    parser.add_argument('--num_workers', default=4, type=int)
    # dataset parameters
    parser.add_argument('--dataset_file', default='ytvos')
    parser.add_argument('--dataset_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')

    parser.add_argument('--output_dir', default='r101_vistr',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    return parser


def main(args):
    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # no validation ground truth for ytvos dataset
    dataset_train = build_dataset(image_set='train', args=args)
    sampler_train = torch.utils.data.RandomSampler(dataset_train)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True)

    data_loader_train = DataLoader(dataset_train,
                                   batch_sampler=batch_sampler_train,
                                   collate_fn=utils.collate_fn,
                                   num_workers=args.num_workers)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    mapped_classes = data_loader_train.dataset.mapped_classes
    it = 0
    for samples, targets in tqdm(data_loader_train):
        vis_and_save(it, samples, targets, args.output_dir, mapped_classes)
        it += 1

def vis_and_save(it, samples, targets, out_dir, mapped_classes):
    for batch in range(samples.tensors.shape[0]):
        img = denormalize_img(samples.tensors[batch])
        box = denormalize_box(targets[0]['boxes'][batch], img.shape)
        box = [int(b) for b in box]
        lbl = mapped_classes[int(targets[0]['labels'][batch])]['name']
        mask = targets[0]['masks'][batch]

        img_box = cv2.rectangle(img.copy(), box[:2], box[2:], (0,255,0), 2)
        img_box = cv2.putText(img_box, lbl, box[:2], cv2.FONT_HERSHEY_SIMPLEX,
                              1, (255,0,0), 2, cv2.LINE_AA)

        img_mask = create_overlay(img, mask, [1,1,1])
        cv2.imwrite(os.path.join(out_dir, 'box%05d_%05d.png'%(it, batch)), img_box)
        cv2.imwrite(os.path.join(out_dir, 'mask%05d_%05d.png'%(it, batch)), img_mask)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('VisTR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
