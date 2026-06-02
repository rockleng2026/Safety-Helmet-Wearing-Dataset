# -*- coding: utf-8 -*-
"""
Created on Sat Aug 31 22:23:35 2019

@author: czz
"""

import os
from gluoncv import model_zoo, data, utils
#from matplotlib import pyplot as plt
import mxnet as mx
import cv2
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Train YOLO networks with random input shape.')
    parser.add_argument('--network', type=str, default='yolo3_darknet53_voc',
                        help="Base network name which serves as feature extraction base. "
                             "Options: yolo3_darknet53_voc, yolo3_mobilenet1.0_voc, yolo3_mobilenet0.25_voc")
    parser.add_argument('--model-path', type=str, default='models',
                        help='Directory containing model parameter files')
    parser.add_argument('--short', type=int, default=416,
                        help='Input data shape for evaluation, use 320, 416, 512, 608, '
                        'larger size for dense object and big size input')
    parser.add_argument('--threshold', type=float, default=0.4,
                        help='confidence threshold for object detection')
    parser.add_argument('--image', type=str, default=None,
                        help='Path to input image for detection')
    parser.add_argument('--cpu', action='store_true',
                        help='use cpu instead of gpu.')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    if args.cpu:
        ctx = mx.cpu()
    else:
        ctx = mx.gpu()

    net = model_zoo.get_model(args.network, pretrained=False)

    classes = ['hat', 'person']
    for param in net.collect_params().values():
        if param._data is not None:
            continue
        param.initialize()
    net.reset_class(classes)
    net.collect_params().reset_ctx(ctx)

    # Build model path from network name and model directory
    if args.network == 'yolo3_darknet53_voc':
        model_file = 'darknet.params'
    elif args.network == 'yolo3_mobilenet1.0_voc':
        model_file = 'mobilenet1.0.params'
    else:
        model_file = 'mobile0.25.params'
    model_path = os.path.join(args.model_path, model_file)
    net.load_parameters(model_path, ctx=ctx)
    print(f'use {args.network} to extract feature, model: {model_path}')

    # Use provided image or default to '1.jpg'
    image_path = args.image if args.image else '1.jpg'
    x, orig_img = data.transforms.presets.yolo.load_test(image_path, short=args.short)
    x = x.as_in_context(ctx)
    box_ids, scores, bboxes = net(x)
    utils.viz.cv_plot_bbox(orig_img, bboxes[0], scores[0], box_ids[0],
                            class_names=net.classes, thresh=args.threshold)
    cv2.imshow('image', orig_img[..., ::-1])
    cv2.waitKey(0)
    cv2.imwrite(image_path.rsplit('.', 1)[0] + '_result.jpg', orig_img[..., ::-1])
    cv2.destroyAllWindows()
