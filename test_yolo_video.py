# -*- coding: utf-8 -*-
"""
YOLO Video Detection - 安全帽佩戴检测视频处理程序
基于 GluonCV YOLO 模型，参照 test_yolo.py 的标注方式
"""

import os
import cv2
import argparse
from gluoncv import model_zoo, data, utils
import mxnet as mx


def parse_args():
    parser = argparse.ArgumentParser(description='YOLO video detection for safety helmet wearing.')
    parser.add_argument('--network', type=str, default='yolo3_mobilenet1.0_voc',
                        help="Base network name. "
                             "Options: yolo3_darknet53_voc, yolo3_mobilenet1.0_voc, yolo3_mobilenet0.25_voc")
    parser.add_argument('--model-path', type=str, default='models',
                        help='Directory containing model parameter files')
    parser.add_argument('--video', type=str, required=True,
                        help='Path to input video file')
    parser.add_argument('--output', type=str, required=True,
                        help='Path to output video file')
    parser.add_argument('--short', type=int, default=416,
                        help='Input data shape for evaluation, use 320, 416, 512, 608')
    parser.add_argument('--threshold', type=float, default=0.4,
                        help='confidence threshold for object detection')
    parser.add_argument('--cpu', action='store_true',
                        help='use cpu instead of gpu.')

    args = parser.parse_args()
    return args


def transform_frame(frame_nd, short, max_size=1024, stride=1,
                    mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    """Transform a frame NDArray for YOLO input."""
    import gluoncv.data.transforms.presets.yolo as yolo_transform
    return yolo_transform.transform_test([frame_nd], short, max_size, stride, mean, std)


def process_video(net, ctx, video_path, output_path, short, threshold):
    """Process a video file and save annotated video."""
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f'Error: Cannot open video file {video_path}')
        return

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Use MJPG codec
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    if not out.isOpened():
        print(f'Error: Cannot open video writer {output_path}')
        cap.release()
        return

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % 10 == 0:
            print(f'Processing frame {frame_count}...', end='\r')

        # Convert BGR (cv2) to RGB, then to MXNet NDArray
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_nd = mx.nd.array(frame_rgb)

        # Transform: resize and normalize
        x, orig_img = transform_frame(frame_nd, short=short)
        x = x.as_in_context(ctx)
        box_ids, scores, bboxes = net(x)

        # Draw bounding boxes using GluonCV's cv_plot_bbox (same as test_yolo.py)
        utils.viz.cv_plot_bbox(orig_img, bboxes[0], scores[0], box_ids[0],
                                class_names=net.classes, thresh=threshold)

        # Convert annotated image back to BGR for video writer
        annotated_bgr = cv2.cvtColor(orig_img, cv2.COLOR_RGB2BGR)

        # Resize to original video resolution for output
        annotated_resized = cv2.resize(annotated_bgr, (width, height))

        # Write to output video
        out.write(annotated_resized)

    cap.release()
    out.release()
    print(f'\nVideo processed: {frame_count} frames saved to {output_path}')


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

    process_video(net, ctx, args.video, args.output, args.short, args.threshold)