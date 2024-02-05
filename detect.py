# Copyright (c) 2021 Horizon Robotics.All Rights Reserved.
#
# The material in this file is confidential and contains trade secrets
# of Horizon Robotics Inc. This is proprietary information owned by
# Horizon Robotics Inc. No part of this work may be disclosed,
# reproduced, copied, transmitted, or used in any way for any purpose,
# without the express written permission of Horizon Robotics Inc.

import os
import click
import logging

from horizon_tc_ui import HB_ONNXRuntime
from horizon_tc_ui.utils.tool_utils import init_root_logger, on_exception_exit
from horizon_tc_ui.data.imagenet_val import imagenet_val
from PIL import Image
from PIL import ImageDraw, ImageFont

import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import colorsys

import sys

import os
import yaml

import cv2

from easydict import EasyDict
sys.path.append("/web_app")
from data.data_transforms import *

class ModelDetect(object):
    def __init__(self, image_source_path ,onnx_model_path, image_save_path, layout):
        self.image_source_path = image_source_path
        self.onnx_model_path = onnx_model_path
        self.image_save_path = image_save_path
        self.layout = layout

        self.sess = HB_ONNXRuntime(model_file=self.onnx_model_path)
        self.image = Image.open(self.image_source_path)

        self.image_transform = None
        self.output = None

        self.model_width, self.model_height = self.get_model_shape()
        self.image_width, self.image_height = self.image.size

    def get_model_shape(self):
        input_info = self.sess.model.graph.input[0]
        input_name = input_info.name
        input_shape = input_info.type.tensor_type.shape.dim
        if self.layout == "NHWC":
            height = input_shape[1].dim_value
            width = input_shape[2].dim_value
        else:
            height = input_shape[2].dim_value
            width = input_shape[3].dim_value
        print("model width:" + str(width))
        print("model height:" + str(height))
        return width,height

    def detect(self):
        self.image_transform = self.get_transforms()
        transformed_image = self.image_transform(self.image)
        image_numpy = transformed_image.numpy()
        if self.layout == "NHWC":
            image_numpy = image_numpy.transpose((1, 2, 0))
        image_numpy = np.expand_dims(image_numpy, axis=0)

        self.sess.set_dim_param(0, 0, '?')
        input_name = self.sess.input_names[0]
        output_name = self.sess.output_names
        self.output = self.sess.run(output_name, {input_name: image_numpy},input_offset=128)
        self.postprocess()

    def get_transforms():
        pass

    def postprocess():
        pass


class Restnet18ModelDetect(ModelDetect):
    def __init__(self, image_source_path, onnx_model_path, image_save_path, layout):
        super().__init__(image_source_path, onnx_model_path, image_save_path, layout)

    def get_transforms(self):
        image_transform = transforms.Compose([
            transforms.Resize((self.model_height, self.model_width)),
            RGB2NV12Transform(),
            NV12ToYUV444Transformer((self.model_height,self.model_width),'HWC'),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x * 255) 
        ])
        return image_transform
        
    def postprocess(self):
        imagedraw = ImageDraw.Draw(self.image)
        prob = np.squeeze(self.output[0])
        x = int((prob[0] * (self.model_width/2) + (self.model_width/2)) * self.image_width / self.model_width)
        y = int(((self.model_height/2) - (self.model_height/2) * prob[1]) * self.image_height / self.model_height)
        x = 5 if x < 5 else x
        x = self.image_width - 6 if x > self.image_width - 6 else x
        y = 5 if y < 5 else y
        y = self.image_height - 6 if y > self.image_width - 6 else y
        for i in range(x - 5, x + 5):
            for j in range(y - 5, y + 5):
                imagedraw.point((i, j), (255,0,0))
        self.image.save(self.image_save_path)

class Yolov5sv2ModelDetect(ModelDetect):
    def __init__(self, image_source_path, onnx_model_path, image_save_path, layout):
        super().__init__(image_source_path, onnx_model_path, image_save_path, layout)
    def get_transforms(self):
        image_transform = transforms.Compose([
            PadResizeTransformer(self.model_height, self.model_width),
            RGB2NV12Transform(),
            NV12ToYUV444Transformer((self.model_height, self.model_width),'HWC'),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x * 255) 
        ])
        return image_transform

    def yolov5s_decoder(self, conv_output, num_anchors, num_classes, anchors, stride):
        def sigmoid(x):
            return 1. / (1 + np.exp(-x))

        # Five dimension output: [batch_size, num_anchors, output_size, output_size, 5 + num_classes]
        batch_size = conv_output.shape[0]
        output_size = conv_output.shape[-2]
        conv_raw_dxdy = conv_output[:, :, :, :, 0:2]
        conv_raw_dwdh = conv_output[:, :, :, :, 2:4]
        conv_raw_conf = conv_output[:, :, :, :, 4:5]
        conv_raw_prob = conv_output[:, :, :, :, 5:]

        y = np.tile(
            np.arange(output_size, dtype=np.int32)[:, np.newaxis],
            [1, output_size])
        x = np.tile(
            np.arange(output_size, dtype=np.int32)[np.newaxis, :],
            [output_size, 1])
        xy_grid = np.concatenate([x[:, :, np.newaxis], y[:, :, np.newaxis]],
                                axis=-1)
        xy_grid = np.tile(xy_grid[np.newaxis, np.newaxis, :, :, :],
                        [batch_size, num_anchors, 1, 1, 1])
        xy_grid = xy_grid.astype(np.float32)

        pred_xy = (sigmoid(conv_raw_dxdy) * 2.0 - 0.5 + xy_grid) * stride
        pred_wh = (sigmoid(conv_raw_dwdh) *
                2.0)**2 * anchors[np.newaxis, :, np.newaxis, np.newaxis, :]
        pred_xywh = np.concatenate([pred_xy, pred_wh], axis=-1)

        pred_conf = sigmoid(conv_raw_conf)
        pred_prob = sigmoid(conv_raw_prob)

        decode_output = np.concatenate([pred_xywh, pred_conf, pred_prob], axis=-1)
        return decode_output

    def postprocess_boxes(self, pred_bbox,
                        org_img_shape,
                        input_shape,
                        score_threshold=0.5):
        """post process boxes"""
        valid_scale = [0, np.inf]
        org_h, org_w = org_img_shape
        input_h, input_w = input_shape
        pred_bbox = np.array(pred_bbox)

        pred_xywh = pred_bbox[:, :4]
        pred_conf = pred_bbox[:, 4]
        pred_prob = pred_bbox[:, 5:]

        # (x, y, w, h) --> (xmin, ymin, xmax, ymax)
        pred_coor = np.concatenate([
            pred_xywh[:, :2] - pred_xywh[:, 2:] * 0.5,
            pred_xywh[:, :2] + pred_xywh[:, 2:] * 0.5
        ],
                                axis=-1)

        # (xmin, ymin, xmax, ymax) -> (xmin_org, ymin_org, xmax_org, ymax_org)
        resize_ratio = min(input_h / org_h, input_w / org_w)
        dw = (input_w - resize_ratio * org_w) / 2
        dh = (input_h - resize_ratio * org_h) / 2
        pred_coor[:, 0::2] = 1.0 * (pred_coor[:, 0::2] - dw) / resize_ratio
        pred_coor[:, 1::2] = 1.0 * (pred_coor[:, 1::2] - dh) / resize_ratio

        # clip the range of bbox
        pred_coor = np.concatenate([
            np.maximum(pred_coor[:, :2], [0, 0]),
            np.minimum(pred_coor[:, 2:], [org_w - 1, org_h - 1])
        ],
                                axis=-1)
        # drop illegal boxes whose max < min
        invalid_mask = np.logical_or((pred_coor[:, 0] > pred_coor[:, 2]),
                                    (pred_coor[:, 1] > pred_coor[:, 3]))
        pred_coor[invalid_mask] = 0

        # discard invalid boxes
        bboxes_scale = np.sqrt(
            np.multiply.reduce(pred_coor[:, 2:4] - pred_coor[:, 0:2], axis=-1))
        scale_mask = np.logical_and((valid_scale[0] < bboxes_scale),
                                    (bboxes_scale < valid_scale[1]))

        # discard boxes with low scores
        classes = np.argmax(pred_prob, axis=-1)
        scores = pred_conf * pred_prob[np.arange(len(pred_coor)), classes]
        score_mask = scores > score_threshold
        mask = np.logical_and(scale_mask, score_mask)
        coors, scores, classes = pred_coor[mask], scores[mask], classes[mask]

        return np.concatenate(
            [coors, scores[:, np.newaxis], classes[:, np.newaxis]], axis=-1)

    def bboxes_iou(self, boxes1, boxes2):
        """calculate iou for a list of bboxes"""
        boxes1 = np.array(boxes1)
        boxes2 = np.array(boxes2)
        boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * \
                    (boxes1[..., 3] - boxes1[..., 1])
        boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * \
                    (boxes2[..., 3] - boxes2[..., 1])
        left_up = np.maximum(boxes1[..., :2], boxes2[..., :2])
        right_down = np.minimum(boxes1[..., 2:], boxes2[..., 2:])
        inter_section = np.maximum(right_down - left_up, 0.0)
        inter_area = inter_section[..., 0] * inter_section[..., 1]
        union_area = boxes1_area + boxes2_area - inter_area
        ious = np.maximum(1.0 * inter_area / union_area, np.finfo(np.float32).eps)

        return ious

    def nms(self, bboxes, iou_threshold, sigma=0.3, method='nms'):
        """
        calculate the nms for bboxes
        """
        classes_in_img = list(set(bboxes[:, 5]))
        best_bboxes = []

        for cls in classes_in_img:
            cls_mask = (bboxes[:, 5] == cls)
            cls_bboxes = bboxes[cls_mask]

            while len(cls_bboxes) > 0:
                max_ind = np.argmax(cls_bboxes[:, 4])
                best_bbox = cls_bboxes[max_ind]
                best_bboxes.append(best_bbox)
                cls_bboxes = np.concatenate(
                    [cls_bboxes[:max_ind], cls_bboxes[max_ind + 1:]])
                iou = self.bboxes_iou(best_bbox[np.newaxis, :4], cls_bboxes[:, :4])
                weight = np.ones((len(iou), ), dtype=np.float32)

                assert method in ['nms', 'soft-nms']

                if method == 'nms':
                    iou_mask = iou > iou_threshold
                    weight[iou_mask] = 0.0
                if method == 'soft-nms':
                    weight = np.exp(-(1.0 * iou**2 / sigma))

                cls_bboxes[:, 4] = cls_bboxes[:, 4] * weight
                score_mask = cls_bboxes[:, 4] > 0.
                cls_bboxes = cls_bboxes[score_mask]

        return best_bboxes

    def draw_bboxs(self, image, bboxes, num_classes, gt_classes_index=None):
        """draw the bboxes in the original image
        """
        imagedraw = ImageDraw.Draw(self.image)
        hsv_tuples = [(1.0 * x / num_classes, 1., 1.) for x in range(num_classes)]
        colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                colors))
        fontScale = 0.5
        bbox_thick = int(0.6 * (self.image_height + self.image_width) / 600)

        for i, bbox in enumerate(bboxes):
            coor = np.array(bbox[:4], dtype=np.int32)

            if gt_classes_index == None:
                class_index = int(bbox[5])
                score = bbox[4]
            else:
                class_index = gt_classes_index[i]
                score = 1

            bbox_color = colors[class_index]
            # scale = self.image_width / self.model_width if self.image_width > self.image_height else self.image_height / self.model_height
            scale = 1
            imagedraw.rectangle([coor[0], coor[1], coor[2], coor[3]], outline=bbox_color, width=2)

            classes_name = class_index
            bbox_mess = '%s: %.2f' % (classes_name, score)

            font = ImageFont.truetype('LiberationSans-Italic.ttf', size=30)
            text = bbox_mess

            text_width, text_height = imagedraw.textsize(text, font=font)
            text_x = coor[0]
            text_y = coor[1]

            imagedraw.text((text_x, text_y), text, fill='black', font=font)
            image.save(self.image_save_path)

    def get_yolov5s_config(self):
        yolov5s_config = EasyDict()
        yolov5s_config.ANCHORS = np.array([
            10, 13, 16, 30, 33, 23, 30, 61, 62, 45, 59, 119, 116, 90, 156, 198,
            373, 326
        ]).reshape((3, 3, 2))
        yolov5s_config.STRIDES = np.array([8, 16, 32])
        yolov5s_config.NUM_CLASSES = int(self.output[0].shape[3] / 3) - 5
        yolov5s_config.INPUT_SHAPE = (self.model_height, self.model_width)
        #TODO
        yolov5s_config.SCORE_THRESHOLD = 0.5
        yolov5s_config.NMS_THRESHOLD = 0.45
        return yolov5s_config

    def postprocess(self):

        yolov5s_config = self.get_yolov5s_config()
        num_classes = yolov5s_config.NUM_CLASSES
        anchors = yolov5s_config.ANCHORS
        num_anchors = anchors.shape[0]
        strides = yolov5s_config.STRIDES
        input_shape = yolov5s_config.INPUT_SHAPE
        score_threshold = yolov5s_config.SCORE_THRESHOLD
        nms_threshold = yolov5s_config.NMS_THRESHOLD
        
        self.output[0] = self.output[0].reshape([1, int(self.model_height / strides[0]), int(self.model_width / strides[0]), 3,
                                                num_classes+5]).transpose([0, 3, 1, 2, 4])
        self.output[1] = self.output[1].reshape([1, int(self.model_height / strides[1]), int(self.model_width / strides[1]), 3,
                                                num_classes+5]).transpose([0, 3, 1, 2, 4])
        self.output[2] = self.output[2].reshape([1, int(self.model_height / strides[2]), int(self.model_width / strides[2]), 3,
                                                num_classes+5]).transpose([0, 3, 1, 2, 4])

        pred_sbbox, pred_mbbox, pred_lbbox = self.output[0], self.output[
            1], self.output[2]

        pred_sbbox = self.yolov5s_decoder(pred_sbbox, num_anchors, num_classes,
                                    anchors[0], strides[0])
        pred_mbbox = self.yolov5s_decoder(pred_mbbox, num_anchors, num_classes,
                                    anchors[1], strides[1])
        pred_lbbox = self.yolov5s_decoder(pred_lbbox, num_anchors, num_classes,
                                    anchors[2], strides[2])
        pred_bbox = np.concatenate([
            np.reshape(pred_sbbox, (-1, 5 + num_classes)),
            np.reshape(pred_mbbox, (-1, 5 + num_classes)),
            np.reshape(pred_lbbox, (-1, 5 + num_classes))
        ], axis=0)

        bboxes = self.postprocess_boxes(pred_bbox, (self.image_height, self.image_width), input_shape=(self.model_height, self.model_width), score_threshold=score_threshold)
        nms_bboxes = self.nms(bboxes, nms_threshold)
        print(f"detected item num: {len(nms_bboxes)}")
        if len(nms_bboxes) != 0:
            if self.image is not None:
                self.draw_bboxs(self.image, nms_bboxes, num_classes)
        else:
            self.image.save(self.image_save_path)