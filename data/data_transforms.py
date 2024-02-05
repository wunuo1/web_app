# Copyright (c) 2022，Horizon Robotics.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import numpy as np

import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from PIL import Image


class FormatTransform(object):
    def __init__(self, format):
        self.format = format
    def __call__(self, img):
        # 将 RGB 图像转换为 BGR
        if(self.format == "bgr"):
            img = np.array(img)[:, :, ::-1]
            img = Image.fromarray(np.uint8(img))
        return img

class RGB2NV12Transform(object):
    def mergeUV(self, u, v):
        if u.shape == v.shape:
            uv = np.zeros(shape=(u.shape[0], u.shape[1] * 2))
            for i in range(0, u.shape[0]):
                for j in range(0, u.shape[1]):
                    uv[i, 2 * j] = u[i, j]
                    uv[i, 2 * j + 1] = v[i, j]
            return uv
        else:
            raise ValueError("size of Channel U is different with Channel V")
    def __call__(self, img):
        if img.mode == "RGB":
            img = np.array(img)
            r = img[:, :, 0]
            g = img[:, :, 1]
            b = img[:, :, 2]
            y = (0.299 * r + 0.587 * g + 0.114 * b)
            u = (-0.169 * r - 0.331 * g + 0.5 * b + 128)[::2, ::2]
            v = (0.5 * r - 0.419 * g - 0.081 * b + 128)[::2, ::2]
            uv = self.mergeUV(u, v)
            yuv = np.vstack((y, uv))
            return Image.fromarray(yuv.astype(np.uint8))
        else:
            raise ValueError("image is not RGB format")

class NV12ToYUV444Transformer(object):
    def __init__(self, target_size, yuv444_output_layout="HWC"):
        super(NV12ToYUV444Transformer, self).__init__()
        self.height = target_size[0]
        self.width = target_size[1]
        self.yuv444_output_layout = yuv444_output_layout

    def __call__(self, data):
        data = np.array(data)
        nv12_data = data.flatten()
        yuv444 = np.empty([self.height, self.width, 3], dtype=np.uint8)
        yuv444[:, :, 0] = nv12_data[:self.width * self.height].reshape(
            self.height, self.width)
        u = nv12_data[self.width * self.height::2].reshape(
            self.height // 2, self.width // 2)
        yuv444[:, :, 1] = Image.fromarray(u).resize((self.width, self.height),
                                                    resample=0)
        v = nv12_data[self.width * self.height + 1::2].reshape(
            self.height // 2, self.width // 2)
        yuv444[:, :, 2] = Image.fromarray(v).resize((self.width, self.height),
                                                    resample=0)
        data = yuv444.astype(np.uint8)
        if self.yuv444_output_layout == "CHW":
            data = np.transpose(data, (2, 0, 1))
        return Image.fromarray(data)

class PadResizeTransformer:
    def __init__(self, height, width):
        self.height = height
        self.width = width

    def __call__(self, img):
        # 计算比例
        original_width, original_height = img.size
        aspect_ratio = original_width / original_height
        if aspect_ratio > 1:
            new_width = self.width
            new_height = int(self.width / aspect_ratio)
        else:
            new_height = self.height
            new_width = int(self.height * aspect_ratio)
        transform = transforms.Compose([
            transforms.Resize((new_height, new_width)),
        ])
        resized_img = transform(img)
        # resize_transform = transforms.Resize((new_height, new_width))
        # pad_transform = transforms.Pad(padding=(int((self.width - new_width)/2), int((self.height - new_height)/2), int((self.width - new_width)/2), int((self.height - new_height)/2)), fill=127)
        # transform = transforms.Compose([
        #     resize_transform,
        #     pad_transform
        # ])
        # resized_padded_img = transform(img)
        pad_image = np.full(shape=[self.height, self.width, 3],
                    fill_value=127).astype(np.uint8)
        dw, dh = (self.width - new_width) // 2, (self.height - new_height) // 2
        pad_image[dh:new_height + dh, dw:new_width + dw, :] = resized_img
        pad_image = pad_image.astype(np.uint8)
        return Image.fromarray(pad_image)

class DataTransforms(object):
    def __init__(self, width, height):
       self.width = width
       self.height = height

    def calibration_transforms(self):
       pass

class Resnet18DataTransforms(DataTransforms):
    def __init__(self, width, height, mean, stddev, format):
       super(Resnet18DataTransforms, self).__init__(width, height)
       self.mean = mean
       self.stddev = stddev
       self.format = format
    
    def calibration_transforms(self):
        transform = transforms.Compose(
            [FormatTransform(self.format),
            transforms.Resize((self.height,self.width)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x * 255)])
        return transform


class Yolov5sv2DataTransforms(DataTransforms):
    def __init__(self, width, height, format):
       super(Yolov5sv2DataTransforms, self).__init__(width, height)
       self.format = format
    
    def calibration_transforms(self):
        transform = transforms.Compose(
            [FormatTransform(self.format),
            PadResizeTransformer(self.height, self.width),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x * 255)])
        return transform