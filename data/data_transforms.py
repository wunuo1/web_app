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
            transforms.Resize((self.width,self.height)),
            transforms.ToTensor(),
            transforms.Normalize(self.mean,self.stddev)])
        return transform