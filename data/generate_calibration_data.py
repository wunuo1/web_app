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
from .data_transforms import *

import argparse
import os
import numpy as np

import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets


def generate_calibration_data(dataset_file_path, outdir, model, width, height, image_format):
    
    if model == "resnet18_track_detection":
        cal_transforms = Resnet18DataTransforms(width, height, image_format)
    elif model == "yolov5s-2.0":
        cal_transforms = Yolov5sv2DataTransforms(width, height, image_format)
    dataset = datasets.ImageFolder(dataset_file_path, transform=cal_transforms.calibration_transforms())
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    # 保存校准数据
    for i in range(len(dataset)):
        if i >= 100:
            break
        img, label = dataset[i]
        img.squeeze().numpy().tofile(outdir + '/' + str(i) + "_." + image_format)
        # print(outdir + '/'+ str(i) + "_." + image_format, flush=True)
      
