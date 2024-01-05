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
from data_transforms import DataTransforms, Resnet18DataTransforms

import argparse
import os
import numpy as np

import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets

def calibration_transforms(args.model):
    if args.model == "resnet18":
        transforms = Resnet18DataTransforms(args.width,args.height,args.mean,args.stddev)
    return transforms.calibration_transforms()


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Train classifier.')
    parser.add_argument(
        '--dataset', '-d', type=str, required=True, help='Root directory of dataset'
    )
    parser.add_argument(
        '--outdir', '-o', type=str, default='/open_explorer/web_app/temporary/calibration_data/', help='Output directory'
    )
    parser.add_argument(
        '--model', '-m', type=str, required=True, help='Model type'
    )
    parser.add_argument(
        '--width', '-w', type=int, default=224, help='Image width'
    )
    parser.add_argument(
        '--height', '-h', type=int, default=224, help='Image height'
    )
    parser.add_argument('--mean', '-n' , nargs=3, type=float, default=(0.485, 0.456, 0.406), help='Mean')
    parser.add_argument('--stddev', '-s' , nargs=3, type=float, default=(0.229, 0.224, 0.225) ,help='Stddev')
    args = parser.parse_args()
    return args

def main(args):

    transforms = calibration_transforms(args)

    dataset = datasets.ImageFolder(args.dataset, transform=transforms)

    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)
    # 保存校准数据
    for i in range(len(dataset)):
        img, label = dataset[i]
        img.squeeze().numpy().tofile(args.outdir + '/' + str(i) + "_.rgb")
      

if __name__ == '__main__':
  args = get_args()
  main(args)