import os
import yaml

# class ModelExport(object):
#     def __init__(self, model_name, pt_model_path, outdir):
#         self.model_name = model_name
#         self.pt_model_path = pt_model_path
#         self.outdir = outdir
#     def get_export_command(self):
#         pass

class Restnet18ModelExport(object):
    def __init__(self, model_name, pt_model_path, outdir):
        self.model_name = model_name
        self.pt_model_path = pt_model_path
        self.outdir = outdir
    def get_export_command(self):
        export_command = f"cd /model_zoo/{self.model_name} && /usr/bin/python3 export.py --model_path {self.pt_model_path} --outdir {self.outdir}"
        return export_command

class Yolov5sv2ModelExport(object):
    def __init__(self, model_name, pt_model_path, outdir, img_size, batch_size):
        self.model_name = model_name
        self.pt_model_path = pt_model_path
        self.outdir = outdir
        self.img_size = img_size
        self.batch_size = batch_size
    def get_export_command(self):
        export_command = f"cd /model_zoo/{self.model_name} && export PYTHONPATH=$PWD && /usr/bin/python3 models/export.py --weights {self.pt_model_path} --outdir {self.outdir} --img-size {self.img_size} --batch-size {self.batch_size}"
        return export_command

def get_export_command(**kwargs):
    model_name, pt_model_path, outdir = '', '', ''
    batch_size, width, height = 0, 0, 0
    for key, value in kwargs.items():
        if key == "model_name":
            model_name = value
        elif key == "pt_model_path":
            pt_model_path = value
        elif key == "outdir":
            outdir = value
        elif key == "width":
            width = value
        elif key == "height":
            height = value
        elif key == "batch_size":
            batch_size = value
    if model_name == "resnet18_track_detection":
        model_export = Restnet18ModelExport(model_name, pt_model_path, outdir)
    elif model_name == "yolov5s-2.0":
        model_export = Yolov5sv2ModelExport(model_name, pt_model_path, outdir, f'{width} {height}', batch_size )
    return model_export.get_export_command()