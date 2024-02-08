import os
import yaml
import shutil
# class ModelTrain(object):
#     def __init__(self, model_name, dataset, batch_size, epochs, outdir):
#         self.model_name = model_name
#         self.dataset = dataset
#         self.batch_size = batch_size
#         self.epochs = epochs
#         self.outdir = outdir
#     def get_train_command(self):
#        pass

class Restnet18ModelTrain(object):
    def __init__(self, model_name, dataset, batch_size, epochs, outdir):
        self.model_name = model_name
        self.dataset = dataset
        self.batch_size = batch_size
        self.epochs = epochs
        self.outdir = outdir
    def get_train_command(self):
        train_command = f"cd /model_zoo/{self.model_name} && /usr/bin/python3 train.py --dataset {self.dataset} --batch-size {self.batch_size} --epochs {self.epochs} --outdir {self.outdir}"
        return train_command

class Yolov5sv2ModelTrain(object):
    def __init__(self, model_name, dataset, batch_size, epochs, outdir, img_size):
        self.model_name = model_name
        self.dataset = dataset
        self.batch_size = batch_size
        self.epochs = epochs
        self.outdir = outdir
        self.img_size = img_size
    def get_train_command(self):
        if os.path.exists(f"{self.outdir}/exp0"):
            shutil.rmtree(f"{self.outdir}/exp0")
        yaml_file_path = None
        train_command = None
        for filename in os.listdir(self.dataset):
            if filename.endswith('.yaml'):
                yaml_file_path = os.path.join(self.dataset, filename)
        if yaml_file_path != None:
            with open(f'{yaml_file_path}', 'r') as file:
                data = yaml.safe_load(file)
                data['train'] = os.path.join(self.dataset, 'train/images')
                data['val'] = os.path.join(self.dataset, 'valid/images')
                data['test'] = os.path.join(self.dataset, 'test/images')
            with open(f'{yaml_file_path}', 'w') as file:
                yaml.dump(data, file)

            train_command = f"cd /model_zoo/{self.model_name} && /usr/bin/python3 train.py --data {yaml_file_path} --weights 'yolov5s.pt' --batch-size {self.batch_size} --img-size {self.img_size} --epochs {self.epochs} --outdir {self.outdir} && cp {self.outdir}/exp0/weights/best.pt {self.outdir}"
        return train_command

def get_train_command(**kwargs):
    model_name, dataset, outdir = '', '', ''
    batch_size, epochs, width, height = 0, 0, 0, 0
    for key, value in kwargs.items():
        if key == "model_name":
            model_name = value
        elif key == "dataset":
            dataset = value
        elif key == "batch_size":
            batch_size = value
        elif key == "epochs":
            epochs = value
        elif key == "outdir":
            outdir = value
        elif key == "width":
            width = value
        elif key == "height":
            height = value
    if model_name == "resnet18_track_detection":
        model_train = Restnet18ModelTrain(model_name, dataset, batch_size, epochs, outdir)
    elif model_name == "yolov5s-2.0":
        model_train = Yolov5sv2ModelTrain(model_name, dataset, batch_size, epochs, outdir, f'{width} {height}')
    return model_train.get_train_command()