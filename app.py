from flask import Flask, render_template,request,redirect,url_for,send_from_directory,flash,jsonify
from flask_socketio import SocketIO, emit
import subprocess
import os
import threading
import sys
import glob
import yaml
from datetime import datetime
from utils.model_train import get_train_command
from utils.model_export import get_export_command
from utils.model_detect import get_detect_result
from data.generate_calibration_data import generate_calibration_data

app = Flask(__name__)
socketio = SocketIO(app)
app.secret_key = 'random string'



def find_file_ending_with(directory_path, endswith):
    for filename in os.listdir(directory_path):
        if filename.endswith(endswith):
            return os.path.join(directory_path, filename)
    return None

def parse_mean_scale(mean_value_text,scale_value_text):
    mean_values = mean_value_text.split()
    scale_values = scale_value_text.split()
    if len(mean_values) != 0:
        if len(mean_values) == 1:
            mean_value_array[0] = float(mean_values[0])
            mean_value_array[1] = float(mean_values[0])
            mean_value_array[2] = float(mean_values[0])
        elif len(mean_values) == 3:
            mean_value_array[0] = float(mean_values[0])
            mean_value_array[1] = float(mean_values[1])
            mean_value_array[2] = float(mean_values[2])
        else:
            socketio.emit('output', "Please enter mean_value one or three numbers\n") 
    if len(scale_values) != 0:
        if len(scale_values) == 1:
            scale_value_array[0] = float(scale_values[0])
            scale_value_array[1] = float(scale_values[0])
            scale_value_array[2] = float(scale_values[0])
        elif len(scale_values) == 3:
            scale_value_array[0] = float(scale_values[0])
            scale_value_array[1] = float(scale_values[1])
            scale_value_array[2] = float(scale_values[2])
        else:
            socketio.emit('output', "Please enter scale_value one or three numbers\n") 

def update_config_file():
    with open(f'{config_file_path}', 'r') as file:
        data = yaml.safe_load(file)
    # 修改值
    if model_type == "onnx":
        data['model_parameters']['onnx_model'] = onnx_file_path
        data['model_parameters']['caffe_model'] = ''
        data['model_parameters']['prototxt'] = ''
    else:
        data['model_parameters']['caffe_model'] = caffemodel_file
        data['model_parameters']['prototxt'] = prototxt_file
        data['model_parameters']['onnx_model'] = ''

    data['input_parameters']['input_type_train'] = input_type_train
    if input_type_train == "nv12":
        data['input_parameters']['input_layout_train'] = ''
    else:
        data['input_parameters']['input_layout_train'] = input_layout_train

    data['input_parameters']['input_type_rt'] = input_type_rt
    if input_type_rt == "nv12":
        data['input_parameters']['input_layout_rt'] = ''
    else:
        data['input_parameters']['input_layout_rt'] = input_layout_rt

    data['input_parameters']['norm_type'] = norm_type
    if norm_type == "data_mean":
        data['input_parameters']['mean_value'] = str(mean_value_array[0])+' '+str(mean_value_array[1])+' '+str(mean_value_array[2])
        data['input_parameters']['scale_value'] = ''
    elif norm_type == "data_scale":
        data['input_parameters']['mean_value'] = ''
        data['input_parameters']['scale_value'] = str(scale_value_array[0])+' '+str(scale_value_array[1])+' '+str(scale_value_array[2])
    elif norm_type == "data_mean_and_scale":
        data['input_parameters']['mean_value'] = str(mean_value_array[0])+' '+str(mean_value_array[1])+' '+str(mean_value_array[2])
        data['input_parameters']['scale_value'] = str(scale_value_array[0])+' '+str(scale_value_array[1])+' '+str(scale_value_array[2])
    else:
        data['input_parameters']['mean_value'] = ''
        data['input_parameters']['scale_value'] = ''
    # 写回YAML文件
    with open(f'{config_file_path}', 'w') as file:
        yaml.dump(data, file)
    return

def create_unique_folder(base_path):
    current_time = datetime.now()
    formatted_time = current_time.strftime("%Y-%m-%d_%H-%M-%S")
    new_folder_path = os.path.join(base_path, formatted_time)
    os.makedirs(new_folder_path)
    return new_folder_path

@app.route('/',methods=["POST","GET"])
def index():
    return render_template('index.html')

@app.route('/upload',methods=["POST","GET"])
def upload():
    global image_folder_path, onnx_file_path, prototxt_file_path,caffemodel_file_path, dataset_file_path, temporary_path
    temporary_path = data_path
    temporary_path = create_unique_folder(temporary_path)
    socketio.emit('clear_output')
    if request.method == 'POST':
        try:
            if has_model == "true":
                if model_type == "onnx":
                    onnx_file = request.files['onnx_file']
                    onnx_file_path = os.path.join(temporary_path, onnx_file.filename)
                    
                    onnx_file.save(onnx_file_path)
                    
                else:
                    prototxt_file = request.files['prototxt_file']
                    prototxt_file_path = os.path.join(temporary_path, prototxt_file.filename)
                    prototxt_file.save(prototxt_file_path)

                    caffemodel_file = request.files['caffemodel_file']
                    caffemodel_file_path = os.path.join(temporary_path, caffemodel_file.filename)
                    caffemodel_file.save(caffemodel_file_path)
                # config_file = request.files['config_file']
                # config_file_path = os.path.join(temporary_path, config_file.filename)
                # config_file.save(config_file_path)

                image_files = request.files.getlist('image_file')
                folder_name = os.path.dirname(image_files[0].filename)
                directory_path = os.path.join(temporary_path, folder_name)
                image_folder_path = directory_path
                # TODO
                directory_path = os.path.join(directory_path, folder_name)
                if not os.path.exists(directory_path):
                    os.makedirs(directory_path)
                for file in image_files:
                    image_file_path = os.path.join(temporary_path, folder_name, file.filename)
                    file.save(image_file_path)
            else:
                dataset_files = request.files.getlist('dataset_file')
                dataset_file_path = os.path.join(temporary_path, (dataset_files[0].filename).split('/')[0])
                for file in dataset_files:
                    if file.filename != '':
                        # 获取文件名，用作文件夹名
                        folder_name = os.path.dirname(file.filename)
                        # 创建文件夹
                        folder_path = os.path.join(temporary_path, folder_name)
                        if not os.path.exists(folder_path):
                            os.makedirs(folder_path, exist_ok=True)
                        # 保存文件到文件夹
                        file.save(os.path.join(folder_path, (os.path.basename(file.filename))))    

            socketio.emit('output', "Upload successful\n") 
        except Exception as r:
            socketio.emit('output', str(r)) 
    return render_template('index.html')

def runCommandAndOutLog(command):
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,text =True)

    def output_reader(pipe, emit_function):
        for line in iter(pipe.readline, b''):
            if len(line) == 0:
                break
            emit_function('output', line)

    stdout_thread = threading.Thread(target=output_reader, args=(process.stdout, socketio.emit))
    stderr_thread = threading.Thread(target=output_reader, args=(process.stderr, socketio.emit))

    stdout_thread.start()
    stderr_thread.start()

    process.wait()

    stdout_thread.join()
    stderr_thread.join()


@app.route('/train', methods=['POST'])
def train():
    socketio.emit('output', "start the training\n") 
    train_command = get_train_command(model_name = model_name, dataset = dataset_file_path, batch_size = batch_size, epochs = epochs, outdir = temporary_path, width = width, height = height)
    if train_command == None:
        socketio.emit('output', "dataset format error\n") 
    #After training, it is necessary to ensure that the pt model is under temporary_path. eg.yolov5
    runCommandAndOutLog(train_command)
    socketio.emit('output', "train completed\n") 
    return '', 200

@app.route('/export', methods=['POST'])
def export():
    pt_model_path = find_file_ending_with(temporary_path, ".pt")
    export_command = get_export_command(model_name = model_name, pt_model_path = pt_model_path, outdir = temporary_path, width = width, height = height, batch_size = 1)
    runCommandAndOutLog(export_command)
    socketio.emit('output', "export completed\n") 
    return '', 200

@app.route('/convert', methods=['POST'])
def convert():
    global config_file_path, onnx_file_path
    try:
        #check
        onnx_file_path = find_file_ending_with(temporary_path, ".onnx")
        runCommandAndOutLog(f"cd {temporary_path} && hb_mapper checker --model-type onnx --model {onnx_file_path} --march bernoulli2")
        socketio.emit('output', "模型检查完成\n") 

        image_path = image_folder_path if has_model == "true" else dataset_file_path
        generate_calibration_data(image_path, temporary_path + '/calibration_data', model_name, width, height, input_type_train)
        socketio.emit('output', "校准数据准备完成\n") 

        # build
        ori_config_file_path = find_file_ending_with(f"/model_zoo/{model_name}", "_config.yaml")
        config_file_path = os.path.join(temporary_path, os.path.basename(ori_config_file_path))
        runCommandAndOutLog(f"cp {ori_config_file_path} {temporary_path}")
        update_config_file()
        runCommandAndOutLog(f"cd {temporary_path} && hb_mapper makertbin --config {config_file_path} --model-type {model_type} ")
        socketio.emit('output', "模型转换完成\n") 

    except Exception as r:
        socketio.emit('output', str(r)) 

    return render_template('index.html')

@app.route('/upload_detect_image',methods=["POST","GET"])
def upload_detect_image():
    global detect_image_file_path
    if request.method == 'POST':
        detect_image_file = request.files['detect_image_file']
        detect_image_file_path = os.path.join(temporary_path, detect_image_file.filename)
        detect_image_file.save(detect_image_file_path)
        socketio.emit('output', "Upload detect image successful\n") 
        return '', 200

@app.route('/detect',methods=["POST","GET"])
def detect():
    detect_model_path = find_file_ending_with(f"{temporary_path}/model_output", f"{detect_model}.onnx")
    layout = "NHWC" if detect_model == "quantized_model" else "NCHW"
    get_detect_result(model_name, detect_image_file_path, detect_model_path, f"{temporary_path}/detect_result.jpg", layout)

    return send_from_directory(temporary_path, "detect_result.jpg")

@app.route('/download_all_model')
def download_all_model():
    os.system(f"cd {temporary_path} && tar zcvf model_output.tar.gz model_output")
    return send_from_directory(temporary_path,"model_output.tar.gz", as_attachment=True)

@app.route('/download_bin_model')
def download_bin_model():
    for filename in os.listdir(f'{temporary_path}/model_output'):
        if filename.endswith(".bin"):
            return send_from_directory(f"{temporary_path}/model_output",filename, as_attachment=True)

@app.route('/update_parameter', methods=['POST'])
def update_parameter():
    global dimension_type, model_name, width, height, model_type, input_type_rt, input_layout_rt, input_type_train, input_layout_train, norm_type, batch_size, epochs, has_model, detect_model
    data = request.get_json()
    model_name = data.get("model_name")
    dimension_type = data.get("dimensionType")
    model_type = data.get("model_type")

    input_type_rt = data.get("input_type_rt")
    input_layout_rt = data.get("input_layout_rt")
    input_type_train = data.get("input_type_train")
    input_layout_train = data.get("input_layout_train")
    norm_type = data.get("norm_type")

    mean_value_text = data.get("mean_value")
    scale_value_text = data.get("scale_value")
    has_model = data.get("has_model")
    parse_mean_scale(mean_value_text, scale_value_text)

    batch_size = data.get("batch_size")
    epochs = data.get("epochs")
    detect_model = data.get("detect_model")

    if dimension_type == "custom":
        width = int(data.get("customWidth"))
        height = int(data.get("customHeight"))
    elif dimension_type == "672x672":
        width = 672
        height = 672
    elif dimension_type == "224x224":
        width = 224
        height = 224
    else:
        width = 224
        height = 224
    # print(data)
    return '', 200

def loda_default_parameter():
    global dimension_type, model_name, width, height, model_type, input_type_rt, input_layout_rt, input_type_train, input_layout_train, norm_type, batch_size, epochs, has_model, detect_model, onnx_file_path, prototxt_file_path, caffemodel_file_path, image_folder_path, config_file_path, mean_value_array, scale_value_array, detect_image_file_path, dataset_file_path, temporary_path, data_path
    with open('./config/config.yaml', 'r') as file:
        config = yaml.safe_load(file)
        onnx_file_path = config["onnx_file_path"]
        prototxt_file_path = config["prototxt_file_path"]
        caffemodel_file_path = config["caffemodel_file_path"]
        image_folder_path = config["image_folder_path"]
        config_file_path = config["config_file_path"]
        model_name = config["model_name"]
        model_type = config["model_type"]
        dimension_type = config["dimension_type"]
        width = config["width"]
        height = config["height"]

        input_type_rt = config["input_type_rt"]
        input_layout_rt = config["input_layout_rt"]
        input_type_train = config["input_type_train"]
        input_layout_train = config["input_layout_train"]
        norm_type = config["norm_type"]
        has_model = config["has_model"]
        mean_value_array = config["mean_value_array"]
        scale_value_array = config["scale_value_array"]

        detect_model = config["detect_model"]
        detect_image_file_path = config["detect_image_file_path"]
        epochs = config["epochs"]
        batch_size = config["batch_size"]

        dataset_file_path = config["dataset_file_path"]
        data_path = config["data_path"]
        temporary_path = config["temporary_path"]

if __name__ == '__main__':
    loda_default_parameter()
    socketio.run(app,host="0.0.0.0",port=5001, debug=True, allow_unsafe_werkzeug=True)

