from flask import Flask, render_template,request,redirect,url_for,send_from_directory,flash,jsonify
from flask_socketio import SocketIO, emit
import subprocess
import os
import threading
import sys
import glob
import yaml

app = Flask(__name__)
socketio = SocketIO(app)
app.secret_key = 'random string'

onnx_file_path = ""
prototxt_file_path = ""
caffemodel_file_path = ""
config_file_path = ""
image_folder_path = ""

#model_name用于匹配工具链对应的文件夹以及模型训练代码仓库
model_name = "resnet18"
model_type = "onnx"
dimension_type = "224x224"
width = 224
height = 224

input_type_rt = "nv12"
input_layout_rt = "NHWC"
input_type_train = "rgb"
input_layout_train = "NCHW"
norm_type = "data_mean_and_scale"
has_model = "true"
# has_dataset = "false"
mean_value_array = [123.675, 116.28, 103.53]
scale_value_array = [0.0171248, 0.017507, 0.0174292]

epochs = 4
batch_size = 8

dataset_file_path = ""
temporary_path = "/open_explorer/web_app/temporary"
search_path='/open_expolorer/horizon_xj3_open_explorer_v2.6.2b-py38_20230606/ddk/samples/ai_toolchain/horizon_model_convert_sample'
preprocess_py = '/open_expolorer/horizon_xj3_open_explorer_v2.6.2b-py38_20230606/ddk/samples/ai_toolchain/horizon_model_convert_sample/data_preprocess.py'

def find_file_in_folder(model_folder_name):
    path = glob.glob(search_path + '/*/*' + model_name + '*/mapper/02_preprocess.sh')
    if len(path) != 0:
        print(path)
        return path[0]
    else:
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
    global input_layout_train, input_type_train, input_type_rt, input_layout_rt, norm_type
    with open(f'{temporary_path}/{model_name}_config.yaml', 'r') as file:
        data = yaml.safe_load(file)
    # 修改值
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
    with open(f'/open_explorer/web_app/temporary/{model_name}_config.yaml', 'w') as file:
        yaml.dump(data, file)
    return

@app.route('/',methods=["POST","GET"])
def index():
    return render_template('index.html')

@app.route('/upload',methods=["POST","GET"])
def upload():
    upload_message = ""
    global model_type,config_file_path,image_folder_path,onnx_file_path,prototxt_file_path,caffemodel_file_path,dataset_file_path, has_model
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
                directory_path = os.path.join(directory_path, folder_name)
                if not os.path.exists(directory_path):
                    os.makedirs(directory_path)
                for file in image_files:
                    image_file_path = os.path.join(temporary_path, folder_name, file.filename)
                    file.save(image_file_path)
            else:
                dataset_files = request.files.getlist('dataset_file')
                dataset_file_path = os.path.join(temporary_path, (dataset_files[0].filename).split('/')[0])
                # print(dataset_files[0])
                for file in dataset_files:
                    if file.filename != '':
                    #     # 获取文件名，用作文件夹名
                        folder_name = os.path.dirname(file.filename)
                        # 创建文件夹
                        folder_path = os.path.join(temporary_path, folder_name)
                        if not os.path.exists(folder_path):
                            os.makedirs(folder_path, exist_ok=True)
                        # 保存文件到文件夹
                        file.save(os.path.join(folder_path, (os.path.basename(file.filename))))    


            upload_message = "Upload successful\n"
        except Exception as r:
            # upload_message = "Upload false\n"
            upload_message = str(r)
        socketio.emit('output', upload_message) 
    return render_template('index.html')

def runCommandAndOutLog(command):
    # socketio.emit('clear_output')
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,text =True)

    def output_reader(pipe, emit_function):
        for line in iter(pipe.readline, b''):
            if len(line) == 0:
                break
            emit_function('output', line)

    # 启动用于读取 stdout 和 stderr 的两个线程
    stdout_thread = threading.Thread(target=output_reader, args=(process.stdout, socketio.emit))
    stderr_thread = threading.Thread(target=output_reader, args=(process.stderr, socketio.emit))

    stdout_thread.start()
    stderr_thread.start()

    # 等待进程完成
    process.wait()

    # 等待线程完成
    stdout_thread.join()
    stderr_thread.join()


@app.route('/train', methods=['POST'])
def train():
    socketio.emit('output', "start the training\n") 
    global model_name, dataset_file_path, batch_size, epochs
    runCommandAndOutLog(f"cd /model_zoo/{model_name} && /usr/bin/python3 train.py --dataset {dataset_file_path} --batch-size {batch_size} --epochs {epochs}")
    socketio.emit('output', "train completed\n") 
    return '', 200

@app.route('/export', methods=['POST'])
def export():
    global model_name
    runCommandAndOutLog(f"cd /model_zoo/{model_name} && /usr/bin/python3 export.py")
    socketio.emit('output', "export completed\n") 
    return '', 200

@app.route('/convert', methods=['POST'])
def convert():
    global onnx_file_path, prototxt_file_path, caffemodel_file_path, image_folder_path, model_name,  width, height, temporary_path,  model_type
    # find_file_in_folder(model_folder_name = model_name)
    if (onnx_file_path != "" or (prototxt_file_path != "" and caffemodel_file_path != "")) and image_folder_path != "":
        #有模型
        #check
        runCommandAndOutLog(f"hb_mapper checker --model-type onnx --model {onnx_file_path} --march bernoulli2")
        socketio.emit('output', "模型检查完成\n") 

        #preprocess
        model_path = os.path.dirname(find_file_in_folder(model_folder_name = model_name))
        # runCommandAndOutLog(f"cd {model_path} && /usr/bin/python3 {preprocess_py} --dst_dir {temporary_path}/calibration_data --pic_ext .rgb --read_mode opencv --saved_data_type float32 --src_dir {image_folder_path} ")
        runCommandAndOutLog(f"/usr/bin/python3 /open_explorer/web_app/data/generate_calibration_data.py  --dataset {image_folder_path} --model {model_name} --width {width} --height {height} --format rgb ")
        socketio.emit('output', "校准数据准备完成\n") 

        # build
        runCommandAndOutLog(f"cp /model_zoo/{model_name}/{model_name}_config.yaml {temporary_path}")
        update_config_file()

        runCommandAndOutLog(f"cd {temporary_path} && hb_mapper makertbin --config {temporary_path}/{model_name}_config.yaml --model-type {model_type} ")
        socketio.emit('output', "模型转换完成\n") 
    else:
        #没有模型，训练的
        #check
        runCommandAndOutLog(f"hb_mapper checker --model-type onnx --model {temporary_path}/best.onnx --march bernoulli2")
        socketio.emit('output', "模型检查完成\n") 

        #preprocess
        runCommandAndOutLog(f"/usr/bin/python3 /open_explorer/web_app/data/generate_calibration_data.py  --dataset {dataset_file_path} --model {model_name} --width {width} --height {height} --format rgb ")
        socketio.emit('output', "校准数据准备完成\n") 

        # build
        runCommandAndOutLog(f"cp /model_zoo/{model_name}/{model_name}_config.yaml {temporary_path}")
        update_config_file()

        runCommandAndOutLog(f"cd {temporary_path} && hb_mapper makertbin --config {temporary_path}/{model_name}_config.yaml --model-type {model_type} ")
        socketio.emit('output', "模型转换完成\n") 

    return render_template('index.html')

@app.route('/download_all_model')
def download_all_model():
    socketio.emit('output', "Packaging all models\n") 
    os.system(f"cd {temporary_path} && tar zcvf model_output.tar.gz model_output")
    return send_from_directory(r"/open_explorer/web_app/temporary","model_output.tar.gz", as_attachment=True)

@app.route('/download_bin_model')
def download_bin_model():
    for filename in os.listdir('/open_explorer/web_app/temporary/model_output'):
        if filename.endswith(".bin"):
            return send_from_directory(r"/open_explorer/web_app/temporary/model_output",filename, as_attachment=True)

@app.route('/update_parameter', methods=['POST'])
def update_parameter():
    global dimension_type, model_name, width, height, model_type, input_type_rt, input_layout_rt, input_type_train, input_layout_train, norm_type, batch_size, epochs, has_model
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

    if dimension_type == "custom":
        width = data.get("customWidth")
        height = data.get("customHeight")
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


if __name__ == '__main__':
    # app.run(host="0.0.0.0",port=5000,debug=True)
    socketio.run(app,host="0.0.0.0",port=5001, debug=True, allow_unsafe_werkzeug=True)

