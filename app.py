from flask import Flask, render_template,request,redirect,url_for,send_from_directory,flash,jsonify
from flask_socketio import SocketIO, emit
import subprocess
import os
import threading
import sys
import glob

app = Flask(__name__)
socketio = SocketIO(app)
app.secret_key = 'random string'

onnx_file_path = ""
prototxt_file_path = ""
caffemodel_file_path = ""
config_file_path = ""
image_folder_path = ""
#model_name用于匹配工具链对应的文件夹以及模型训练代码仓库
model_name = "mobilenetv1"
model_type = "onnx"

temporary_path = "/open_explorer/web_app/temporary"
search_path='/open_expolorer/horizon_xj3_open_explorer_v2.6.2b-py38_20230606/ddk/samples/ai_toolchain/horizon_model_convert_sample'
preprocess_py = '/open_expolorer/horizon_xj3_open_explorer_v2.6.2b-py38_20230606/ddk/samples/ai_toolchain/horizon_model_convert_sample/data_preprocess.py'

dimension_type = ""
wdith = 672
height = 672

@app.route('/',methods=["POST","GET"])
def index():
    return render_template('index.html')

@app.route('/upload',methods=["POST","GET"])
def upload():
    upload_message = ""
    global model_type,config_file_path,image_folder_path,onnx_file_path,prototxt_file_path,caffemodel_file_path
    if request.method == 'POST':
        try:
            if model_type == "onnx":
                onnx_file = request.files['onnx_file']
                onnx_file_path = os.path.join(temporary_path, onnx_file.filename)
                onnx_file.save(onnx_file_path)
            else:
                prototxt_file = request.files['prototxt_file']
                prototxt_file_path = os.path.join(temporary_path, onnx_file.filename)
                prototxt_file.save(prototxt_file_path)

                caffemodel_file = request.files['caffemodel_file']
                caffemodel_file_path = os.path.join(temporary_path, onnx_file.filename)
                caffemodel_file.save(caffemodel_file_path)


            config_file = request.files['config_file']
            config_file_path = os.path.join(temporary_path, config_file.filename)
            config_file.save(config_file_path)

            image_files = request.files.getlist('image_file')
            folder_name = os.path.dirname(image_files[0].filename)
            directory_path = os.path.join(temporary_path, folder_name)
            if not os.path.exists(directory_path):
                os.makedirs(directory_path)
            image_folder_path = directory_path
            for file in image_files:
                image_file_path = os.path.join(temporary_path, file.filename)
                file.save(image_file_path)
    
            upload_message = "Upload successful"
        except Exception as r:
            upload_message = r
        socketio.emit('output', upload_message) 
    return render_template('index.html')

def runCommandAndOutLog(command):
    # socketio.emit('clear_output')
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,text =True)
    for line in iter(process.stderr.readline, b''):
        if(len(line) == 0):
            break
        socketio.emit('output', line)
    for line in iter(process.stdout.readline, b''):
        if(len(line) == 0):
            break
        socketio.emit('output', line)

@app.route('/train', methods=['POST'])
def train():
    runCommandAndOutLog(f"cd /train/{model_name} && /usr/bin/python3 train.py")
    socketio.emit('output', "train completed") 
    return '', 200

@app.route('/export', methods=['POST'])
def export():
    runCommandAndOutLog(f"cd /train/{model_name} && /usr/bin/python3 export.py")
    socketio.emit('output', "export completed") 
    return '', 200

@app.route('/convert', methods=['POST'])
def convert():
    # find_file_in_folder(model_folder_name = model_name)
    if (onnx_file_path != "" or (prototxt_file_path != "" and caffemodel_file_path != ""))and config_file_path != "" and image_folder_path != "":
        #有模型
        #check
        runCommandAndOutLog(f"hb_mapper checker --model-type onnx --model {onnx_file_path} --march bernoulli2")
        socketio.emit('output', "模型检查完成") 

        #preprocess
        model_path = os.path.dirname(find_file_in_folder(model_folder_name = model_name))
        print(model_path)
        runCommandAndOutLog(f"cd {model_path} && /usr/bin/python3 {preprocess_py} --dst_dir {temporary_path}/calibration_data --pic_ext .rgb --read_mode opencv --saved_data_type float32 --src_dir {image_folder_path} ")
        socketio.emit('output', "校准数据准备完成") 

        # build
        runCommandAndOutLog(f"cd {temporary_path} && hb_mapper makertbin --config {config_file_path} --model-type {model_type} ")
        socketio.emit('output', "模型转换完成") 
    else:
        #没有模型，训练的
        #check
        runCommandAndOutLog(f"hb_mapper checker --model-type onnx --model {temporary_path}/best.onnx --march bernoulli2")
        socketio.emit('output', "模型检查完成\n") 

        #preprocess
        runCommandAndOutLog(f"/usr/bin/python3 data/generate_calibration_data.py  --dataset ./temporary/dataset --model {model_name}")
        socketio.emit('output', "校准数据准备完成") 

        # build
        runCommandAndOutLog(f"cd {temporary_path} && hb_mapper makertbin --config ./resnet18_config.yaml --model-type {model_type} ")
        socketio.emit('output', "模型转换完成") 

    return render_template('index.html')


def find_file_in_folder(model_folder_name):
    path = glob.glob(search_path + '/*/*' + model_name + '*/mapper/02_preprocess.sh')
    if len(path) != 0:
        print(path)
        return path[0]
    else:
        return None

@app.route('/download_all_model')
def download_all_model():
    os.system(f"tar zcvf temporary/model_output.tar.gz temporary/model_output")
    return send_from_directory(r"/open_explorer/web_app/temporary","model_output.tar.gz", as_attachment=True)

@app.route('/download_bin_model')
def download_bin_model():
    return send_from_directory(r"/open_explorer/web_app/temporary/model_output","yolov5s_672x672_nv12.bin", as_attachment=True)

@app.route('/update_parameter', methods=['POST'])
def update_parameter():
    global dimension_type, model_name, width, height, model_type
    data = request.get_json()
    model_name = data.get("model_name")
    dimension_type = data.get("dimensionType")
    model_type = data.get("model_type")
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
        width = 0
        height = 0
    print(data)
    return '', 200
    
if __name__ == '__main__':
    # app.run(host="0.0.0.0",port=5000,debug=True)
    socketio.run(app,host="0.0.0.0",port=5001, debug=True)

