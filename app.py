from flask import Flask, render_template,request,redirect,url_for,send_from_directory,flash,jsonify
from flask_socketio import SocketIO, emit
import subprocess
import os
import threading
import sys
app = Flask(__name__)
socketio = SocketIO(app)
app.secret_key = 'random string'

onnx_file_path = ""
prototxt_file_path = ""
caffemodel_file_path = ""
config_file_path = ""
image_folder_path = ""
model_name = "mobilenetv1"
model_type = "onnx"

uploads_path = "/open_explorer/web_app/uploads"
search_path='/open_expolorer/horizon_xj3_open_explorer_v2.6.2b-py38_20230606/ddk/samples/ai_toolchain/horizon_model_convert_sample'
preprocess_py = '/open_expolorer/horizon_xj3_open_explorer_v2.6.2b-py38_20230606/ddk/samples/ai_toolchain/horizon_model_convert_sample/data_preprocess.py'
@app.route('/',methods=["POST","GET"])
def upload():
    upload_message = ""
    global model_name,model_type,config_file_path,image_folder_path,onnx_file_path,prototxt_file_path,caffemodel_file_path
    if request.method == 'POST':
        try:
            model_name = request.form.get("model_name")
            model_type = request.form.get("model_type")

            if model_type == "onnx":
                onnx_file = request.files['onnx_file']
                onnx_file_path = os.path.join(uploads_path, onnx_file.filename)
                onnx_file.save(onnx_file_path)
            else:
                prototxt_file = request.files['prototxt_file']
                prototxt_file_path = os.path.join(uploads_path, onnx_file.filename)
                prototxt_file.save(prototxt_file_path)

                caffemodel_file = request.files['caffemodel_file']
                caffemodel_file_path = os.path.join(uploads_path, onnx_file.filename)
                caffemodel_file.save(caffemodel_file_path)


            config_file = request.files['config_file']
            config_file_path = os.path.join(uploads_path, config_file.filename)
            config_file.save(config_file_path)

            image_files = request.files.getlist('image_file')
            folder_name = os.path.dirname(image_files[0].filename)
            directory_path = os.path.join(uploads_path, folder_name)
            if not os.path.exists(directory_path):
                os.makedirs(directory_path)
            image_folder_path = directory_path
            for file in image_files:
                image_file_path = os.path.join(uploads_path, file.filename)
                file.save(image_file_path)
    
            upload_message = "Upload successful"
        except Exception as r:
            upload_message = r
        flash(upload_message)
    return render_template('index.html')

def runCommandAndOutLog(command):
    socketio.emit('clear_output')
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,text =True)
    for line in iter(process.stderr.readline, b''):
        if(len(line) == 0):
            break
        socketio.emit('output', line)
    for line in iter(process.stdout.readline, b''):
        if(len(line) == 0):
            break
        socketio.emit('output', line)

@app.route('/check', methods=['POST'])
def check():
    if model_type == "onnx":
        # runCommandAndOutLog(f"hb_mapper checker --model-type onnx --model /open_explorer/web_app/uploads/best.onnx --march bernoulli2")
        runCommandAndOutLog(f"hb_mapper checker --model-type onnx --model {onnx_file_path} --march bernoulli2")
    else:
        runCommandAndOutLog(f"hb_mapper checker --model-type caffe --proto {prototxt_file_path} --model {caffemodel_file_path} --march bernoulli2") 

    socketio.emit('output', "Check completed") 

    # return jsonify(result=result)
    return render_template('index.html')

@app.route('/preprocess', methods=['POST'])
def preprocess():
    model_path = os.path.dirname(find_file_in_folder(model_folder_name = model_name))
    runCommandAndOutLog(f"cd {model_path} && python3 {preprocess_py} --dst_dir {uploads_path}/calibration_data --pic_ext .rgb --read_mode opencv --saved_data_type float32 --src_dir {image_folder_path} ")
    socketio.emit('output', "Preprocess completed") 
    return render_template('index.html')

@app.route('/build', methods=['POST'])
def build():
    runCommandAndOutLog(f"hb_mapper makertbin --config {config_file_path} --model-type {model_type} ")
    socketio.emit('output', "Build completed") 
    return render_template('index.html')

def find_file_in_folder(model_folder_name,file_name = '02_preprocess.sh'):
    for root, dirs, files in os.walk(search_path):
        if file_name in files and model_folder_name in root.split(os.path.sep):
            return os.path.join(root, file_name)
    return None

@app.route('/download_all_model')
def download_all_model():
    os.system(f"tar zcvf uploads/model_output.tar.gz uploads/model_output")
    return send_from_directory(r"/open_explorer/web_app/uploads","model_output.tar.gz", as_attachment=True)

@app.route('/download_bin_model')
def download_bin_model():
    print('downloading ...')
    return send_from_directory(r"/open_explorer/web_app/uploads/model_output","yolov5s_672x672_nv12.bin", as_attachment=True)


if __name__ == '__main__':
    # app.run(host="0.0.0.0",port=5000,debug=True)
    socketio.run(app,host="0.0.0.0",port=5000, debug=True)

