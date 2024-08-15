import os
import shutil
from flask import Flask, render_template, request, redirect, url_for, send_file, jsonify
from werkzeug.utils import secure_filename
from PIL import Image
from image_processor import process_image_app, data, clean_folder
import json
import math

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULT_FOLDER'] = 'results'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

if not os.path.exists(app.config['RESULT_FOLDER']):
    os.makedirs(app.config['RESULT_FOLDER'])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    user_input = request.form.get('user_input')
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        result_images, result_text, result_data = process_images(file_path, user_input)
        json_dir = os.path.join(app.config['RESULT_FOLDER'], 'passport_text_executor')

        if not os.path.exists(json_dir):
            os.makedirs(json_dir)

        with open(os.path.join(json_dir, 'result_data.json'), 'w', encoding='utf-8') as json_file:
            json.dump(result_data, json_file, ensure_ascii=False, indent=4)

        return render_template('results.html', images=result_images, text=result_text, json_data=result_data)
    return redirect(url_for('index'))

@app.route('/download/<filename>')
def download_file(filename):
    return send_file(os.path.join(app.config['RESULT_FOLDER'], filename), as_attachment=True)

def process_images(current_photo, base_dir):
    process_image_app(current_photo, base_dir)
    photo_name = os.path.basename(current_photo)

    if photo_name in data:
        text = data[photo_name]['text']
        transformed_image_dir = data[photo_name]['transformed_image_dir']

        if os.path.exists(transformed_image_dir) and os.path.isdir(transformed_image_dir):
            all_photos = [os.path.join(transformed_image_dir, file) for file in os.listdir(transformed_image_dir) if
                          os.path.isfile(os.path.join(transformed_image_dir, file))]
            return all_photos, text, data[photo_name]
        else:
            return [], text, data[photo_name]
    else:
        return [], '', {}

if __name__ == '__main__':
    app.run(debug=True)