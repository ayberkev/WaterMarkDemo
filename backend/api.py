from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
from main import remove_watermark
import os
import uuid

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD = os.path.join(BASE_DIR, 'uploads')
RESULTS = os.path.join(BASE_DIR, 'results')
ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp'}

os.makedirs(UPLOAD, exist_ok=True)
os.makedirs(RESULTS, exist_ok=True)


def allowed_file(filename: str) -> bool:
    return os.path.splitext(filename.lower())[1] in ALLOWED_EXTENSIONS


@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})


@app.route('/process', methods=['POST'])
def process():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'İstek içinde image alanı bulunamadı.'}), 400

        file = request.files['image']
        if not file or not file.filename:
            return jsonify({'error': 'Lütfen bir görsel seçin.'}), 400

        if not allowed_file(file.filename):
            return jsonify({'error': 'Desteklenmeyen dosya türü. JPG, JPEG, PNG veya WEBP yükleyin.'}), 400

        sensitivity = (request.form.get('sensitivity') or 'medium').lower().strip()
        if sensitivity not in {'low', 'medium', 'high'}:
            sensitivity = 'medium'

        original_name = secure_filename(file.filename)
        stem, ext = os.path.splitext(original_name)
        unique_id = uuid.uuid4().hex[:8]
        input_filename = f'{stem}_{unique_id}{ext}'
        output_filename = f'{stem}_{unique_id}_cleaned{ext}'
        mask_filename = f'{stem}_{unique_id}_mask.png'

        input_path = os.path.join(UPLOAD, input_filename)
        output_path = os.path.join(RESULTS, output_filename)
        mask_output_path = os.path.join(RESULTS, mask_filename)

        file.save(input_path)
        result, mask_path = remove_watermark(
            input_path,
            output_path,
            sensitivity=sensitivity,
            mask_output_path=mask_output_path,
        )

        if result is None or not os.path.exists(output_path):
            return jsonify({'error': 'Görsel işlenemedi.'}), 500

        base_url = request.host_url.rstrip('/')
        payload = {
            'result': f'{base_url}/results/{output_filename}',
            'filename': output_filename,
            'sensitivity': sensitivity,
            'message': 'İşlem tamamlandı.',
        }
        if mask_path and os.path.exists(mask_output_path):
            payload['mask'] = f'{base_url}/results/{mask_filename}'

        return jsonify(payload)
    except Exception as exc:
        return jsonify({'error': f'İşlem sırasında hata oluştu: {str(exc)}'}), 500


@app.route('/results/<path:filename>')
def serve_result(filename):
    return send_from_directory(RESULTS, filename)


if __name__ == '__main__':
    app.run(debug=True)
