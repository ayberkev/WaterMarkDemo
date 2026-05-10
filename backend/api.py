from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
from main import remove_watermark
import os

app = Flask(__name__)
CORS(app)

UPLOAD = "uploads"
RESULTS = "results"

os.makedirs(UPLOAD, exist_ok=True)
os.makedirs(RESULTS, exist_ok=True)


@app.route("/process", methods=["POST"])
def process():
    file = request.files["image"]

    filename = secure_filename(file.filename)
    input_path = os.path.join(UPLOAD, filename)
    output_path = os.path.join(RESULTS, filename)

    file.save(input_path)

    remove_watermark(input_path, output_path)

    return jsonify({
        "result": f"http://localhost:5000/{output_path}"
    })


@app.route('/results/<path:filename>')
def serve_result(filename):
    return send_from_directory(RESULTS, filename)


if __name__ == "__main__":
    app.run(debug=True)