import shutil

import torch
from flask import Flask, request, render_template, jsonify
import os

from lib.grammar_checker import GrammarChecker
from lib.image2text_recognition import load_images_from_directory, OCRModel, process_batch
from lib.preprocess_image import remove_background, crop_lines, load_image_as_bytes
from lib.spell_checker import SpellChecker

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT_DIR)
IMAGES_FROM_FRONT = "static/images"
IMAGES_FOR_LIB = os.path.join("../front", IMAGES_FROM_FRONT)
PROCESSED_IMAGES = os.path.join(IMAGES_FROM_FRONT, "processed")
PROCESSED_IMAGES_LIB = os.path.join("../front", PROCESSED_IMAGES)
LINES_DIR = os.path.join(PROCESSED_IMAGES, "lines")
LINES_DIR_LIB = os.path.join("../front", LINES_DIR)

app = Flask(__name__)
device = torch.device("cpu")

def clean_lines_dir(directory_path = LINES_DIR):
    if os.path.exists(directory_path) and os.path.isdir(directory_path):
        shutil.rmtree(directory_path)

def crop_with_projection(projection_value):
    clean_lines_dir()
    line_images_dir = LINES_DIR

    image = load_image_as_bytes(PROCESSED_IMAGES_LIB, "preprocessed_image.png")
    crop_lines(image, LINES_DIR_LIB, projection_value=projection_value)
    lines_images = [os.path.join(line_images_dir, filename) for filename in os.listdir(LINES_DIR) if
                        filename.endswith('.png')]

    return lines_images

def preprocess_image(image_path, image_name):
    # Placeholder: Preprocess the image and return the preprocessed image path and list of image paths for lines
    processed_image = os.path.join(PROCESSED_IMAGES, "preprocessed_image.png")
    # Directory where line images are stored
    # os.makedirs(line_images_dir, exist_ok=True)

    clean_lines_dir()

    image = remove_background(image_path, image_name, PROCESSED_IMAGES_LIB, "preprocessed_image.png")

    crop_lines(image, LINES_DIR_LIB)

    # Example: Assuming images are already created or you process them here
    # Get all image files from the directory
    lines_images = [os.path.join(LINES_DIR, filename) for filename in os.listdir(LINES_DIR) if
                    filename.endswith('.png')]
    return processed_image, lines_images

def extract_text(lines_dir, batch_size=4, deviceSelected=None):
    images = load_images_from_directory(lines_dir)
    if deviceSelected:
        model = OCRModel(deviceSelected)
    else:
        model = OCRModel(device)

    # Process images in batches
    generated_text = process_batch(images, model, batch_size)

    return generated_text

def syntax_check(text):
    speller = SpellChecker()
    corrected_text = speller.correct(text)
    return corrected_text

def split_text_by_sentence(text):
    portions = []
    while len(text) > 512:
        # Find the last period within the first 512 characters
        split_index = text[:512].rfind('.')
        if split_index == -1:
            # If no period is found, split at 512 characters
            split_index = 512
        else:
            # Include the period in the split
            split_index += 1
        # Append the portion and trim the text
        portions.append(text[:split_index].strip())
        text = text[split_index:].strip()
    # Append any remaining text
    if text:
        portions.append(text)
    return portions

def grammar_check(text, deviceSelected=None):
    if deviceSelected:
        grammar_checker = GrammarChecker(deviceSelected, model_path="prithivida/grammar_error_correcter_v1")
    else:
        grammar_checker = GrammarChecker(device, model_path="prithivida/grammar_error_correcter_v1")

    portions = split_text_by_sentence(text)

    corrected_text = ""
    for portion in portions:
        corrected_text += grammar_checker.correct(portion)

    return corrected_text

def verify_and_set_device(device_selected_text):
    global device
    if device_selected_text == "cuda":
        if not torch.cuda.is_available():
            return False, "GPU is unavailable on your device!"
        device = torch.device("cuda")
    elif device_selected_text == "tpu":
        try:
            import torch_xla.core.xla_model as xm
            # Check if TPU device is accessible
            device = xm.xla_device()  # This will succeed if XLA is available
        except ImportError:
            return False, "TPU is unavailable on your device!"
    elif device_selected_text == "mps":
        if not torch.backends.mps.is_available():
            return False, "Metal is unavailable on your device!"
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    return True, "OK"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/check-image-quality', methods=['POST'])
def check_image_quality():
    # Handle user decision on image quality
    decision = request.json.get('decision')
    if decision == 'yes':
        preprocessed_image, lines = preprocess_image(IMAGES_FOR_LIB, "uploaded_image.jpg")
        # {{url_for('static', preprocessed_image)}}
        return jsonify({"status": "preprocessed", "image": preprocessed_image, "lines": lines})
    return jsonify({"status": "redo"})

@app.route('/extract-text', methods=['POST'])
def check_preprocessing():
    decision = request.json.get('decision')
    if decision == 'yes':
        extracted_text = extract_text(LINES_DIR_LIB)
        return jsonify({"status": "text_extracted", "text": extracted_text})
    return jsonify({"status": "redo"})

@app.route('/check-syntax', methods=['POST'])
def check_syntax():
    text = request.json.get('text')
    syntax_fixed, fixed_version = syntax_check(text)
    return jsonify({"original": syntax_fixed, "fixed": fixed_version})

@app.route('/set-device', methods=['POST'])
def set_device():
    selected_device_text = request.json.get('device')

    status, reason = verify_and_set_device(selected_device_text)

    print(selected_device_text)

    print(device.type)

    return jsonify({"status": status, "device": selected_device_text, "reason": reason})

@app.route('/check-grammar', methods=['POST'])
def check_grammar():
    text = request.json.get('text')
    grammar_fixed, fixed_version = grammar_check(text)
    return jsonify({"original": grammar_fixed, "fixed": fixed_version})

@app.route('/check-syntax-and-grammar', methods=['POST'])
def check_syntax_and_grammar():
    text = request.json.get('text')

    fixed_syntax_version = syntax_check(text)
    fixed_grammar_version = grammar_check(fixed_syntax_version)

    return jsonify({"syntaxFixed": fixed_syntax_version, "grammarFixed": fixed_grammar_version})

@app.route('/crop_with_projection', methods=['POST'])
def crop_projection():
    projection = request.json.get('projectionValue')
    lines = crop_with_projection(projection)
    print(lines)
    return jsonify({"lines": lines})


if __name__ == '__main__':
    app.run(debug=True)