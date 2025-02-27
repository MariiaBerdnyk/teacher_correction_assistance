import shutil
import threading
import torch
from flask import Flask, request, render_template, jsonify
import os
import time
import psutil  # For CPU temperature and frequency monitoring
import cpuinfo  # For detailed CPU information

from grammar_checker import GrammarChecker
from image2text_recognition import load_images_from_directory, OCRModel, process_batch
from preprocess_image import remove_background, crop_lines, load_image_as_bytes
from spell_checker import SpellChecker

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT_DIR)
IMAGES_FROM_FRONT = "static/images"
IMAGES_FOR_LIB = os.path.join("", IMAGES_FROM_FRONT)
PROCESSED_IMAGES = os.path.join(IMAGES_FROM_FRONT, "processed")
PROCESSED_IMAGES_LIB = os.path.join("", PROCESSED_IMAGES)
LINES_DIR = os.path.join(PROCESSED_IMAGES, "lines")
LINES_DIR_LIB = os.path.join("", LINES_DIR)

app = Flask(__name__)
device = torch.device("cpu")
monitoring = True  # Flag to control monitoring thread

def monitor_temperature_and_frequency():
    """Logs CPU temperature and frequency every second."""
    while monitoring:
        temp = get_cpu_temperature()
        freq = get_cpu_frequency()
        print(f"[MONITOR] CPU Temperature: {temp}°C, Frequency: {freq} MHz")
        time.sleep(1)


def benchmark(func):
    """Decorator to measure the execution time, CPU temperature, and frequency of a function."""
    def wrapper(*args, **kwargs):
        start_time = time.time()

        # Capture initial CPU temperature and frequency
        temp_start = get_cpu_temperature()
        freq_start = get_cpu_frequency()

        result = func(*args, **kwargs)

        # Capture final CPU temperature and frequency
        temp_end = get_cpu_temperature()
        freq_end = get_cpu_frequency()

        end_time = time.time()
        print(f"Execution time for {func.__name__}: {end_time - start_time:.4f} seconds")
        print(f"Temperature - Start: {temp_start}°C, End: {temp_end}°C")
        print(f"Frequency - Start: {freq_start} MHz, End: {freq_end} MHz")

        return result
    return wrapper

def get_cpu_temperature():
    """Retrieve CPU temperature in Celsius (if supported)."""
    try:
        sensors = psutil.sensors_temperatures()
        if "coretemp" in sensors:  # Common label for CPU sensors
            core_temps = sensors["coretemp"]
            return core_temps[0].current  # Return the temperature of the first core
        return "N/A"
    except Exception as e:
        print(f"Could not retrieve CPU temperature: {e}")
        return "N/A"

def get_cpu_frequency():
    """Retrieve current CPU frequency in MHz."""
    try:
        return psutil.cpu_freq().current  # Current frequency in MHz
    except Exception as e:
        print(f"Could not retrieve CPU frequency: {e}")
        return "N/A"

@benchmark
def clean_lines_dir(directory_path=LINES_DIR):
    if os.path.exists(directory_path) and os.path.isdir(directory_path):
        shutil.rmtree(directory_path)

@benchmark
def crop_with_projection(projection_value):
    clean_lines_dir()
    line_images_dir = LINES_DIR

    image = load_image_as_bytes(PROCESSED_IMAGES_LIB, "preprocessed_image.png")
    crop_lines(image, LINES_DIR_LIB, projection_value=projection_value)
    lines_images = [os.path.join(line_images_dir, filename) for filename in os.listdir(LINES_DIR) if
                    filename.endswith('.png')]

    return lines_images

@benchmark
def preprocess_image(image_path, image_name):
    processed_image = os.path.join(PROCESSED_IMAGES, "preprocessed_image.png")
    clean_lines_dir()

    image = remove_background(image_path, image_name, PROCESSED_IMAGES_LIB, "preprocessed_image.png")
    crop_lines(image, LINES_DIR_LIB)

    lines_images = [os.path.join(LINES_DIR, filename) for filename in os.listdir(LINES_DIR) if
                    filename.endswith('.png')]
    return processed_image, lines_images

@benchmark
def extract_text():
    images = load_images_from_directory(LINES_DIR_LIB)
    model = OCRModel(device, model_path="../models/quantized_ocr_fp16.pth",load_local_model=True)


    # Process images in batches
    generated_text = process_batch(images, model)

    return generated_text


@benchmark
def syntax_check(text):
    speller = SpellChecker()
    corrected_text = speller.correct(text)
    return corrected_text

@benchmark
def grammar_check(text):
    grammar_checker = GrammarChecker(device, model_path="../models/quantized_grammar_checker_fp16.pth",load_local_model=True)

    corrected_text = grammar_checker.correct(text)
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
            device = xm.xla_device()
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
    decision = request.json.get('decision')
    if decision == 'yes':
        preprocessed_image, lines = preprocess_image(IMAGES_FOR_LIB, "uploaded_image.jpg")
        return jsonify({"status": "preprocessed", "image": preprocessed_image, "lines": lines})
    return jsonify({"status": "redo"})

@app.route('/extract-text', methods=['POST'])
def check_preprocessing():
    decision = request.json.get('decision')
    if decision == 'yes':
        extracted_text = extract_text()
        return jsonify({"status": "text_extracted", "text": extracted_text})
    return jsonify({"status": "redo"})

@app.route('/check-syntax', methods=['POST'])
def check_syntax():
    text = request.json.get('text')
    fixed_version = syntax_check(text)
    return jsonify({"original": text, "fixed": fixed_version})

@app.route('/set-device', methods=['POST'])
def set_device():
    selected_device_text = request.json.get('device')
    status, reason = verify_and_set_device(selected_device_text)
    return jsonify({"status": status, "device": selected_device_text, "reason": reason})

@app.route('/check-grammar', methods=['POST'])
def check_grammar():
    text = request.json.get('text')
    fixed_version = grammar_check(text)
    return jsonify({"original": text, "fixed": fixed_version})

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
    return jsonify({"lines": lines})

if __name__ == '__main__':
    # Start monitoring thread
    monitoring_thread = threading.Thread(target=monitor_temperature_and_frequency, daemon=True)
    monitoring_thread.start()

    try:
        app.run(debug=True, host="0.0.0.0", port=5000)
    finally:
        # Stop monitoring when application shuts down
        monitoring = False
        monitoring_thread.join()

