import os
import torch

from lib.grammar_checker import GrammarChecker
from lib.image2text_recognition import OCRModel
from lib.model import Model


def dynamic_quantization_model(model, dtype: torch.dtype):
    quantization_successful = True
    exception = None
    try:
        model = torch.quantization.quantize_dynamic(
            model,
            {torch.nn.Linear},  # Specify which layers to quantize
            dtype=dtype  # Use 8-bit integer quantization
        )
        quantization_successful = True
    except Exception as e:
        quantization_successful = False
        exception = e
    return quantization_successful, model, exception

def create_dir_if_not_exist(path):
    """Create a dir at path if not exists."""
    os.makedirs(path, exist_ok=True)

def save_model(model_wrapper: Model, save_path):
    torch.save(model_wrapper.get_model(), save_path)

def quantize_all_models(device, grammar_checker_path, ocr_model_path):
    grammar_checker = GrammarChecker(device, model_path=grammar_checker_path)
    ocr_model = OCRModel(device, model_path=ocr_model_path)
    grammar_checker.quantize_model()
    ocr_model.quantize_model()
    create_dir_if_not_exist("models/")
    save_model(grammar_checker, "models/quantized_grammar_checker_fp16.pth")
    save_model(ocr_model, "models/quantized_ocr_fp16.pth")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else xm.xla_device())
    quantize_all_models(device, "prithivida/grammar_error_correcter_v1", "microsoft/trocr-base-handwritten")
