import os
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

class GrammarModel:
    def __init__(self, quantization: str = "default", model_path: str = None, quantized_model_int8_path: str = None):
        # Resolve base path relative to this file
        self.base_path = os.path.dirname(os.path.abspath(__file__))  # Get the directory of this script

        # Initialize tokenizer
        self.tokenizer = T5Tokenizer.from_pretrained(model_path if model_path else "t5-small")

        # Initialize model based on parameters
        if model_path:
            print("Loading selected model...")
            self.model = T5ForConditionalGeneration.from_pretrained(model_path)
            self.model_name = "Custom Grammar Model"
        elif quantized_model_int8_path or quantization == "int8":
            print("Loading quantized model [int8]...")
            path = quantized_model_int8_path or os.path.join(self.base_path,"..", "..", "models", "quantized_grammar_checker_int8.pth")
            try:
                self.model = torch.load(path)
                self.model.eval()
                self.model_name = "Quantized Grammar Model [int8]"
            except FileNotFoundError:
                raise ValueError(f"Quantized model not found at {path}")
        elif quantization == "float16":
            print("Loading prithivida/grammar_error_correcter_v1 [quantized into float16] model...")
            self.model = T5ForConditionalGeneration.from_pretrained(os.path.join(self.base_path, "..", "..","models", "prithivida_grammar_error_correcter_v1_fp16"))
            self.model_name = "Quantized Grammar Model [float16]"
        else:
            print("Loading default prithivida/grammar_error_correcter_v1 model...")
            self.model = T5ForConditionalGeneration.from_pretrained(os.path.join(self.base_path, "..", "..","models", "prithivida_grammar_error_correcter_v1"))
            self.model_name = "Default Grammar Model"

    def correct(self, text: str):
        if not text.strip():
            return "Input text is empty."
        
        # Preprocess the input text
        input_text = "gec: " + text  # "gec:" is the prefix for grammar correction in this model
        input_ids = self.tokenizer.encode(input_text, return_tensors="pt")

        # Generate corrected text
        outputs = self.model.generate(input_ids, max_length=512)
        corrected_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return corrected_text

    def summary(self):
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        return f"Model Summary:\n- Total Parameters: {total_params:,}\n- Trainable Parameters: {trainable_params:,}"

    def model_name(self):
        return self.model_name

    def get_model(self):
        print("Model:", self.model)
        return self.model

    def set_model(self, model):
        print("Model set. New model:", model)
        self.model = model

    def get_tokenizer(self):
        return self.tokenizer

    def tokenize(self, text: str):
        return self.tokenizer.tokenize(text)

    def quantize(self):
        try:
            self.model = torch.quantization.quantize_dynamic(
                self.model,
                {torch.nn.Linear},  # Specify which layers to quantize
                dtype=torch.qint8  # Use 8-bit integer quantization
            )
            print("Model quantized successfully.")
        except Exception as e:
            print(f"Quantization failed: {e}")

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)
