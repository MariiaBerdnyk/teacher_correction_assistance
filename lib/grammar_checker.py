import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

class GrammarChecker:
    def __init__(self, device, quantization: str = "default", model_path: str = None, quantized_model_int8_path: str = None, load_local_model=False):
        # Validate arguments
        if load_local_model and not model_path:
            raise ValueError("load_local_model is True, but no model_path is provided.")

        # Initialize tokenizer
        self.tokenizer = T5Tokenizer.from_pretrained("prithivida/grammar_error_correcter_v1")
        print(f"Tokenizer initialized: {type(self.tokenizer)}")

        # Initialize model
        if load_local_model:
            self.model = torch.load(model_path, weights_only=False)
        elif model_path:
            self.model = T5ForConditionalGeneration.from_pretrained(model_path)
        elif quantized_model_int8_path or quantization == "int8":
            path = quantized_model_int8_path or "./models/quantized_grammar_checker_int8.pth"
            self.model = torch.load(path)
        elif quantization == "float16":
            self.model = T5ForConditionalGeneration.from_pretrained("./models/prithivida_grammar_error_correcter_v1_fp16")
        else:
            self.model = T5ForConditionalGeneration.from_pretrained("./models/prithivida_grammar_error_correcter_v1")
        
        self.model = self.model.to(device=device)
        self.device = device

    def correct(self, text: str) -> str:
        if not text.strip():
            return "Input text is empty."

        input_text = "gec: " + text
        input_ids = self.tokenizer.encode(input_text, return_tensors="pt").to(device=self.device)
        outputs = self.model.generate(input_ids, max_length=512)
        corrected_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return corrected_text

    def get_tokenizer(self):
        return self.tokenizer

    def tokenize(self, text: str):
        return self.tokenizer.tokenize(text)

    def quantize(self):
        try:
            self.model = torch.quantization.quantize_dynamic(self.model, {torch.nn.Linear}, dtype=torch.qint8)
            print("Model quantized successfully.")
        except Exception as e:
            print(f"Quantization failed: {e}")

    def summary(self):
        return str(self.model)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    grammar_checker = GrammarChecker(device, model_path="prithivida/grammar_error_correcter_v1")

    influent_sentences = [
        "He are moving here.",
        "I am doing fine. How is you?",
        "Matt like fish",
    ]

    for input_text in influent_sentences:
        corrected_text = grammar_checker.correct(input_text)
        print("Original:", input_text)
        print("Corrected:", corrected_text)
        print("----------\n")

