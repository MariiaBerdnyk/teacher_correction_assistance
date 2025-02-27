import torch
# import torch_xla.core.xla_model as xm

from .model import Model
from transformers import T5Tokenizer, T5ForConditionalGeneration


class GrammarChecker(Model):
    def __init__(self, device, quantization: str = "default", model_path: str = None, quantized_model_int8_path: str = None):
        # Initialize tokenizer
        self.tokenizer = T5Tokenizer.from_pretrained(model_path if model_path else "t5-small")

        # Initialize model based on parameters
        if model_path:
            print("Loading selected model...")
            self.model = T5ForConditionalGeneration.from_pretrained(model_path)
            self.model_name = "Custom Grammar Model"
        elif quantized_model_int8_path or quantization == "int8":
            print("Loading quantized model [int8]...")
            path = quantized_model_int8_path or "./models/quantized_grammar_checker_int8.pth"
            try:
                self.model = torch.load(path)
                self.model.eval()
                self.model_name = "Quantized Grammar Model [int8]"
            except FileNotFoundError:
                raise ValueError(f"Quantized model not found at {path}")
        elif quantization == "float16":
            print("Loading prithivida/grammar_error_correcter_v1 [quantized into float16] model...")
            self.model = T5ForConditionalGeneration.from_pretrained("./models/prithivida_grammar_error_correcter_v1_fp16")
            self.model_name = "Quantized Grammar Model [float16]"
        else:
            print("Loading default prithivida/grammar_error_correcter_v1 model...")
            self.model = T5ForConditionalGeneration.from_pretrained("./models/prithivida_grammar_error_correcter_v1")
            self.model_name = "Default Grammar Model"
        self.device = device
        self.model = self.model.to(device=device)


    def correct(self, text: str) -> str:
        """
            Correct grammar error of text using the model
            :param text: text to correct
            :return the text with the grammmar errors corrected
        """
        if not text.strip():
            return "Input text is empty."

        # Preprocess the input text
        input_text = "gec: " + text  # "gec:" is the prefix for grammar correction in this model
        input_ids = self.tokenizer.encode(input_text, return_tensors="pt").to(device=self.device)

        # Generate corrected text
        outputs = self.model.generate(input_ids, max_length=512)
        corrected_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return corrected_text

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


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Initialize the grammar checker
    # Initialize the grammar checker
    grammar_checker = GrammarChecker(device, model_path="prithivida/grammar_error_correcter_v1")

    # Print the summary of the model
    print(grammar_checker.summary())

    # Example input text
    influent_sentences = [
        "He are moving here.",
        "I am doing fine. How is you?",
        "How is they?",
        "Matt like fish",
        "the collection of letters was original used by the ancient Romans",
        "We enjoys horror movies",
        "Anna and Mike is going skiing",
        "I walk to the store and I bought milk",
        " We all eat the fish and then made dessert",
        "I will eat fish for dinner and drink milk",
        "what be the reason for everyone leave the company",
    ]

    # Perform grammar correction
    for input_text in influent_sentences:
        corrected_text = grammar_checker.correct(input_text)
        print("Original:", input_text)
        print("Corrected:", corrected_text)
        print("----------\n")
