from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

class GrammarChecker:
    def __init__(self, model_path="./models/prithivida_grammar_error_correcter_v1"):
        # Load the locally saved model and tokenizer
        self.tokenizer = T5Tokenizer.from_pretrained(model_path)
        self.model = T5ForConditionalGeneration.from_pretrained(model_path)
        # True if ends with fp16
        self.quantized_model = model_path.endswith("fp16")

    def correct(self, text):
        # Preprocess the input text
        input_text = "gec: " + text  # "gec:" is the prefix for grammar correction in this model
        input_ids = self.tokenizer.encode(input_text, return_tensors="pt")

        # Generate corrected text
        outputs = self.model.generate(input_ids, max_length=512)
        corrected_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return corrected_text

    def summary(self):
        # Calculate and print the model's summary
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        return f"Model Summary:\n- Total Parameters: {total_params:,}\n- Trainable Parameters: {trainable_params:,}"
    
    def name(self):
        if self.quantized_model:
            return "T5 Quantized Grammar Checker"
        return "T5 Grammar Checker"
    
    def get_model(self):
        return self.model
    
    def set_model(self, model):
        self.model = model
    
    def get_tokenizer(self):
        return self.tokenizer
    
    def tokenize(self, text):
        return self.tokenizer.tokenize(text)

    def quantize(self):
        # Apply dynamic quantization
        self.model = torch.quantization.quantize_dynamic(
            self.model,
            {torch.nn.Linear},  # Specify which layers to quantize
            dtype=torch.qint8  # Use 8-bit integer quantization
        )
        print("Model quantized successfully.")

if __name__ == "__main__":
    # Initialize the grammar checker
    grammar_checker = GrammarChecker(model_path="./models/prithivida_grammar_error_correcter_v1_fp16")

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