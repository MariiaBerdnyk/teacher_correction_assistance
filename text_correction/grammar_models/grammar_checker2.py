from transformers import T5Tokenizer, T5ForConditionalGeneration

class GrammarChecker:
    def __init__(self, model_path="./models/grammar-synthesis-small"):
        # Load the locally saved model and tokenizer
        self.tokenizer = T5Tokenizer.from_pretrained(model_path)
        self.model = T5ForConditionalGeneration.from_pretrained(model_path)

    def correct(self, text):
        # Preprocess the input text
        input_text = text  # Prefix for grammar correction in this model
        input_ids = self.tokenizer.encode(input_text, return_tensors="pt", truncation=True)

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
        return "Grammar Synthesis Small"

if __name__ == "__main__":
    # Initialize the grammar checker
    grammar_checker = GrammarChecker()

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