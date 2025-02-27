from transformers import T5Tokenizer, T5ForConditionalGeneration

class SpellChecker:
    def __init__(self, model_path="./models/t5_base_spellchecker_model"):
        self.tokenizer = T5Tokenizer.from_pretrained(model_path)
        self.model = T5ForConditionalGeneration.from_pretrained(model_path)
        self.max_length = 150

    def correct(self, input_text):
        # Preprocess the input text
        input_ids = self.tokenizer("spellcheck: " + input_text, return_tensors="pt").input_ids
        
        # Generate corrected text
        outputs = self.model.generate(
            input_ids,
            do_sample=False,
            max_length=self.max_length,
            # top_p=0.99,
            num_return_sequences=1
        )
        corrected_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return corrected_text
    
    def name(self):
        return "T5 Spell Checker"

if __name__ == "__main__":
    # Path to the locally saved model
    local_model_path = "./models/t5_base_spellchecker_model"

    # Initialize the SpellChecker class
    spell_checker = SpellChecker(local_model_path)

    # Example input text
    input_text = "christmas is celbrated on decembr 25 evry ear"
    
    # Correct the text
    corrected_text = spell_checker.correct(input_text)
    print("#" * 95, "\nCorrected text:", corrected_text)

    input_text2 = "the schol is vry far awy from my yhouse"

    corrected_text2 = spell_checker.correct(input_text2)
    print("#" * 95, "\nCorrected text:", corrected_text2)