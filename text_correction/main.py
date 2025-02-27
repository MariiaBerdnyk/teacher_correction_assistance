from model_frameworks.grammar import GrammarModel
from model_frameworks.spelling import SpellingModel

class Corrector:
    def __init__(self):
        print("Initializing Corrector...")
        self.grammar_corrector = GrammarModel(quantization="int8")
        self.spelling_corrector = SpellingModel()
    
    def correct_text(self, text: str) -> str:
        print(f"Original text: {text}")
        
        # Step 1: Correct spelling
        corrected_spelling = self.correct_spelling(text)
        print(f"After spelling correction: {corrected_spelling}")
        
        # Step 2: Correct grammar
        corrected_text = self.correct_grammar(corrected_spelling)
        print(f"After grammar correction: {corrected_text}")
        
        return corrected_text

    def correct_grammar(self, text: str) -> str:
        try:
            return self.grammar_corrector.correct(text)
        except Exception as e:
            print(f"Error in grammar correction: {e}")
            return text

    def correct_spelling(self, text: str) -> str:
        try:
            return self.spelling_corrector.correct(text)
        except Exception as e:
            print(f"Error in spelling correction: {e}")
            return text


if __name__ == "__main__":
    # Example usage
    corrector = Corrector()

    # Test sentences
    sentences = [
        "He are moving here.",
        "I am doin fine. How is yu?",
        "They is happy with ther new home.",
    ]

    print("\nStarting correction process...\n")
    for sentence in sentences:
        corrected = corrector.correct_text(sentence)
        print(f"Final Output: {corrected}")
        print("-----------\n")
