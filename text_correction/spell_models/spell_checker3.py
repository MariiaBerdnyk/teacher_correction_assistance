from autocorrect import Speller

class SpellChecker():
    def __init__(self, lang='', fast=False):
        # Fast mode is faster (micro seconds), however words with double typos won't be corrected 
        if lang == '':
            self.speller = Speller(fast=fast)
        else:
            self.speller = Speller(lang=lang, fast=fast)

    def correct(self, input_text):
        return self.speller(input_text)
    
    def name(self):
        return "Autocorrect Spell Checker"
    
if __name__ == "__main__":
    speller = SpellChecker(fast=True)

    sentences_with_spelling_errors = [
        "She is goinng to the libary to borrow a bok.",
        "The weather is beatiful tooday for a piknic.",
        "I recived an emale from my freind yestarday.",
        "My favorate subject in scool is mathamatics.",
        "He forgott his umbrela at home.",
        "Th ecat was chasing it's own tail in circls.",
        "We dicided to viset the musium this wekend.",
        "The child was excitted to open his birthday presant.",
        "Pleese rember to bring your lunch tomorow.",
        "I tride to solve the puzzel but it was too dificult."
    ]

    for sentence in sentences_with_spelling_errors:
        print("------", "\nOriginal text:", sentence)
        corrected_text = speller.correct(sentence)
        print("Corrected text:", corrected_text)
