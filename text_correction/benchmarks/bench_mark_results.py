from spell_checker import SpellChecker as SpellChecker1
from spell_checker2 import SpellChecker as SpellChecker2
from spell_checker3 import SpellChecker as SpellChecker3
from grammar_checker import GrammarChecker as GrammarChecker1
from grammar_checker2 import GrammarChecker as GrammarChecker2
import torch
import time
from rouge import Rouge  # Install with pip install rouge


class ResultFormatter:
    @staticmethod
    def display_results(checker_name, original_sentences, corrections, references, rouge_scores):
        print(f"Results for {checker_name}:\n")
        for i, (original, corrected, reference, score) in enumerate(
            zip(original_sentences, corrections, references, rouge_scores)
        ):
            print(f"Sentence {i + 1}:")
            print(f"  Original: {original}")
            print(f"  Corrected: {corrected}")
            print(f"  Reference: {reference}")
            print(f"  ROUGE-L Score: {score:.4f}")
            print("----------\n")

    @staticmethod
    def display_summary(checker_name, average_time, min_time, max_time, average_score):
        print(f"Summary for {checker_name}:")
        print(f"  Average Time: {average_time:.4f} seconds")
        print(f"  Min Time: {min_time:.4f} seconds")
        print(f"  Max Time: {max_time:.4f} seconds")
        print(f"  Average ROUGE-L Score: {average_score:.4f}")
        print("==========\n")

class QuantizedGrammarChecker:
    def __init__(self, quantized_model_path="./models/quantized_grammar_checker_int8.pth"):
        # Load the quantized model directly
        self.model = torch.load(quantized_model_path)
        self.model.eval()  # Ensure the model is in evaluation mode

        gc = GrammarChecker1()
        gc.set_model(self.model)

        self.tokenizer = gc.get_tokenizer()

        print(f"Quantized model loaded from {quantized_model_path}.")


    def correct(self, text):
        input_text = "gec: " + text
        input_ids = self.tokenizer.encode(input_text, return_tensors="pt")
        outputs = self.model.generate(input_ids, max_length=512)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def name(self):
        return "Quantized Int Grammar Checker"


def evaluate_with_rouge(predictions, references):
    rouge = Rouge()
    scores = []
    for pred, ref in zip(predictions, references):
        score = rouge.get_scores(pred, ref, avg=True)['rouge-l']['f']
        scores.append(score)
    return scores


def time_execution(checkers, sentences, references, checker_type="Checker"):
    recap = {}
    for checker in checkers:
        checker_name = checker.name()
        times = []
        predictions = []
        for sentence in sentences:
            start_time = time.time()
            corrected_text = checker.correct(sentence)
            end_time = time.time()
            times.append(end_time - start_time)
            predictions.append(corrected_text)

        rouge_scores = evaluate_with_rouge(predictions, references)
        recap[checker_name] = {
            "average_time": sum(times) / len(times),
            "min_time": min(times),
            "max_time": max(times),
            "average_score": sum(rouge_scores) / len(rouge_scores),
            "corrections": predictions,
            "rouge_scores": rouge_scores,
        }
    return recap


# Define sentences and references
spelling_error_sentences = [
    "She is goinng to the libary to borrow a bok.",
    "The weather is beatiful tooday for a piknic.",
    "I recived an emale from my freind yestarday.",
    "My favorate subject in scool is mathamatics.",
    "He forgott his umbrela at home.",
    "The cat was chasing it's own tail in circls.",
    "We dicided to viset the musium this wekend.",
    "Pleese rember to bring your lunch tomorow.",
    "I tride to solve the puzzel but it was too dificult.",
    "Christmas is celbrated on decembr 25 evry ear."
]

reference_spelling = [
    "She is going to the library to borrow a book.",
    "The weather is beautiful today for a picnic.",
    "I received an email from my friend yesterday.",
    "My favorite subject in school is mathematics.",
    "He forgot his umbrella at home.",
    "The cat was chasing its own tail in circles.",
    "We decided to visit the museum this weekend.",
    "Please remember to bring your lunch tomorrow.",
    "I tried to solve the puzzle but it was too difficult.",
    "Christmas is celebrated on December 25 every year."
]

grammar_error_sentences = [
    "He are moving here.",
    "I am doing fine. How is you?",
    "How is they?",
    "Matt like fish",
    "Anna and Mike is going skiing",
    "I walk to the store and I bought milk.",
    "We all eat the fish and then made dessert.",
    "I will eat fish for dinner and drink milk.",
    "what be the reason for everyone leave the company.",
    "Christmas are celebrating on December 25 every year."
]

reference_grammar = [
    "He is moving here.",
    "I am doing fine. How are you?",
    "How are they?",
    "Matt likes fish.",
    "Anna and Mike are going skiing.",
    "I walked to the store and I bought milk.",
    "We all ate the fish and then made dessert.",
    "I will eat fish for dinner and drink milk.",
    "What is the reason for everyone leaving the company?",
    "Christmas is celebrated on December 25 every year."
]

# Benchmark spell checkers
spell_checkers = [SpellChecker1(), SpellChecker2(), SpellChecker3()]
spell_recap = time_execution(spell_checkers, spelling_error_sentences, reference_spelling, checker_type="Spell Checker")

# Benchmark grammar checkers
grammar_checkers = [GrammarChecker1(), 
                GrammarChecker1(model_path="./models/prithivida_grammar_error_correcter_v1_fp16"),
                QuantizedGrammarChecker(), 
                GrammarChecker2()]
grammar_recap = time_execution(grammar_checkers, grammar_error_sentences, reference_grammar, checker_type="Grammar Checker")

# Display results
formatter = ResultFormatter()

print("Spell Checker Benchmark:")
for checker_name, values in spell_recap.items():
    formatter.display_results(
        checker_name,
        spelling_error_sentences,
        values["corrections"],
        reference_spelling,
        values["rouge_scores"],
    )
    formatter.display_summary(
        checker_name,
        values["average_time"],
        values["min_time"],
        values["max_time"],
        values["average_score"],
    )

print("Grammar Checker Benchmark:")
for checker_name, values in grammar_recap.items():
    formatter.display_results(
        checker_name,
        grammar_error_sentences,
        values["corrections"],
        reference_grammar,
        values["rouge_scores"],
    )
    formatter.display_summary(
        checker_name,
        values["average_time"],
        values["min_time"],
        values["max_time"],
        values["average_score"],
    )
