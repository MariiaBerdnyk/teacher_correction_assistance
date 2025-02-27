from transformers import BartTokenizer, BartForConditionalGeneration

# Load the model
tokenizer = BartTokenizer.from_pretrained("oliverguhr/spelling-correction-english-base")
model = BartForConditionalGeneration.from_pretrained("oliverguhr/spelling-correction-english-base")

# Save the model locally
model.save_pretrained("./models/spelling-correction-english-base")
tokenizer.save_pretrained("./models/spelling-correction-english-base")