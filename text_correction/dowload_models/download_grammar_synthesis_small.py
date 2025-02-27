# Load model directly
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("pszemraj/grammar-synthesis-small")
model = AutoModelForSeq2SeqLM.from_pretrained("pszemraj/grammar-synthesis-small")

# Save the model locally
model.save_pretrained("./models/grammar-synthesis-small")
tokenizer.save_pretrained("./models/grammar-synthesis-small")