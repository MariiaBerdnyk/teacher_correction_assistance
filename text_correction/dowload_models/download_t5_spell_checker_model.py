from transformers import T5Tokenizer, T5ForConditionalGeneration

# Load the model
tokenizer = T5Tokenizer.from_pretrained("Bhuvana/t5-base-spellchecker")
model = T5ForConditionalGeneration.from_pretrained("Bhuvana/t5-base-spellchecker")

# Save the model locally
model.save_pretrained("./models/t5_base_spellchecker_model")
tokenizer.save_pretrained("./models/t5_base_spellchecker_model")