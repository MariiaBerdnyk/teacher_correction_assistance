from transformers import T5Tokenizer, T5ForConditionalGeneration

def download_model():
    model_name = "prithivida/grammar_error_correcter_v1"

    # Download tokenizer and model
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)

    # Save locally
    model.save_pretrained("./models/prithivida_grammar_error_correcter_v1")
    tokenizer.save_pretrained("./models/prithivida_grammar_error_correcter_v1")
    print("Model saved locally at ./models/prithivida_grammar_error_correcter_v1")

if __name__ == "__main__":
    download_model()
