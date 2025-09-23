from transformers import AutoTokenizer, AutoModelForQuestionAnswering

def load_model_and_tokenizer():
    """
    This function loads a QA model and its corresponding tokenizer
    from Hugging Face and returns them
    """
    # Define de model's name that will be used
    # 'pierreguillou/bert-base-cased-squad-v1.1-portuguese' is a model
    # trained for for the task of Question-Answering in Portuguese
    model_name = "pierreguillou/bert-base-cased-squad-v1.1-portuguese"

    print(f"Starting tokenizer download for the model: '{model_name}'...\n")
    try:
        # Loads the tokenizer associated to the model
        # The tokenizer prepares the data for the model (text -> numbers)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        print("Tokenizer successfully loaded.\n")

        print(f"Starting model download: '{model_name}'...\n")
        # Loads the pretrained model
        # This is the "brain" that will proccess the information
        model = AutoModelForQuestionAnswering.from_pretrained(model_name)
        print("Model successfully loaded.\n")

        return model, tokenizer
    
    except Exception as e:
        print(f"An error occurred during loading: {e}\n")
        return None, None
    
def test_tokenizer(tokenizer):
    """
    Simple function to demonstrate what the tokenizer does
    """
    if tokenizer:
        example_sentence = "A UFRGS fica em Porto Alegre."
        print(f"--- Testing the Tokenizer ---\n")
        print(f"Original sentence: '{example_sentence}'\n")

        # The tokenization proccess converts the sentence into numeric IDs
        coded_tokens = tokenizer.encode_plus(example_sentence)

        print(f"Token IDs (input_ids): {coded_tokens['input_ids']}\n")
        print("These are the numberes that the model really sees.")

if __name__ == "__main__":
    loaded_model, loaded_tokenizer = load_model_and_tokenizer()

    if loaded_model and loaded_tokenizer:
        print("Environment set and loading test finished.\n")
        test_tokenizer(loaded_tokenizer)