import torch
from transformers import pipeline

# Load the conversational model (choose one of the following)
model_name = "microsoft/DialoGPT-small"
conversational_model = pipeline("conversation", model=model_name, framework="torch")

def converse(text):
    # Preprocess input text
    inputs = conversational_model.tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=1024,
        return_attention_mask=True,
        return_tensors='pt',
    )

    # Generate response
    output = conversational_model(**inputs)

    # Get the response from the model
    response = output['generated_text'][0]

    return response

def main():
    while True:
        user_input = input("You: ")
        if user_input.lower() == "quit":
            break
        response = converse(user_input)
        print(f"Model: {response}")

if __name__ == "__main__":
    main()
