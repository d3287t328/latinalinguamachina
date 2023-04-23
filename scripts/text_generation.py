# In this script, the generate_text function takes as input the trained language model, the tokenizer used to preprocess the input, the prompt to generate text from, the maximum length of the generated text, and the decoding strategy to use (greedy, beam, top-k, top-p, or temperature). The num_beams, top_k, top_p, and temperature arguments are only used for certain decoding strategies.

# The function first encodes the prompt using the tokenizer and sends the resulting input IDs to the model's device (CPU or GPU). It then generates the output using the specified decoding strategy and parameters, and decodes the resulting output IDs using the tokenizer. The generated text is returned as a string.

# In the example usage, the script loads the pre-trained GPT-2 model and tokenizer from the Hugging Face Transformers library, and generates text using the greedy decoding strategy. You can experiment with different prompts and decoding strategies by modifying the arguments passed to generate_text.

# Note that this is just an example, and you may want to modify the script to suit your specific use case. For example, you may want to add more advanced features like prompt engineering, conditioning the model on external data, or fine-tuning the model on specific tasks. 

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

def generate_text(model, tokenizer, prompt, max_length, decoding_strategy='greedy', num_beams=1, top_k=0, top_p=1.0, temperature=1.0):
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    input_ids = input_ids.to(model.device)

    # Generate the output using the specified decoding strategy
    if decoding_strategy == 'greedy':
        output = model.generate(input_ids=input_ids, max_length=max_length, pad_token_id=tokenizer.eos_token_id)
    elif decoding_strategy == 'beam':
        output = model.generate(input_ids=input_ids, max_length=max_length, pad_token_id=tokenizer.eos_token_id, num_beams=num_beams)
    elif decoding_strategy == 'top-k':
        output = model.generate(input_ids=input_ids, max_length=max_length, pad_token_id=tokenizer.eos_token_id, do_sample=True, top_k=top_k)
    elif decoding_strategy == 'top-p':
        output = model.generate(input_ids=input_ids, max_length=max_length, pad_token_id=tokenizer.eos_token_id, do_sample=True, top_p=top_p)
    elif decoding_strategy == 'temperature':
        output = model.generate(input_ids=input_ids, max_length=max_length, pad_token_id=tokenizer.eos_token_id, do_sample=True, temperature=temperature)
    else:
        raise ValueError(f"Invalid decoding strategy '{decoding_strategy}'")

    return tokenizer.decode(output[0], skip_special_tokens=True)

# Example usage
model_name = 'gpt2'
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

prompt = "Hello, how are you today?"
generated_text = generate_text(model, tokenizer, prompt, max_length=100, decoding_strategy='greedy')
print(generated_text)
