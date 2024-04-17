"""
Data preparation on the json file
- Tokenize, Truncate and Pad
    - Since we did not use packing, need to perform padding so that the sequences of different lengths can be batched together
    - see: https://wandb.ai/capecape/alpaca_ft/reports/How-to-Fine-Tune-an-LLM-Part-1-Preparing-a-Dataset-for-Instruction-Tuning--Vmlldzo1NTcxNzE2#second-option:-batching-multiple-sequences-of-different-lengths
- Split the dataset into train and test

# TODO: output safetensors of the model
# TODO: W&B logs?
"""
import json
import datasets

def add_prompt_format(data, model_id="codegemma"):
    """
    Add the required prompt format (varies according to model)
    For CodeGemma, essentially the setup is for infilling.

    Where:
    <|fim_suffix|> is where you expect the cursor to be
    <|fim_prefix|>  indicates content above the cursor
    <|fim_middle|> indicates context after the cursor. 

    See: https://huggingface.co/blog/codegemma#using-transformers

    go through all the instructions and add the the formatting
    """

    if "codegemma" in model_id:
        formatted_data = []
        for item in data:
            instruction = item["instruction"]
            response = item["response"]
            
            # Split the instruction into lines
            instruction_lines = instruction.split("\n")
            
            # Extract the function definition line
            function_def = instruction_lines[0]
            
            # Join the remaining lines of the instruction
            instruction_body = "\n".join(instruction_lines[1:])
            
            # Format the instruction and response
            formatted_item = {
                "instruction": f"<|fim_prefix|>{function_def}\n\"\"\"\n<|fim_suffix|>\n\"\"\"\n{instruction_body}<|fim_middle|>",
                "response": response
            }
            
            formatted_data.append(formatted_item)
        
        return formatted_data
    else:
        # If model_id is not "codegemma", return the original data without formatting
        return data

def tokenize_truncate_pad(batch_data, tokenizer, max_length):
    """
    Tokenize: Concatenate the insturction and the response to tokenize
    Truncate: From the right
    Pad: To the max_length by adding eos token
    """
    # If I take a for loop in a dictionary (batch_data), it will go through the keys
    # Each batch has 2 keys, but the keys themselves are lists

    assert(len(batch_data['response'])==len(batch_data['instruction']))

    # Essentially end of sequence token indicates model to stop generation after that point but here we already have the <> tokens
    # So it makes sense to add the eos token at the end of the concatenated text
    concatenated_text= ""
    for i in range(len(batch_data['instruction'])):
        concatenated_text+= batch_data['instruction'][i] + batch_data['response'][i] + tokenizer.eos_token 
        
    # print(f"Concatenated Text: {concatenated_text}")

    tokenizer.pad_token = tokenizer.eos_token
    tokenized_inputs = tokenizer(
        concatenated_text,
        return_tensors="pt",
        padding = True
    )

    # Whichever is the smallest, the max_length or the length of the tokenized input
    max_length = min(tokenized_inputs["input_ids"].shape[1], max_length)
    tokenizer.truncation_side = "left" # Left part is less important (because its the instruction side)
    tokenizer.paddding_side = "right" # Pad on the right side
    tokenized_inputs = tokenizer(
        concatenated_text,
        max_length=max_length,
        return_tensors="pt",
        truncation=True
    )

    # print(f"Tokenized inputs: {tokenized_inputs}")
    return tokenized_inputs

def preprocess(data_path, tokenizer, max_length, batch_size):
    """
    Sequentially perform the steps
    """
    # Load data, add prompt format and save it as json again
    raw_data = datasets.load_dataset("json", data_files=data_path, split="train") # There is no test split so far
    formatted_data = add_prompt_format(raw_data)

    # Convert the list to JSON
    json_data = json.dumps(formatted_data, indent=2)
    # Save the JSON data to a file
    with open("./data/formatted_data.json", "w") as file:
        file.write(json_data)

    data = datasets.load_dataset("json", data_files='./data/formatted_data.json', split="train") # There is no test split so far

    # Illustrate with an example before and after the process
    print(f"\n\nExample before tokenization:\nInstruction:\n{data[0]['instruction']}\n\nResponse:\n{data[0]['response']}")
    tokenized_instruction = tokenizer(data[0]['instruction'], padding="max_length", truncation=True, max_length=max_length)['input_ids']
    tokenized_response = tokenizer(data[0]['response'], padding="max_length", truncation=True, max_length=max_length)['input_ids']
    print(f"\nExample tokenized:\nInstruction:\n{tokenized_instruction, len(tokenized_instruction)}\n\nResponse:\n{tokenized_response,  len(tokenized_response)}\n")
    
    # Return the tokenized data
    tokenized_data = data.map(
        tokenize_truncate_pad,
        batched=True,
        batch_size=batch_size,
        drop_last_batch=True,
        fn_kwargs={"tokenizer": tokenizer, "max_length": max_length}
    )

    print(f"\nAfter pre-processing:\n{tokenized_data}, {type(tokenized_data)}\n")

    return tokenized_data