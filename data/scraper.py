import json
import random

# Function to extract text and code from each item
def extract_text_and_code(item, filename):
    if filename == 'mbpp.jsonl':
      task_id = item.get("task_id", "")
        # Check if the task ID falls within the desired range
      if task_id >= 11 and task_id <= 510:
        return {"description": item.get("text", ""), "function": item.get("code", "")}
    if filename == 'HumanEval.jsonl':
      return {"description": item.get("prompt", ""), "function": item.get("canonical_solution", "")}

jsonl_files = ['mbpp.jsonl', 'HumanEval.jsonl']
inference_data = {"inference_data": {"source": {}}}

for filename in jsonl_files:
    source = filename.split('.')[0]
    source_data = {"data": {}}
    
    # Open the JSONL file and read each line
    data = []
    with open(filename, 'r') as file:
        for line in file:
            item = json.loads(line)
            # Extract text and code elements
            extracted_item = extract_text_and_code(item, filename)
            if extracted_item is not None:  # Filter out None values
                data.append(extracted_item)
    
    # Randomly select 5 items
    random_items = random.sample(data, 10)
    
    # Add selected items to source_data dictionary
    for i, item in enumerate(random_items):
        source_data["data"][str(i+1)] = item
    
    # Add source_data to inference_data dictionary
    inference_data["inference_data"]["source"][source] = source_data

# Write the selected items to a new JSON file
with open('inference_data3.json', 'w') as outfile:
    json.dump(inference_data, outfile, indent=2)

