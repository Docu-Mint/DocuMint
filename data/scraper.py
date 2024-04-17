import json
import random

# get one function from the apps list of solutions
def extract_function(solutions):
    # Remove the outer square brackets and double quotes
    solutions = solutions.strip("[]\"")
    
    # Split the input string into individual functions
    functions = solutions.split('\\n",')
    
    # Remove any remaining leading/trailing whitespace and quotes
    functions = [func.strip(' "\n') for func in functions]
    
    # Find the first function that starts with "def" and exclude class objects
    for func in functions:
        lines = func.split('\n')
        if any(line.startswith('def ') for line in lines) and not any(line.startswith('class ') for line in lines):
            return func
    
    return None

# Function to extract text and code from each item
def extract_text_and_code(item, filename):
    if filename == 'mbpp.jsonl':
      task_id = item.get("task_id", "")
        # Check if the task ID falls within the desired range
      if task_id >= 11 and task_id <= 510:
        return {"task_id": item.get("task_id", ""), "description": item.get("text", ""), "function": item.get("code", "").strip()}
      
    if filename == 'HumanEval.jsonl':
      function_header = item.get("canonical_solution", "")
      function_header = function_header.strip()
      if function_header.startswith('def '):
        return {"task_id": item.get("task_id", ""), "description": item.get("prompt", ""), "function": item.get("canonical_solution", "").strip()}
      
    if filename == 'apps.jsonl':
      difficulty = item.get("difficulty", "")
      solutions = item.get("solutions", "")
      result = extract_function(solutions)
      if difficulty == "interview" and result:
        return {"task_id": item.get("id", ""), "description": item.get("question", ""), "function": result.strip()}

jsonl_files = ['mbpp.jsonl', 'HumanEval.jsonl', 'apps.jsonl']
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
    
    random.seed(0)
    random_items = random.sample(data, 7)
    
    # Add selected items to source_data dictionary
    for i, item in enumerate(random_items):
        source_data["data"][str(i+1)] = item
    
    # Add source_data to inference_data dictionary
    inference_data["inference_data"]["source"][source] = source_data

# Write the selected items to a new JSON file
with open('inference_data5.json', 'w') as outfile:
    json.dump(inference_data, outfile, indent=2)

