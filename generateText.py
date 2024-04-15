import argparse
import json
from huggingface_hub import login
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# NOTE:
# Most of this code is placeholder for parsing arguments and running SLMs through this main Python file.
# Code subject to frequent change.
"""
parser = argparse.ArgumentParser(description='Docu-mint')

parser.add_argument('-model', type=str,
                    help='gemma, codellama, starcoder, deepseek',
					default="")
parser.add_argument('-data', type=str,
					help='NAME.json',
					default="")
					
args = parser.parse_args()

match args.model:
	case "gemma":
		print("Run inference on Gemma")
	case "codellama":
		print("Run inference on Codellama")
	case "starcoder":
		print("Run inference on StarCoder")
	case "deepseek":
		print("Run inference on DeepSeek")
	case _:
		print("Error: model not recognized")
		
data_file = open(args.data, encoding="utf8")
data = json.load(data_file)

for i in data["inference_data"]["source"]:
	print(i)
	
data_file.close()
"""

class DocstringGen:
	def __init__(self, model_id:str, max_seq_len:int, data_path:str):
		self.model_id = model_id
		self.max_seq_len = max_seq_len
		self.data_path = data_path
		
		self.data = []
		self.access_token = None
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		
	def login(self):
		login()
		print("\nEnter access token:")
		self.access_token = input()
		print(f"Obtained access token: {self.access_token}")

	def load_dataset(self):
		print(f"\nLoading dataset from {self.data_path}...")
		
		# Read the JSON file
		with open(self.data_path, 'r') as f:
			data = json.load(f)

		# Set system prompt
		system_prompt = "Your task is to create docstrings that are:\n\nConcise: Brief and to the point, focusing on essential information.\n\nComplete: Cover functionality, parameters, return values, and exceptions.\n\nClear: Use simple language and avoid ambiguity."


		# Accessing the nested data
		inference_data = data['inference_data']
		source_data = inference_data['source']

		# Accessing MBPP data
		mbpp_data = source_data['mbpp']['data']
		he_data = source_data['HumanEval']['data']
		#mbpp_plus__data = source_data['MBPP+']['data']
		#he_plus_data = source_data['HumanEval+']['data']
		
		# Accessing description and function for each key in MBPP data
		for key, value in mbpp_data.items():
			description = value['description']
			function = value['function']
			template = "{sp}\n\nDescription:\n{description}\n\nFunction:\n{function}"
			self.data.append(template.format(sp=system_prompt, description=description, function=function))
			
		for key, value in he_data.items():
			description = value['description']
			function = value['function']
			template = "{sp}\n\nDescription:\n{description}\n\nFunction:\n{function}"
			self.data.append(template.format(sp=system_prompt, description=description, function=function))
		
		#print(self.data[0])
		
	def generate_text(self):
		torch.cuda.empty_cache()
		
		print(f"\nLoading model {self.model_id} and tokenizer...\nThis may take a while...\n")
        # Load model and tokenizer
		model = AutoModelForCausalLM.from_pretrained(self.model_id, device_map = self.device, token = self.access_token)
		tokenizer = AutoTokenizer.from_pretrained(self.model_id)
		
		for prompt in self.data:
			#input_text = "Write me a poem about machine learning."
		
			input_ids = tokenizer(prompt, return_tensors="pt").to(self.device)
			output_tokens = model.generate(**input_ids, max_length=self.max_seq_len)
			output_text = tokenizer.decode(output_tokens[0], skip_special_tokens=False)
			print(f"\nOutput text:\n{output_text}")
		

def main(args):
	ds_gen = DocstringGen(args.model_id, args.max_seq_len, args.data_path)
	
	ds_gen.login()
	ds_gen.load_dataset()
	ds_gen.generate_text()
	
if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	# google/codegemma-7b-it
	# codellama/CodeLlama-7b-Instruct-hf,
	# bigcode/starcoder2-7b,
	# deepseek-ai/deepseek-coder-6.7b-instruct
	parser.add_argument('--model_id', type=str, default='google/codegemma-7b-it', help='Model ID')
	parser.add_argument('--max_seq_len', type=int, default=256, help='Max output sequence length') # max_new_tokens
	parser.add_argument('--data_path', type=str, default='example.json', help='JSON data file')
	main(parser.parse_args())