import argparse
import json
from huggingface_hub import login
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# CodeGemma Tokens
FIM_PREFIX = '<|fim_prefix|>'
FIM_SUFFIX = '<|fim_suffix|>'
FIM_MIDDLE = '<|fim_middle|>'

# Llama3 Tokens
BEGIN_OF_TEXT = '<|begin_of_text|>'
EOT_ID = '<|eot_id|>'

START_HEADER_ID = '<|start_header_id|>'
END_HEADER_ID = '<|end_header_id|>'
END_OF_TEXT = '<|end_of_text|>'

# DeepSeek-Coder Tokens
FIM_BEGIN = '<|fim_begin|>'
FIM_END = '<|fim_end|>'

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
		system_prompt = "You are a helpful AI assistant designed to generate high-quality docstrings for Python code. Your task is to create docstrings that are:\n\nConcise: Brief and to the point, focusing on essential information.\n\nComplete: Cover functionality, parameters, return values, and exceptions.\n\nClear: Use simple language and avoid ambiguity.\n\nGenerate the docstrings in this format: \"\"\"<your generated docstring>\"\"\".\n\nPlease generate docstrings for the following functions:\n\n"


		# Accessing the nested data
		inference_data = data['inference_data']
		source_data = inference_data['source']

		# Accessing MBPP data
		mbpp_data = source_data['mbpp']['data']
		he_data = source_data['HumanEval']['data']
		apps_data = source_data['apps']['data']
		
		# Accessing description and function for each key in MBPP data
		for key, value in mbpp_data.items():
			function = value['function']
			
			if (self.model_id == 'codegemma-7b-it'):
				template = f'''{FIM_PREFIX}{system_prompt}\n\"\"\"\n{FIM_SUFFIX}\n\"\"\"\n{function}{FIM_MIDDLE}'''
			elif (self.model_id == 'meta-llama/Meta-Llama-3-8B-Instruct'):
				template = f'''{BEGIN_OF_TEXT}{system_prompt}\n\"\"\"\n\"\"\"\n{function}'''
			elif (self.model_id == 'bigcode/starcoder2-7b'):
				template = f'''{system_prompt}\n\"\"\"\n\"\"\"\n{function}'''
			else:
				template = f'''{FIM_PREFIX}{system_prompt}\n\"\"\"\n{FIM_SUFFIX}\n\"\"\"\n{function}{FIM_MIDDLE}'''
			
			self.data.append(template)
			
		for key, value in he_data.items():
			function = value['function']
			
			if (self.model_id == 'codegemma-7b-it'):
				template = f'''{FIM_PREFIX}{system_prompt}\n\"\"\"\n{FIM_SUFFIX}\n\"\"\"\n{function}{FIM_MIDDLE}'''
			elif (self.model_id == 'meta-llama/Meta-Llama-3-8B-Instruct'):
				template = f'''{BEGIN_OF_TEXT}{system_prompt}\n\"\"\"\n\"\"\"\n{function}'''
			elif (self.model_id == 'bigcode/starcoder2-7b'):
				template = f'''{system_prompt}\n\"\"\"\n\"\"\"\n{function}'''
			else:
				template = f'''{FIM_PREFIX}{system_prompt}\n\"\"\"\n{FIM_SUFFIX}\n\"\"\"\n{function}{FIM_MIDDLE}'''
			
			self.data.append(template)
		
		for key, value in apps_data.items():
			function = value['function']
			
			if (self.model_id == 'codegemma-7b-it'):
				template = f'''{FIM_PREFIX}{system_prompt}\n\"\"\"\n{FIM_SUFFIX}\n\"\"\"\n{function}{FIM_MIDDLE}'''
			elif (self.model_id == 'meta-llama/Meta-Llama-3-8B-Instruct'):
				template = f'''{BEGIN_OF_TEXT}{system_prompt}\n\"\"\"\n\"\"\"\n{function}'''
			elif (self.model_id == 'bigcode/starcoder2-7b'):
				template = f'''{system_prompt}\n\"\"\"\n\"\"\"\n{function}'''
			else:
				template = f'''{FIM_PREFIX}{system_prompt}\n\"\"\"\n{FIM_SUFFIX}\n\"\"\"\n{function}{FIM_MIDDLE}'''
			
			self.data.append(template)
		
	def generate_text(self):
		torch.cuda.empty_cache()
		
		print(f"\nLoading model {self.model_id} and tokenizer...\nThis may take a while...\n")
        # Load model and tokenizer
		model = AutoModelForCausalLM.from_pretrained(self.model_id, device_map = self.device, token = self.access_token)
		tokenizer = AutoTokenizer.from_pretrained(self.model_id)
		
		for prompt in self.data:
		
			input_ids = tokenizer(prompt, return_tensors="pt").to(self.device)
			output_tokens = model.generate(**input_ids, max_length=self.max_seq_len)
			output_text = tokenizer.decode(output_tokens[0], skip_special_tokens=False)
			
			target = open("output.txt", "a")
			target.write("%s\n\n" % (output_text))
			target.close()

			print("Output generated...")
		print("Done!")

def main(args):
	ds_gen = DocstringGen(args.model_id, args.max_seq_len, args.data_path)
	
	ds_gen.login()
	ds_gen.load_dataset()
	ds_gen.generate_text()
	
if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	# google/codegemma-7b-it
	# meta-llama/Meta-Llama-3-8B-Instruct,
	# bigcode/starcoder2-7b,
	# deepseek-ai/deepseek-coder-6.7b-instruct
	parser.add_argument('--model_id', type=str, default='google/codegemma-7b-it', help='Model ID')
	parser.add_argument('--max_seq_len', type=int, default=256, help='Max output sequence length') # max_new_tokens
	parser.add_argument('--data_path', type=str, default='example.json', help='JSON data file')
	main(parser.parse_args())