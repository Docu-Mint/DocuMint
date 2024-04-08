import argparse
import json
import torch
from huggingface_hub import login

import gemma
import codellama
import starcoder
import deepseek

# NOTE:
# Most of this code is placeholder for parsing arguments and running SLMs through this main Python file.
# Code subject to frequent change.

login()

torch.cuda.empty_cache()

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
		gemma.main()
	case "codellama":
		print("Run inference on Codellama")
		codellama.main()
	case "starcoder":
		print("Run inference on StarCoder")
		starcoder.main()
	case "deepseek":
		print("Run inference on DeepSeek")
		deepseek.main()
	case _:
		print("Error: model not recognized")
		
data_file = open(args.data, encoding="utf8")
data = json.load(data_file)

num_funcs = len(data["inference_data"]["data"])
print("Number of input functions:", num_funcs)
	
data_file.close()