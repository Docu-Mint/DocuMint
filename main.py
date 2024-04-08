import argparse
import json

# NOTE:
# Most of this code is placeholder for parsing arguments and running SLMs through this main Python file.
# Code subject to frequent change.

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