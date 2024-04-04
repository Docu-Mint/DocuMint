import argparse

# NOTE:
# Most of this code is placeholder for parsing arguments and running SLMs through this main Python file.
# Code subject to frequent change.

parser = argparse.ArgumentParser(description='Docu-mint')

parser.add_argument('-model', type=str,
                    help='gemma, codellama, starcoder, deepseek',
					default="gemma")
					
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