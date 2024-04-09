
"""
Supervised fine tune a base model using LoRA on DocuMint dataset
Simply change the hugging-face model id for different models
"""

# Get the token from huggingface_hub
from huggingface_hub import login
import json
import torch 
import argparse
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM

class FineTuneInstructor:
    def __init__(self, 
                 model_id:str, 
                 max_seq_len:int,
                 batch_size:int, 
                 learning_rate:float, 
                 epochs:int,
                 data_path:str):
        
        self.model_id = model_id
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.data_path = data_path

        self.data = [] 
        self.access_token = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"\nDevice: {self.device}\n")

    def login(self):
        login()
        print("\nEnter access token again:")
        self.access_token = input()
        print(f"Obtained access token: {self.access_token}")

    def load_dataset(self):
        """
        For now, using the databricks-dolly-15k dataset
        """
        print(f"\nLoading dataset from {self.data_path}...")

        # TODO: Make the data loading version for our data
        with open(self.data_path) as f:
            for line in f:
                features = json.loads(line)
                if features["context"]:
                    continue
                template = "Instruction:\n{instruction}\n\nResponse:\n{response}"
                self.data.append(template.format(**features))
        
        print(f"Loaded {len(self.data)} examples")

    def fine_tune(self):
        # clear cache
        torch.cuda.empty_cache()
        
        print(f"\nLoading model {self.model_id} and tokenizer...\nThis may take a while...\n")
        # Load model and tokenizer
        model = AutoModelForCausalLM.from_pretrained(self.model_id, device_map = self.device, token = self.access_token)
        tokenizer = AutoTokenizer.from_pretrained(self.model_id)


        # generate output on dummy input
        input_text = "Hello, my dog is cute. Write a poem on him"
        input_ids = tokenizer(input_text, return_tensors="pt").to(self.device)
        output_tokens = model.generate(**input_ids, max_length= self.max_seq_len)
        output_text = tokenizer.decode(output_tokens[0], skip_special_tokens=False)
        print(f"\nOutput text:\n{output_text}")

def main(args):
    # Define the fine tune instructor
    fine_tune_instructor = FineTuneInstructor(args.model_id, 
                                              args.max_seq_len,
                                              args.batch_size, 
                                              args.learning_rate, 
                                              args.epochs,
                                              args.data_path)

    fine_tune_instructor.login()
    fine_tune_instructor.load_dataset()
    fine_tune_instructor.fine_tune()


if __name__ == '__main__':
    """
    Relevant model ids:
    - google/gemma-2b-it (instruction tuned, not for fine-tuning)
    - google/gemma-2b
    - google/codegemma-2b (No instruction tuned variant for this model)

    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_id', type=str, default='google/codegemma-2b', help='Model ID')
    parser.add_argument('--max_seq_len', type=int, default=256, help='Max output sequence length')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=5e-5, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=3, help='Number of epochs')
    parser.add_argument('--data_path', type=str, default='./data/databricks-dolly-15k.json', help='Path to the dataset')
    main(parser.parse_args())