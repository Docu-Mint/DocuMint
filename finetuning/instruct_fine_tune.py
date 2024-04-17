
"""
Supervised fine tune a base model using LoRA on DocuMint dataset
Simply change the hugging-face model id for different models
"""

# Get the token from huggingface_hub
from huggingface_hub import login
import json
import torch 
import argparse
import numpy as np
import pandas as pd
from data_preparation import preprocess
from transformers import TrainingArguments
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM

from peft import LoraConfig, PeftModel
from trl import SFTTrainer

class FineTuneInstructor:
    def __init__(self, args):
                 
        self.model_id = args.model_id
        self.max_seq_len = args.max_seq_len
        self.batch_size = args.batch_size
        self.learning_rate = args.learning_rate
        self.epochs = args.epochs
        self.data_path = args.data_path
        self.test_split = args.test_split
        self.train_data = None
        self.test_data = None
        self.access_token = None
        self.model_path = args.model_path

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"\nDevice: {self.device}\n")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.tokenizer.padding_side = 'right' # to prevent warnings

    def login(self):
        login()
        print("\nEnter access token again:")
        self.access_token = input()
        print(f"Obtained access token: {self.access_token}")

    def load_dataset(self):
        """
        Data for IT can be of any expressed as pairs of (input, output), (question, answer), or (instruction, response)
        """
        print(f"\nLoading dataset from {self.data_path}...")

        # Pre-process and tokenizethe dataset 
        tokenized_data = preprocess(self.data_path, self.tokenizer, self.max_seq_len, self.batch_size)

        split_data = tokenized_data.train_test_split(test_size= self.test_split, shuffle=True, seed=123)
        print(f"\nAfter split:")
        self.train_data = split_data["train"]
        self.test_data = split_data["test"]

        print(f"\nTrain dataset: {self.train_data}\nTest dataset: {self.test_data}")

        # Count total training tokens (multiply with epochs to get the final total training tokens count)
        count = np.sum([len(item) for item in self.train_data['input_ids']])
        print(f"Total training tokens: {self.epochs*count}")

    def fine_tune(self):
        # clear cache
        torch.cuda.empty_cache()

        print(f"\nLoading model {self.model_id} and tokenizer...\nThis may take a while...\n")
        # Load model and tokenizer
        model = AutoModelForCausalLM.from_pretrained(self.model_id, device_map = self.device, token = self.access_token)
        

        # The models used here will base models and not necessarily are for chat. Promot format is not applicable ?
        # print(f"Default chat template for tokenizer: {tokenizer.default_chat_template}\n")

        print(f"\nModel summary:\n{model}\n") # Keras style summary does not exist here
       
        print(f"Total named parameters: {sum(p.numel() for p in model.parameters())}\n")
        print(f"Total trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}\n")
        
        # generate output on dummy input
        # For regular text completion/ instruct models
        # input_text = "Hello, my dog is cute. Write a poem on him"      
        # input_ids = self.tokenizer(input_text, return_tensors="pt").to(self.device)
        # output_tokens = model.generate(**input_ids, max_length= self.max_seq_len) #max_new_tokens=100)
        # output_text = self.tokenizer.decode(output_tokens[0], skip_special_tokens=False)
        # print(f"\nOutput text:\n{output_text}")

        # setup LoRA
        # TODO: 1. What should be appropriate rank value
        # TODO: 2. What should be the target modules (are these the adapters?). Are these same for every model_id?
        # Follow some techniqes from the efficient training from single GPU: https://huggingface.co/docs/transformers/v4.20.1/en/perf_train_gpu_one
        # Official Gemma Fine Tuning guide: https://huggingface.co/blog/gemma-peft

        # The formatting function
        # Relevant because we fine-tune a base model
        # Alpaca pompt template (Sharon Zhou)
        # There are several ways to create a dataset: https://www.philschmid.de/fine-tune-llms-in-2024-with-trl
        # https://huggingface.co/docs/trl/en/sft_trainer specifies a formatting function as returning a list but in practice this gives an error

        def formatting_func(examples):
            # print(f"Formatting function called with {examples} examples")
            output_texts = []
            for i in range(len(examples['instruction'])):
                text = f"### Instruction:\n{examples['instruction'][i]}\n### Response:\n{examples['response'][i]}"
                output_texts.append(text)
            return output_texts
          
        lora_config = LoraConfig(

            # Rank, LoRA attention dimension
            r = 64, # , higher is better at the cost of more memory
            
            # The alpha parameter for LoRA scaling?
            lora_alpha = 16, # default 8

            # The dropout rates for LoRA layers
            lora_dropout = 0.1, # default 0.0

            # Bias of which layers to update during training.  Can be ‘none’, ‘all’ or ‘lora_only’.
            bias = "none", # default

            # The names of modules to which adapters will be applied
            # In the attention of the model, the linear layers have the projections (k, v, q: key, value, query) on following names
            target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
            
            task_type="CAUSAL_LM",
            )
        
        lora_model = PeftModel(model, lora_config)
        # Total Lora parameters to be trained
        print(f"Total LoRA parameters: {sum(p.numel() for _,p in lora_model.named_parameters() if p.requires_grad)}\n")

        training_args = TrainingArguments(
            # By default uses 32 bit float. 
            bf16= False, # Whether to user bf16 or not, b stands for brain 
            fp16= False, # Whether to user fp16 or not

            # The batch size per GPU
            per_device_train_batch_size=2, # See GPU usage and adjust. For a 2B model, 3090 should support high (default 8).

            # How many batches to accumulate before doing a backward pass. Higher means more memory but less time
            gradient_accumulation_steps=128,
            
            # Number of steps used for a linear warmup from 0 to learning_rate.
            warmup_steps=2, #default 0
            
            # the total number of training steps to perform. Overrides num_train_epochs.
            # max_steps=10,
            
            # The initial learning rate for AdamW optimizer.
            learning_rate=2e-4,
            
            # The number of epochs to train the model.
            num_train_epochs=self.epochs,

            #  "no": No logging is done during training. "epoch": Logging is done at the end of each epoch. "steps": Logging is done every logging_steps.
            logging_strategy = "steps", # default "steps"

            # Number of update steps between two logs if logging_strategy="steps"
            logging_steps=1, # default 500

            # The output directory where the model predictions and checkpoints will be written.
            output_dir="./fine_tuning_outputs/",

            # Overwrite the content of the output directory.
            overwrite_output_dir = True, # default False

            # The string identifier of the optimizer to use.
            # The standard AdamW optimizer takes as much memory as the model itself.
            # The adamw_8bit uses quantized 8-bit gradients.
            # OMG, CUDA supports unified memory (not available in PyTorch as of now, but available in bitsandbytes
            # paged means when GPU memory runs out, it will be copied to CPU memory
            # Using a higher bit is probably not a good idea as it be slow 
            optim="paged_adamw_8bit", # default adamw_torch

            seed = 42,
        )

        # TODO: whether or not to do eval
        # There is an option to train the model only on completions i.e to ignore the generation of instructions
        trainer = SFTTrainer(
            model = model, 
            tokenizer = self.tokenizer,
            train_dataset = self.train_data,
            args=training_args,
            peft_config=lora_config,
            formatting_func=formatting_func, # Formatting func needs to be passed. Although we already did a lot of formatting. still need to specify which one is instruction and which is response
            eval_dataset = self.test_data, 
            max_seq_length = self.max_seq_len,# Defaults to 1024

            # Packing of the dataset, allows faster training 
            # What is the operation? multiple short examples are packed in the same input sequence to increase training efficiency
            packing = False, # https://wandb.ai/capecape/alpaca_ft/reports/How-to-Fine-Tune-an-LLM-Part-1-Preparing-a-Dataset-for-Instruction-Tuning--Vmlldzo1NTcxNzE2#packing:-combining-multiple-samples-into-a-longer-sequence
            # We are not using packing here, instesad we pad the sequences to the same length 

            # Special tokens are essential for chat format but not here, we add the special tokens for code completion in pre-processing step
            dataset_kwargs={
                "add_special_tokens": False,  # We template with special tokens
                "append_concat_token": False, # No need to add additional separator token
                        }
            )

        torch.cuda.empty_cache()
        trainer.train()
        
        trainer.save_model(f"{self.model_path}")
        print(f"\nModel saved to: {self.model_path}")
        print(f"\nFine-tuning complete!\n")

    def sample_output(self, model, input_text):
        """
        Sample output on a fine-tuned model
        """
        pass 

def main(args):
    # Define the fine tune instructor
    fine_tune_instructor = FineTuneInstructor(args)

    fine_tune_instructor.login()
    fine_tune_instructor.load_dataset()
    fine_tune_instructor.fine_tune()


if __name__ == '__main__':
    """
    Relevant model ids:
        Non-code models:
        - google/gemma-2b-it (instruction tuned, not for fine-tuning)
        - google/gemma-2b

        Code models:
        - google/codegemma-2b: No instruction tuned variant for this model. Intended for use in both code generation and completion.

    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_id', type=str, default='google/codegemma-2b', help='Model ID')
    parser.add_argument('--max_seq_len', type=int, default=256, help='Max output sequence length')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=5e-5, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--data_path', type=str, default='./data/sample_data.json', help='Path to the dataset')
    parser.add_argument('--model_path', type=str, default='./fine_tuning_outputs/fine_tuned_model', help='Path to save the fine-tuned model')
    parser.add_argument('--test_split', type=float, default=0.1, help='Test split ratio')
    main(parser.parse_args())