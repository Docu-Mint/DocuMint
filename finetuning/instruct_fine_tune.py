
"""
Supervised fine tune a base model using LoRA on DocuMint dataset
Simply change the hugging-face model id for different models
"""

# Get the token from huggingface_hub
from huggingface_hub import login
import json
import torch 
import argparse
from transformers import TrainingArguments
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM

from peft import LoraConfig, PeftModel
from trl import SFTTrainer

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

        print(f"\nModel summary:\n{model}\n") # Keras style summary does not exist here
       
        print(f"Total named parameters: {sum(p.numel() for p in model.parameters())}\n")
        print(f"Total trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}\n")
        
        # generate output on dummy input
        # For regular text completion/ instruct models
        input_text = "Hello, my dog is cute. Write a poem on him"    

        # Input text specific to code models
        # For CodeGemma    
        
        input_ids = tokenizer(input_text, return_tensors="pt").to(self.device)
        output_tokens = model.generate(**input_ids, max_length= self.max_seq_len) #max_new_tokens=100)
        output_text = tokenizer.decode(output_tokens[0], skip_special_tokens=False)
        print(f"\nOutput text:\n{output_text}")

        # setup LoRA
        # TODO: 1. What should be appropriate rank value
        # TODO: 2. What should be the target modules (are these the adapters?). Are these same for every model_id?
        # Follow some techniqes from the efficient training from single GPU: https://huggingface.co/docs/transformers/v4.20.1/en/perf_train_gpu_one
        # Official Gemma Fine Tuning guide: https://huggingface.co/blog/gemma-peft

        # The formatting function
        # 
        def formatting_func(examples):
            return tokenizer(examples["instruction"], return_tensors="pt", max_length=self.max_seq_len, truncation=True)
          
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
            bf16= False, # Whether to user bf16 or not
            fp16=True, # Whether to user fp16 or not

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
            tokenizer = tokenizer,
            train_dataset = self.data, # TODO: User train_data
            args=training_args,
            peft_config=lora_config,
            formatting_func=formatting_func,
            eval_dataset = None, # TODO: User eval_data

            # 
            #packing = True, # Allows faster training
        )

        torch.cuda.empty_cache()
        trainer.train()

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
        Non-code models:
        - google/gemma-2b-it (instruction tuned, not for fine-tuning)
        - google/gemma-2b

        Code models:
        - google/codegemma-2b: No instruction tuned variant for this model. Intended for use in both code generation and completion.

    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_id', type=str, default='google/codegemma-2b', help='Model ID')
    parser.add_argument('--max_seq_len', type=int, default=256, help='Max output sequence length')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=5e-5, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=3, help='Number of epochs')
    parser.add_argument('--data_path', type=str, default='./data/databricks-dolly-15k.json', help='Path to the dataset')
    main(parser.parse_args())