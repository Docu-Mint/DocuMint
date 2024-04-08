import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer # , BitsAndBytesConfig

def main():
    torch.cuda.empty_cache()

    #quantization_config = BitsAndBytesConfig(load_in_8bit=True)

    checkpoint = "bigcode/starcoder2-7b"
    device = "cuda" # for GPU usage or "cpu" for CPU usage

    model = AutoModelForCausalLM.from_pretrained(checkpoint).to(device) # quantization_config=quantization_config
    # for multiple GPUs install accelerate and do `model = AutoModelForCausalLM.from_pretrained(checkpoint, device_map="auto")`


    tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    inputs = tokenizer.encode("Write documentation for the following code: def print_hello_world(): print(\"Hello world\")", return_tensors="pt").to(device)
    outputs = model.generate(inputs)
    print(tokenizer.decode(outputs[0]))

if __name__ == "__main__":
    main()




