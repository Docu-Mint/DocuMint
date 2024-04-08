import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def main():
    torch.cuda.empty_cache()

    tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-coder-6.7b-instruct", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained("deepseek-ai/deepseek-coder-6.7b-instruct", trust_remote_code=True, torch_dtype=torch.bfloat16).cuda()

    input_text = "#Write documentation for only the following function without examples: def add(a,b): return a + b"
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

    outputs = model.generate(**inputs, max_length=256)
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))

if __name__ == "__main__":
    main()


