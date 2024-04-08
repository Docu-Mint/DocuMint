from transformers import LlamaForCausalLM, CodeLlamaTokenizer


def main():
    access_token = "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" # <-- YOUR ACCESS TOKEN HERE

    tokenizer = CodeLlamaTokenizer.from_pretrained("codellama/CodeLlama-7b-Instruct-hf")
    model = LlamaForCausalLM.from_pretrained("codellama/CodeLlama-7b-Instruct-hf")

    PROMPT = '''Write documentation for the following code:
            def add(a, b):
                return a + b
    '''

    input_ids = tokenizer(PROMPT, return_tensors="pt")["input_ids"]
    generated_ids = model.generate(input_ids, max_new_tokens=128)

    output = tokenizer.batch_decode(generated_ids[:, input_ids.shape[1]:], skip_special_tokens = True)[0]
    print(output)

if __name__ == "__main__":
    main()



