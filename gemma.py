from transformers import GemmaModel, GemmaConfig
from transformers import GemmaTokenizerFast
from transformers import AutoModelForCausalLM

def main():
    access_token = "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"

    tokenizer = GemmaTokenizerFast.from_pretrained("hf-internal-testing/dummy-gemma", token = access_token)
    tokenizer.encode("Hello this is a test")

    # Initializing a Gemma gemma-7b style configuration
    configuration = GemmaConfig()

    # Initializing a model from the gemma-7b style configuration
    model = GemmaModel(configuration)

    # Accessing the model configuration
    configuration = model.config

    # Download and load model
    model = AutoModelForCausalLM.from_pretrained("google/gemma-7b-it", device_map="cuda", token = access_token)

    input_text = "Write me a poem about Machine Learning."
    input_ids = tokenizer(input_text, return_tensors="pt").to("cuda")

    outputs = model.generate(**input_ids)

    print(outputs)
    print(tokenizer.decode(outputs[0]))

if __name__ == "__main__":
    main()