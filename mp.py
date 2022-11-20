import bittensor as bt

from transformers import AutoModelForCausalLM
from parallelformers import parallelize


if __name__ == "__main__":
    model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-125M")
    parallelize(model, num_gpus=4, fp16=True)

    tokenizer = bt.tokenizer()

    inputs_ids = tokenizer.encode("Hello, my dog is cute", return_tensors="pt")

    outputs = model(inputs_ids)
    print(outputs)