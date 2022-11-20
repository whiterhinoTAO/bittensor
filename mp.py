import bittensor as bt

from transformers import AutoModelForCausalLM
from parallelformers import parallelize


model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-125M")
model = parallelize(model, num_gpus=4)

tokenizer = bt.tokenizer()

inputs_ids = tokenizer.encode("Hello, my dog is cute", return_tensors="pt")

outputs = model(inputs_ids)
print(outputs)