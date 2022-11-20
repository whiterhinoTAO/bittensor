import pdb
import bittensor as bt

from transformers import AutoModelForCausalLM


# if __name__ == "__main__":
model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-125M")

tokenizer = bt.tokenizer()

def split_models(model, num_gpus: int):
    pdb.set_trace()



if __name__ == "__main__":
    split_models(model, 1)